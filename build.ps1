<#
.SYNOPSIS
    Builds FluidSimCpp.cpp with maximum MSVC optimization (the -O3 equivalent).

.DESCRIPTION
    Locates Visual Studio via vswhere, imports the MSVC environment and compiles
    the single translation unit directly with cl.exe - no .sln, no MSBuild.

    MSVC has no /O3 switch. The closest equivalent to gcc/clang -O3 is:
        /O2 /Ob3 /Oi /Ot /GL /Gy /Gw /fp:fast  + /LTCG /OPT:REF /OPT:ICF
    -Native additionally enables /arch:AVX2 and /GS- (faster, but the binary
    then requires an AVX2 capable CPU and drops buffer security checks).

.EXAMPLE
    .\build.ps1                 # optimized x64 build
    .\build.ps1 -Run            # build, then start the exe
    .\build.ps1 -Native -Run    # AVX2 build, then start
    .\build.ps1 -Watch -Run     # rebuild + restart whenever the source changes
    .\build.ps1 -Clean          # delete the build directory
#>
[CmdletBinding()]
param(
    [ValidateSet('x64', 'x86')]
    [string]$Arch = 'x64',

    [switch]$Native,
    [switch]$Run,
    [switch]$Watch,
    [switch]$Clean,

    [string]$OutDir
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not reliably populated inside the param block, so resolve here.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $OutDir) { $OutDir = Join-Path $ScriptDir 'build' }

$SourceFile = Join-Path $ScriptDir 'FluidSimCpp.cpp'
$ArchDir    = Join-Path $OutDir $Arch
$ObjDir     = Join-Path $ArchDir 'obj'
$ExePath    = Join-Path $ArchDir 'FluidSimCpp.exe'

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Find-VisualStudio {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found - install Visual Studio 2017 or newer (Desktop development with C++)."
    }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $path) {
        throw "No Visual Studio installation with the MSVC C++ toolset found."
    }
    return $path.Trim()
}

# vcvarsall.bat only exports into its own cmd.exe process, so run it there,
# dump the resulting environment and copy it into this PowerShell session.
function Import-VcVars([string]$VsPath, [string]$Target) {
    $bat = Join-Path $VsPath 'VC\Auxiliary\Build\vcvarsall.bat'
    if (-not (Test-Path $bat)) { throw "vcvarsall.bat not found at $bat" }

    $dump = cmd.exe /c "`"$bat`" $Target >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat $Target failed (exit $LASTEXITCODE)." }

    foreach ($line in $dump) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-Build {
    if (-not (Test-Path $SourceFile)) { throw "Source file not found: $SourceFile" }
    New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null

    $compilerArgs = @(
        '/nologo'
        '/std:c++20'
        '/permissive-'
        '/EHsc'
        '/W3'
        '/MT'                       # static CRT - single self-contained exe
        '/DNDEBUG', '/DUNICODE', '/D_UNICODE', '/D_CONSOLE'
        '/O2'                       # maximize speed
        '/Ob3'                       # most aggressive inlining
        '/Oi'                        # intrinsics instead of library calls
        '/Ot'                        # favor fast code over small code
        '/GL'                        # whole program optimization (needs /LTCG)
        '/Gy', '/Gw'                 # function/data level linking -> dead code stripping
        '/GF'                        # pool identical string literals
        '/fp:fast'                   # relaxed float math - fine for a particle sim
    )
    if ($Native) { $compilerArgs += @('/arch:AVX2', '/GS-') }

    $compilerArgs += @("/Fo$ObjDir\", "/Fe$ExePath", $SourceFile)

    $linkerArgs = @(
        '/link'
        '/SUBSYSTEM:WINDOWS'         # wWinMain entry point
        '/LTCG'                      # link time code generation, pairs with /GL
        '/OPT:REF', '/OPT:ICF'       # drop unused funcs, fold identical ones
        '/INCREMENTAL:NO'
        'user32.lib', 'gdi32.lib'
    )

    Write-Step "Compiling $Arch$(if ($Native) { ' (AVX2)' })"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cl.exe @compilerArgs @linkerArgs
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    if ($exitCode -ne 0) {
        Write-Host "==> BUILD FAILED (cl.exe exit $exitCode)" -ForegroundColor Red
        return $false
    }

    $sizeKb = [math]::Round((Get-Item $ExePath).Length / 1KB, 1)
    Write-Host "==> OK  $ExePath  ($sizeKb KB, $([math]::Round($sw.Elapsed.TotalSeconds, 2))s)" -ForegroundColor Green
    return $true
}

# --- main -------------------------------------------------------------------

if ($Clean) {
    if (Test-Path $OutDir) {
        Write-Step "Removing $OutDir"
        Remove-Item -Recurse -Force $OutDir
    }
    if (-not ($Run -or $Watch)) { return }
}

$vsPath = Find-VisualStudio
Write-Step "Visual Studio: $vsPath"
Import-VcVars -VsPath $vsPath -Target $(if ($Arch -eq 'x64') { 'x64' } else { 'x86' })

$process = $null

function Start-App {
    if ($script:process -and -not $script:process.HasExited) {
        $script:process.Kill()
        $script:process.WaitForExit()
    }
    Write-Step "Starting $ExePath"
    $script:process = Start-Process -FilePath $ExePath -PassThru
}

$built = Invoke-Build
if (-not $built -and -not $Watch) { exit 1 }
if ($built -and $Run) { Start-App }

if ($Watch) {
    Write-Step "Watching $SourceFile - press Ctrl+C to stop"
    $lastWrite = (Get-Item $SourceFile).LastWriteTimeUtc
    while ($true) {
        Start-Sleep -Milliseconds 500
        $current = (Get-Item $SourceFile).LastWriteTimeUtc
        if ($current -ne $lastWrite) {
            $lastWrite = $current
            Start-Sleep -Milliseconds 200   # let the editor finish writing
            Write-Host ''
            Write-Step "Change detected"
            if ((Invoke-Build) -and $Run) { Start-App }
        }
    }
}
