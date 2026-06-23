#include <windows.h>
#include <windowsx.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cwchar>
#include <algorithm>
#include <random>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

struct Vec2 { float x; float y; };

namespace cfg {
    constexpr int MaxParticles = 10000;
    constexpr int InitialCount = 1500;
    constexpr float MouseForce = -2500.0f;
    constexpr float MouseRadius = 100.0f;
    constexpr int ParticlesToSpawn = 10;
    constexpr float WallMargin = 25.0f;
    constexpr float GravityY = 9.81f * 100.0f;
    constexpr float WallForce = 2000.0f + GravityY;
    constexpr float ParticleMass = 5.0f;
    constexpr float CollisionRadius = 10.0f;
    constexpr float RepulsionForce = 2000.0f;
    constexpr float DampingFactor = 10.0f;
    constexpr float PhysikHzRate = 100.0f;
    constexpr float PhysicsStep = 1.0f / PhysikHzRate;
    constexpr int GridCellSize = 10;
    constexpr uint32_t ParticleColor = 0x0087CEEBu;
}

class Simulation {
public:
    Simulation(int width, int height)
        : particleCount(0), screenWidth(width), screenHeight(height),
          gridCols(0), gridRows(0), mouseX(0.0f), mouseY(0.0f),
          leftDown(false), rightDown(false), accumulator(0.0),
          rng(std::random_device{}())
    {
        pos.resize(cfg::MaxParticles);
        vel.resize(cfg::MaxParticles);
        acc.resize(cfg::MaxParticles);
        nextParticle.resize(cfg::MaxParticles);
        rebuildGrid();
        initParticles();
    }

    void resize(int width, int height) {
        if (width <= 0 || height <= 0) return;
        screenWidth = width;
        screenHeight = height;
        rebuildGrid();
    }

    void setMouse(float x, float y, bool left, bool right) {
        mouseX = x;
        mouseY = y;
        leftDown = left;
        rightDown = right;
    }

    void update(float dt) {
        if (dt > 0.25f) dt = 0.25f;
        accumulator += dt;
        while (accumulator >= cfg::PhysicsStep) {
            step(cfg::PhysicsStep);
            accumulator -= cfg::PhysicsStep;
        }
    }

    void render(uint32_t* bits, int width, int height) const {
        for (int i = 0; i < particleCount; ++i) {
            int px = (int)pos[i].x;
            int py = (int)pos[i].y;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = py + dy;
                if (yy < 0 || yy >= height) continue;
                uint32_t* row = bits + (size_t)yy * width;
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = px + dx;
                    if (xx < 0 || xx >= width) continue;
                    row[xx] = cfg::ParticleColor;
                }
            }
        }
    }

    int count() const { return particleCount; }

private:
    void rebuildGrid() {
        gridCols = (screenWidth / cfg::GridCellSize) + 1;
        gridRows = (screenHeight / cfg::GridCellSize) + 1;
        gridHeads.assign((size_t)gridCols * (size_t)gridRows, -1);
    }

    int randRange(int lo, int hiExclusive) {
        std::uniform_int_distribution<int> d(lo, hiExclusive - 1);
        return d(rng);
    }

    void initParticles() {
        for (int i = 0; i < cfg::InitialCount; ++i) {
            pos[i] = Vec2{
                (float)randRange(screenWidth / 2 - 100, screenWidth / 2 + 100),
                (float)randRange(screenHeight / 2 - 100, screenHeight / 2 + 100)
            };
            vel[i] = Vec2{ 0.0f, 0.0f };
            acc[i] = Vec2{ 0.0f, 0.0f };
        }
        particleCount = cfg::InitialCount;
    }

    void spawnParticles() {
        for (int k = 0; k < cfg::ParticlesToSpawn; ++k) {
            if (particleCount >= cfg::MaxParticles) break;
            int idx = particleCount;
            pos[idx] = Vec2{ mouseX + (float)randRange(-10, 10), mouseY + (float)randRange(-10, 10) };
            vel[idx] = Vec2{ 0.0f, 50.0f };
            acc[idx] = Vec2{ 0.0f, 0.0f };
            ++particleCount;
        }
    }

    void step(float dt) {
        if (rightDown) spawnParticles();
        if (particleCount == 0) return;

        std::fill(gridHeads.begin(), gridHeads.end(), -1);

        for (int i = 0; i < particleCount; ++i) {
            int cx = (int)(pos[i].x / cfg::GridCellSize);
            int cy = (int)(pos[i].y / cfg::GridCellSize);
            if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;
            int cell = cy * gridCols + cx;
            nextParticle[i] = gridHeads[cell];
            gridHeads[cell] = i;
        }

        computeForces();
        integrate(dt);
    }

    void computeForces() {
        const float collisionRadiusSqr = cfg::CollisionRadius * cfg::CollisionRadius;
        for (int i = 0; i < particleCount; ++i) {
            float forceX = 0.0f;
            float forceY = 0.0f;
            float myX = pos[i].x;
            float myY = pos[i].y;
            float myVX = vel[i].x;
            float myVY = vel[i].y;

            if (myX < cfg::WallMargin)
                forceX += (cfg::WallMargin - myX) * cfg::WallForce;
            else if (myX > screenWidth - cfg::WallMargin)
                forceX -= (myX - (screenWidth - cfg::WallMargin)) * cfg::WallForce;

            if (myY < cfg::WallMargin)
                forceY += (cfg::WallMargin - myY) * cfg::WallForce;
            else if (myY > screenHeight - cfg::WallMargin)
                forceY -= (myY - (screenHeight - cfg::WallMargin)) * cfg::WallForce;

            int cx = (int)(myX / cfg::GridCellSize);
            int cy = (int)(myY / cfg::GridCellSize);
            if (cx < 0) cx = 0; else if (cx >= gridCols) cx = gridCols - 1;
            if (cy < 0) cy = 0; else if (cy >= gridRows) cy = gridRows - 1;

            int startX = cx > 0 ? cx - 1 : 0;
            int endX = cx < gridCols - 1 ? cx + 1 : gridCols - 1;
            int startY = cy > 0 ? cy - 1 : 0;
            int endY = cy < gridRows - 1 ? cy + 1 : gridRows - 1;

            for (int y = startY; y <= endY; ++y) {
                int rowOffset = y * gridCols;
                for (int x = startX; x <= endX; ++x) {
                    int neighborIdx = gridHeads[rowOffset + x];
                    while (neighborIdx != -1) {
                        if (i != neighborIdx) {
                            float offX = myX - pos[neighborIdx].x;
                            float offY = myY - pos[neighborIdx].y;
                            float distSqr = offX * offX + offY * offY;
                            if (distSqr < collisionRadiusSqr && distSqr > 0.0001f) {
                                float distance = std::sqrt(distSqr);
                                float factor = (cfg::CollisionRadius - distance) / distance * cfg::RepulsionForce;
                                forceX += offX * factor;
                                forceY += offY * factor;
                                float relVelX = vel[neighborIdx].x - myVX;
                                float relVelY = vel[neighborIdx].y - myVY;
                                forceX += relVelX * cfg::DampingFactor;
                                forceY += relVelY * cfg::DampingFactor;
                            }
                        }
                        neighborIdx = nextParticle[neighborIdx];
                    }
                }
            }

            float accX = forceX / cfg::ParticleMass;
            float accY = cfg::GravityY + (forceY / cfg::ParticleMass);

            if (leftDown) {
                float tmX = mouseX - myX;
                float tmY = mouseY - myY;
                float dSq = tmX * tmX + tmY * tmY;
                if (dSq < cfg::MouseRadius * cfg::MouseRadius) {
                    float dist = std::sqrt(dSq);
                    float f = cfg::MouseForce / (dist + 1.0f);
                    accX += tmX * f;
                    accY += tmY * f;
                }
            }

            acc[i] = Vec2{ accX, accY };
        }
    }

    void integrate(float dt) {
        const float boundaryFriction = 0.5f;
        const float bounce = -0.2f;
        for (int i = 0; i < particleCount; ++i) {
            Vec2 v = vel[i];
            Vec2 a = acc[i];
            Vec2 p = pos[i];

            v.x += a.x * dt;
            v.y += a.y * dt;
            p.x += v.x * dt;
            p.y += v.y * dt;

            if (p.x < 0.0f) { p.x = 0.0f; v.x *= bounce; }
            if (p.y < 0.0f) { p.y = 0.0f; v.y *= bounce; }
            if (p.x > (float)screenWidth) { p.x = (float)screenWidth; v.x *= bounce; }
            if (p.y > (float)screenHeight) { p.y = (float)screenHeight; v.y *= bounce; v.x *= boundaryFriction; }

            vel[i] = v;
            pos[i] = p;
        }
    }

    std::vector<Vec2> pos;
    std::vector<Vec2> vel;
    std::vector<Vec2> acc;
    std::vector<int> gridHeads;
    std::vector<int> nextParticle;
    int particleCount;
    int screenWidth;
    int screenHeight;
    int gridCols;
    int gridRows;
    float mouseX;
    float mouseY;
    bool leftDown;
    bool rightDown;
    double accumulator;
    std::mt19937 rng;
};

static Simulation* g_sim = nullptr;
static HDC g_memDC = nullptr;
static HBITMAP g_bitmap = nullptr;
static HGDIOBJ g_oldBitmap = nullptr;
static uint32_t* g_bits = nullptr;
static int g_bbWidth = 0;
static int g_bbHeight = 0;
static int g_clientWidth = 800;
static int g_clientHeight = 450;
static float g_mouseX = 0.0f;
static float g_mouseY = 0.0f;
static bool g_leftDown = false;
static bool g_rightDown = false;

static void destroyBackbuffer() {
    if (g_memDC) {
        if (g_oldBitmap) SelectObject(g_memDC, g_oldBitmap);
        if (g_bitmap) DeleteObject(g_bitmap);
        DeleteDC(g_memDC);
    }
    g_memDC = nullptr;
    g_bitmap = nullptr;
    g_oldBitmap = nullptr;
    g_bits = nullptr;
}

static void createBackbuffer(HWND hwnd, int w, int h) {
    destroyBackbuffer();
    HDC wdc = GetDC(hwnd);
    g_memDC = CreateCompatibleDC(wdc);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    g_bitmap = CreateDIBSection(wdc, &bmi, DIB_RGB_COLORS, (void**)&g_bits, nullptr, 0);
    g_oldBitmap = SelectObject(g_memDC, g_bitmap);
    ReleaseDC(hwnd, wdc);

    g_bbWidth = w;
    g_bbHeight = h;
}

static void drawOverlay(float fps, int particleCount) {
    SetBkMode(g_memDC, TRANSPARENT);
    HGDIOBJ oldFont = SelectObject(g_memDC, GetStockObject(DEFAULT_GUI_FONT));
    HBRUSH dark = CreateSolidBrush(RGB(20, 20, 20));

    wchar_t buf[128];

    RECT left = { 5, 5, 165, 46 };
    FillRect(g_memDC, &left, dark);
    SetTextColor(g_memDC, RGB(0, 255, 0));
    swprintf(buf, 128, L"FPS: %.1f", fps);
    TextOutW(g_memDC, 12, 8, buf, (int)wcslen(buf));
    SetTextColor(g_memDC, RGB(128, 128, 128));
    swprintf(buf, 128, L"PhysSteps: %.0f Hz", cfg::PhysikHzRate);
    TextOutW(g_memDC, 12, 26, buf, (int)wcslen(buf));

    RECT right = { g_bbWidth - 160, 5, g_bbWidth - 5, 28 };
    FillRect(g_memDC, &right, dark);
    SetTextColor(g_memDC, RGB(255, 255, 255));
    swprintf(buf, 128, L"Particles: %d", particleCount);
    TextOutW(g_memDC, g_bbWidth - 153, 8, buf, (int)wcslen(buf));

    DeleteObject(dark);
    SelectObject(g_memDC, oldFont);
}

static void renderFrame(HWND hwnd, float fps) {
    if (!g_bits || !g_sim) return;
    std::memset(g_bits, 0, (size_t)g_bbWidth * (size_t)g_bbHeight * sizeof(uint32_t));
    g_sim->render(g_bits, g_bbWidth, g_bbHeight);
    drawOverlay(fps, g_sim->count());

    HDC wdc = GetDC(hwnd);
    BitBlt(wdc, 0, 0, g_bbWidth, g_bbHeight, g_memDC, 0, 0, SRCCOPY);
    ReleaseDC(hwnd, wdc);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_SIZE: {
        int w = LOWORD(lParam);
        int h = HIWORD(lParam);
        if (w > 0 && h > 0) {
            g_clientWidth = w;
            g_clientHeight = h;
            createBackbuffer(hwnd, w, h);
            if (g_sim) g_sim->resize(w, h);
        }
        return 0;
    }
    case WM_MOUSEMOVE:
        g_mouseX = (float)GET_X_LPARAM(lParam);
        g_mouseY = (float)GET_Y_LPARAM(lParam);
        return 0;
    case WM_LBUTTONDOWN:
        g_leftDown = true;
        SetCapture(hwnd);
        return 0;
    case WM_RBUTTONDOWN:
        g_rightDown = true;
        SetCapture(hwnd);
        return 0;
    case WM_LBUTTONUP:
        g_leftDown = false;
        if (!g_rightDown) ReleaseCapture();
        return 0;
    case WM_RBUTTONUP:
        g_rightDown = false;
        if (!g_leftDown) ReleaseCapture();
        return 0;
    case WM_ERASEBKGND:
        return 1;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    const wchar_t* className = L"FluidSimCppWindow";

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = className;
    RegisterClassExW(&wc);

    RECT rc = { 0, 0, g_clientWidth, g_clientHeight };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowExW(
        0, className, L"Fluid Simulation",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd) return 0;

    g_sim = new Simulation(g_clientWidth, g_clientHeight);
    createBackbuffer(hwnd, g_clientWidth, g_clientHeight);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    LARGE_INTEGER freq, prev, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    float smoothedFps = 0.0f;
    MSG msg{};
    bool running = true;

    while (running) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) { running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;

        if (dt > 0.0f) {
            float instFps = 1.0f / dt;
            if (smoothedFps <= 0.0f) smoothedFps = instFps;
            else smoothedFps = smoothedFps * 0.95f + instFps * 0.05f;
        }

        g_sim->setMouse(g_mouseX, g_mouseY, g_leftDown, g_rightDown);
        g_sim->update(dt);
        renderFrame(hwnd, smoothedFps);
    }

    destroyBackbuffer();
    delete g_sim;
    g_sim = nullptr;
    UnregisterClassW(className, hInstance);
    return 0;
}
