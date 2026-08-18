#define NOMINMAX          // windows.h defines min/max macros that break std::min/std::max
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <immintrin.h>
#include <intrin.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cwchar>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "shell32.lib")

struct Vec2 { float x; float y; };

namespace cfg {
    constexpr int MaxParticles = 200000;
    constexpr int InitialCount = 1500;
    constexpr float MouseForce = -2500.0f;
    constexpr float MouseRadius = 100.0f;
    constexpr int ParticlesToSpawn = 25;
    constexpr float WallMargin = 25.0f;
    constexpr float GravityY = 9.81f * 100.0f;
    constexpr float WallForce = 2000.0f + GravityY;
    constexpr float ParticleMass = 5.0f;
    constexpr float CollisionRadius = 10.0f;
    constexpr float RepulsionForce = 2000.0f;
    constexpr float DampingFactor = 10.0f;
    // Repulsion is a linear spring: the force vector is offset * (R-d)/d * F,
    // and since |offset| is d that comes out as |F| = (R-d) * RepulsionForce -
    // capped at R * RepulsionForce. It cannot resist arbitrary compression, so
    // the step size is what keeps particles from tunnelling into each other.
    //
    // The hard criterion is CFL-like: the fastest particle must not cross more
    // than this fraction of the interaction radius in one step. Measured at
    // 88k particles - 5.2 px of travel per step collapses 1.6% of all particles
    // into stacks (they then render as one pixel and clear a void of radius R
    // around themselves), 2.3 px brings that down to 0.04%.
    constexpr float MaxTravelPerStep = 0.25f;      // of CollisionRadius
    constexpr float RateSafetyMargin = 1.25f;
    // Density only provides a floor for the very first frames, before any
    // speed measurement exists.
    constexpr float BasePhysicsHz = 100.0f;
    constexpr float MaxPhysicsHz = 1000.0f;
    // Particle count per 800x450 that the base rate is calibrated for.
    constexpr float RateReferenceCount = 20000.0f;
    constexpr float RateReferenceArea = 800.0f * 450.0f;
    constexpr int GridCellSize = 10;
    constexpr float InvGridCellSize = 1.0f / (float)GridCellSize;
    constexpr uint32_t ParticleColor = 0x0087CEEBu;

    // Above this density SMT threads start to pay off: the per-particle
    // neighbourhood is then large enough that a second thread per core hides
    // memory latency instead of just contending for the FMA units. Calibrated
    // as 90k particles in an 800x450 window.
    constexpr float SmtDensityThreshold = 90000.0f / (800.0f * 450.0f);
    constexpr float SmtHysteresis = 0.9f;

    // Lowest frame rate the accumulator still tries to keep up with. Below it
    // the backlog is dropped, otherwise a slow frame would queue more physics
    // steps and make the next frame slower still - the spiral of death.
    constexpr float MinRealtimeFps = 20.0f;
}

// Density-derived floor. Only used until the CFL limit below has real speed
// data to work with.
static float physicsRateFor(int particleCount, int width, int height) {
    const float area = (float)width * (float)height;
    if (area <= 0.0f || particleCount <= 0) return cfg::BasePhysicsHz;

    const float density = (float)particleCount / area;
    const float reference = cfg::RateReferenceCount / cfg::RateReferenceArea;
    float scale = std::sqrt(density / reference);
    if (scale < 1.0f) scale = 1.0f;

    const float rate = cfg::BasePhysicsHz * scale;
    return rate > cfg::MaxPhysicsHz ? cfg::MaxPhysicsHz : rate;
}

// Rate needed to keep the fastest approach between two interacting particles
// under MaxTravelPerStep. Relative motion is the quantity that matters:
// particles falling together as a block never tunnel, however fast they drop,
// and a particle with no neighbour in range contributes nothing at all.
static float cflRateFor(float maxSpeed) {
    if (maxSpeed <= 0.0f) return 0.0f;
    const float maxTravel = cfg::MaxTravelPerStep * cfg::CollisionRadius;
    return cfg::RateSafetyMargin * maxSpeed / maxTravel;
}

static int maxSubStepsFor(float rate) {
    const int steps = (int)std::ceil(rate / cfg::MinRealtimeFps);
    return steps < 2 ? 2 : steps;
}

// Measured on a 12C/24T Xeon: at 20k particles the physical cores are twice as
// fast as the full logical set, at 65k the two are level, at 100k the logical
// set wins by ~17%. The switch therefore follows density, with a hysteresis
// band so a count hovering at the threshold cannot flap the pool.
static int threadCountFor(int particleCount, int width, int height,
                          int physical, int logical, int current) {
    if (logical <= physical) return physical;

    const float area = (float)width * (float)height;
    if (area <= 0.0f) return physical;

    const float density = (float)particleCount / area;
    const float threshold = (current >= logical)
        ? cfg::SmtDensityThreshold * cfg::SmtHysteresis
        : cfg::SmtDensityThreshold;

    return density > threshold ? logical : physical;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static bool hasAvx2Support() {
    int info[4] = { 0, 0, 0, 0 };
    __cpuid(info, 0);
    if (info[0] < 7) return false;

    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return false;

    // XMM and YMM state must actually be saved by the OS, otherwise AVX faults.
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;

    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}

// Physical cores and logical processors. Deriving physical as logical/2 would
// be wrong on any CPU without SMT, so the real topology is queried.
struct ThreadTopology { int physical; int logical; };

static ThreadTopology detectTopology() {
    ThreadTopology topo{ 0, 0 };

    unsigned hw = std::thread::hardware_concurrency();
    topo.logical = hw > 0 ? (int)hw : 4;

    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length > 0) {
        std::vector<uint8_t> buffer(length);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()),
                &length)) {
            DWORD offset = 0;
            while (offset < length) {
                const auto* record =
                    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
                if (record->Size == 0) break;
                if (record->Relationship == RelationProcessorCore) ++topo.physical;
                offset += record->Size;
            }
        }
    }

    if (topo.physical < 1 || topo.physical > topo.logical) {
        topo.physical = topo.logical > 1 ? topo.logical / 2 : 1;
    }
    return topo;
}

// 32-byte aligned float storage with 8 floats of slack, so an AVX2 load may
// overrun the logical end of the data without leaving the allocation.
class AlignedFloats {
public:
    AlignedFloats() : data_(nullptr) {}
    ~AlignedFloats() { release(); }
    AlignedFloats(const AlignedFloats&) = delete;
    AlignedFloats& operator=(const AlignedFloats&) = delete;

    void resize(size_t count) {
        release();
        const size_t bytes = (count + 8) * sizeof(float);
        data_ = static_cast<float*>(_aligned_malloc(bytes, 32));
        if (data_) std::memset(data_, 0, bytes);
    }

    float* get() const { return data_; }

private:
    void release() {
        if (data_) { _aligned_free(data_); data_ = nullptr; }
    }
    float* data_;
};

// Persistent worker threads. Creating threads per physics step would dominate
// the cost at 100+ steps per second, so the pool is spun up once and parked on
// a generation counter between jobs.
//
// The pool always owns one worker per logical processor, but only the first
// activeThreads of them take part in a job. Threads beyond that sit the job out
// and idle instead of spinning - an oversubscribed spin barrier is actively
// harmful, because a spinning SMT sibling steals issue slots from the thread
// doing real work on the same core.
class ThreadPool {
public:
    explicit ThreadPool(int capacity)
        : threadCapacity(capacity < 1 ? 1 : capacity)
    {
        activeThreads.store(threadCapacity, std::memory_order_relaxed);
        for (int i = 1; i < threadCapacity; ++i) {
            workers.emplace_back([this, i] { workerLoop(i); });
        }
    }

    ~ThreadPool() {
        stopping.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (auto& w : workers) if (w.joinable()) w.join();
    }

    int capacity() const { return threadCapacity; }
    int size() const { return activeThreads.load(std::memory_order_acquire); }

    // Must only be called between run() calls - a job in flight has already
    // committed to a thread count.
    void resize(int threads) {
        if (threads < 1) threads = 1;
        if (threads > threadCapacity) threads = threadCapacity;
        activeThreads.store(threads, std::memory_order_release);
    }

    // Runs fn(threadIndex) for indices 0..size()-1 and blocks until all finish.
    // The calling thread takes index 0 and does its share of the work.
    void run(const std::function<void(int)>& fn) {
        const int threads = activeThreads.load(std::memory_order_acquire);
        if (threads == 1) { fn(0); return; }

        job = &fn;
        completed.store(0, std::memory_order_relaxed);
        generation.fetch_add(1, std::memory_order_release);

        fn(0);

        int spins = 0;
        while (completed.load(std::memory_order_acquire) < threads - 1) {
            backoff(spins);
        }
        job = nullptr;
    }

private:
    static void backoff(int& spins) {
        ++spins;
        if (spins < 4000) _mm_pause();
        else if (spins < 20000) std::this_thread::yield();
        else std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    void workerLoop(int index) {
        int seen = 0;
        for (;;) {
            int spins = 0;
            while (generation.load(std::memory_order_acquire) == seen) {
                backoff(spins);
            }
            seen = generation.load(std::memory_order_acquire);
            if (stopping.load(std::memory_order_acquire)) return;

            // activeThreads is published before the generation bump, so an
            // acquire load here always sees the count this job was started
            // with.
            if (index >= activeThreads.load(std::memory_order_acquire)) {
                // Deactivated: not counted in this job's barrier. Idle rather
                // than spin, and re-check on the next job.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            (*job)(index);
            completed.fetch_add(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> workers;
    const std::function<void(int)>* job = nullptr;
    std::atomic<int> generation{ 0 };
    std::atomic<int> completed{ 0 };
    std::atomic<int> activeThreads{ 1 };
    std::atomic<bool> stopping{ false };
    int threadCapacity;
};

// ---------------------------------------------------------------------------
// Simulation interface - lets the optimized and reference paths be swapped at
// runtime for A/B measurement.
// ---------------------------------------------------------------------------

class ISimulation {
public:
    virtual ~ISimulation() = default;
    virtual void resize(int width, int height) = 0;
    virtual void setMouse(float x, float y, bool left, bool right) = 0;
    virtual void step(float dt) = 0;
    virtual void render(uint32_t* bits, int width, int height) = 0;
    virtual int count() const = 0;
    virtual void seedUniform(int n, uint32_t seed) = 0;
    virtual void collectPositions(std::vector<float>& out) const = 0;
    // Largest relative speed between any two interacting particles in the
    // last step. Drives the CFL step-size limit.
    virtual float lastMaxSpeed() const = 0;
};

// ---------------------------------------------------------------------------
// Reference implementation - the original AoS + linked-list-grid version.
// Kept as the correctness and speed baseline, reachable via -scalar.
// ---------------------------------------------------------------------------

class ReferenceSimulation : public ISimulation {
public:
    ReferenceSimulation(int width, int height)
        : particleCount(0), screenWidth(width), screenHeight(height),
          gridCols(0), gridRows(0), mouseX(0.0f), mouseY(0.0f),
          leftDown(false), rightDown(false), rng(1234u)
    {
        pos.resize(cfg::MaxParticles);
        vel.resize(cfg::MaxParticles);
        acc.resize(cfg::MaxParticles);
        nextParticle.resize(cfg::MaxParticles);
        rebuildGrid();
        initParticles();
    }

    void resize(int width, int height) override {
        if (width <= 0 || height <= 0) return;
        screenWidth = width;
        screenHeight = height;
        rebuildGrid();
    }

    void setMouse(float x, float y, bool left, bool right) override {
        mouseX = x; mouseY = y; leftDown = left; rightDown = right;
    }

    int count() const override { return particleCount; }
    float lastMaxSpeed() const override { return maxSpeed; }

    void render(uint32_t* bits, int width, int height) override {
        std::memset(bits, 0, (size_t)width * (size_t)height * sizeof(uint32_t));
        for (int i = 0; i < particleCount; ++i) {
            const int xx = (int)pos[i].x;
            const int yy = (int)pos[i].y;
            if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
            bits[(size_t)yy * width + xx] = cfg::ParticleColor;
        }
    }

    void seedUniform(int n, uint32_t seed) override {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dx(cfg::WallMargin, (float)screenWidth - cfg::WallMargin);
        std::uniform_real_distribution<float> dy(cfg::WallMargin, (float)screenHeight - cfg::WallMargin);
        if (n > cfg::MaxParticles) n = cfg::MaxParticles;
        for (int i = 0; i < n; ++i) {
            pos[i] = Vec2{ dx(gen), dy(gen) };
            vel[i] = Vec2{ 0.0f, 0.0f };
            acc[i] = Vec2{ 0.0f, 0.0f };
        }
        particleCount = n;
    }

    void collectPositions(std::vector<float>& out) const override {
        out.resize((size_t)particleCount * 2);
        for (int i = 0; i < particleCount; ++i) {
            out[(size_t)i * 2 + 0] = pos[i].x;
            out[(size_t)i * 2 + 1] = pos[i].y;
        }
    }

    void step(float dt) override {
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

    void computeForces() {
        const float collisionRadiusSqr = cfg::CollisionRadius * cfg::CollisionRadius;
        float maxRelSqr = 0.0f;
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

                                const float relSqr = relVelX * relVelX + relVelY * relVelY;
                                if (relSqr > maxRelSqr) maxRelSqr = relSqr;
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
        maxSpeed = std::sqrt(maxRelSqr);
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
    float maxSpeed = 0.0f;
    std::mt19937 rng;
};

// ---------------------------------------------------------------------------
// Optimized implementation - SoA storage, counting-sort grid, AVX2 kernel,
// work spread over a thread pool.
// ---------------------------------------------------------------------------

// Reciprocal square root with one Newton-Raphson refinement. _mm256_rsqrt_ps
// alone carries only ~12 bits of mantissa; the refinement lifts it to ~23.
static inline __m256 rsqrtRefined(__m256 x) {
    const __m256 r = _mm256_rsqrt_ps(x);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 threeHalves = _mm256_set1_ps(1.5f);
    const __m256 rr = _mm256_mul_ps(r, r);
    const __m256 t = _mm256_fnmadd_ps(_mm256_mul_ps(half, x), rr, threeHalves);
    return _mm256_mul_ps(r, t);
}

static inline float horizontalMax(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static inline float horizontalSum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// Lane masks: maskFor(k) enables the first k of 8 lanes.
alignas(32) static const int32_t kLaneMaskData[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1,
     0,  0,  0,  0,  0,  0,  0,  0
};
static inline __m256 maskFor(int lanes) {
    return _mm256_castsi256_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kLaneMaskData + 8 - lanes)));
}

class FastSimulation : public ISimulation {
public:
    FastSimulation(int width, int height, ThreadPool& threadPool)
        : pool(threadPool), particleCount(0), screenWidth(width), screenHeight(height),
          gridCols(0), gridRows(0), mouseX(0.0f), mouseY(0.0f),
          leftDown(false), rightDown(false), rng(1234u)
    {
        live.allocate(cfg::MaxParticles);
        sorted.allocate(cfg::MaxParticles);
        cellOf.resize(cfg::MaxParticles);
        threadMaxSpeedSqr.assign((size_t)pool.capacity(), 0.0f);
        rebuildGrid();
        initParticles();
    }

    void resize(int width, int height) override {
        if (width <= 0 || height <= 0) return;
        screenWidth = width;
        screenHeight = height;
        rebuildGrid();
    }

    void setMouse(float x, float y, bool left, bool right) override {
        mouseX = x; mouseY = y; leftDown = left; rightDown = right;
    }

    int count() const override { return particleCount; }

    void seedUniform(int n, uint32_t seed) override {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dx(cfg::WallMargin, (float)screenWidth - cfg::WallMargin);
        std::uniform_real_distribution<float> dy(cfg::WallMargin, (float)screenHeight - cfg::WallMargin);
        if (n > cfg::MaxParticles) n = cfg::MaxParticles;
        for (int i = 0; i < n; ++i) {
            live.x[i] = dx(gen);
            live.y[i] = dy(gen);
            live.vx[i] = 0.0f;
            live.vy[i] = 0.0f;
        }
        particleCount = n;
    }

    float lastMaxSpeed() const override { return maxSpeed; }

    void collectPositions(std::vector<float>& out) const override {
        out.resize((size_t)particleCount * 2);
        for (int i = 0; i < particleCount; ++i) {
            out[(size_t)i * 2 + 0] = live.x[i];
            out[(size_t)i * 2 + 1] = live.y[i];
        }
    }

    void step(float dt) override {
        if (rightDown) spawnParticles();
        if (particleCount == 0) return;

        buildSortedGrid();
        forcesAndIntegrate(dt);
    }

    // Each thread owns a horizontal band of scanlines and only ever writes
    // inside it, so clearing and rasterizing need no synchronization. Particles
    // are scanned in full by every thread - at 65k that is a handful of
    // microseconds and avoids depending on positions still matching the grid.
    // One particle is one pixel.
    void render(uint32_t* bits, int width, int height) override {
        const int threads = pool.size();
        const int n = particleCount;
        const float* px = live.x;
        const float* py = live.y;

        pool.run([&](int t) {
            const int bandHeight = (height + threads - 1) / threads;
            const int y0 = t * bandHeight;
            const int y1 = std::min(y0 + bandHeight, height);
            if (y0 >= y1) return;

            std::memset(bits + (size_t)y0 * width, 0,
                        (size_t)(y1 - y0) * (size_t)width * sizeof(uint32_t));

            for (int i = 0; i < n; ++i) {
                const int yy = (int)py[i];
                if (yy < y0 || yy >= y1) continue;
                const int xx = (int)px[i];
                if (xx < 0 || xx >= width) continue;
                bits[(size_t)yy * width + xx] = cfg::ParticleColor;
            }
        });
    }

private:
    struct Buffers {
        AlignedFloats sx, sy, svx, svy;
        float *x = nullptr, *y = nullptr, *vx = nullptr, *vy = nullptr;

        void allocate(size_t capacity) {
            sx.resize(capacity); sy.resize(capacity);
            svx.resize(capacity); svy.resize(capacity);
            x = sx.get(); y = sy.get(); vx = svx.get(); vy = svy.get();
        }
    };

    void rebuildGrid() {
        gridCols = (screenWidth / cfg::GridCellSize) + 1;
        gridRows = (screenHeight / cfg::GridCellSize) + 1;
        cellCount = gridCols * gridRows;
        cellStart.assign((size_t)cellCount + 1, 0);
        // Sized for the pool's capacity, not its current thread count - the
        // active count can grow at runtime.
        histograms.assign((size_t)cellCount * (size_t)pool.capacity(), 0);
    }

    int randRange(int lo, int hiExclusive) {
        std::uniform_int_distribution<int> d(lo, hiExclusive - 1);
        return d(rng);
    }

    void initParticles() {
        for (int i = 0; i < cfg::InitialCount; ++i) {
            live.x[i] = (float)randRange(screenWidth / 2 - 100, screenWidth / 2 + 100);
            live.y[i] = (float)randRange(screenHeight / 2 - 100, screenHeight / 2 + 100);
            live.vx[i] = 0.0f;
            live.vy[i] = 0.0f;
        }
        particleCount = cfg::InitialCount;
    }

    void spawnParticles() {
        for (int k = 0; k < cfg::ParticlesToSpawn; ++k) {
            if (particleCount >= cfg::MaxParticles) break;
            const int idx = particleCount;
            live.x[idx] = mouseX + (float)randRange(-10, 10);
            live.y[idx] = mouseY + (float)randRange(-10, 10);
            live.vx[idx] = 0.0f;
            live.vy[idx] = 50.0f;
            ++particleCount;
        }
    }

    static inline void chunkFor(int t, int threads, int n, int& lo, int& hi) {
        const int per = (n + threads - 1) / threads;
        lo = t * per;
        hi = std::min(lo + per, n);
        if (lo > hi) lo = hi;
    }

    // Counting sort that physically reorders the particle data by grid cell.
    // Afterwards particles of one cell are contiguous, and because cells of a
    // grid row are contiguous too, the 3x3 neighbourhood collapses into three
    // sequential runs instead of nine pointer-chased linked lists.
    void buildSortedGrid() {
        const int threads = pool.size();
        const int n = particleCount;
        const int cols = gridCols;
        const int rows = gridRows;
        const int cells = cellCount;

        // Pass 1 - per-thread histograms over a private slice, so no atomics.
        pool.run([&](int t) {
            int lo, hi;
            chunkFor(t, threads, n, lo, hi);
            int* hist = histograms.data() + (size_t)t * cells;
            std::memset(hist, 0, (size_t)cells * sizeof(int));

            for (int i = lo; i < hi; ++i) {
                int cx = (int)(live.x[i] * cfg::InvGridCellSize);
                int cy = (int)(live.y[i] * cfg::InvGridCellSize);
                if (cx < 0) cx = 0; else if (cx >= cols) cx = cols - 1;
                if (cy < 0) cy = 0; else if (cy >= rows) cy = rows - 1;
                const int cell = cy * cols + cx;
                cellOf[i] = cell;
                ++hist[cell];
            }
        });

        // Pass 2 - exclusive prefix sum, turning each histogram slot into the
        // write cursor for that (thread, cell) pair. Serial, but only
        // cells * threads iterations.
        int running = 0;
        for (int c = 0; c < cells; ++c) {
            cellStart[c] = running;
            for (int t = 0; t < threads; ++t) {
                int& slot = histograms[(size_t)t * cells + c];
                const int countInSlot = slot;
                slot = running;
                running += countInSlot;
            }
        }
        cellStart[cells] = running;

        // Pass 3 - scatter into cell order. Cursors are disjoint per thread.
        pool.run([&](int t) {
            int lo, hi;
            chunkFor(t, threads, n, lo, hi);
            int* cursor = histograms.data() + (size_t)t * cells;

            for (int i = lo; i < hi; ++i) {
                const int dst = cursor[cellOf[i]]++;
                sorted.x[dst] = live.x[i];
                sorted.y[dst] = live.y[i];
                sorted.vx[dst] = live.vx[i];
                sorted.vy[dst] = live.vy[i];
            }
        });
    }

    // Reads the sorted snapshot, writes the integrated result back into the
    // live buffer. Because reads and writes use different buffers, every
    // particle sees the same position snapshot - identical semantics to the
    // reference two-pass version, but in a single sweep and without acc[].
    // Work is handed out in blocks through an atomic cursor rather than split
    // up front. Cost per particle is proportional to how crowded its
    // neighbourhood is, and a settled fluid is far from uniform - a dense
    // bottom and an empty top would leave most threads idle under a static
    // split.
    void forcesAndIntegrate(float dt) {
        const int n = particleCount;
        constexpr int blockSize = 512;
        const int blocks = (n + blockSize - 1) / blockSize;

        workCursor.store(0, std::memory_order_relaxed);
        std::fill(threadMaxSpeedSqr.begin(), threadMaxSpeedSqr.end(), 0.0f);

        pool.run([&](int t) {
            float localMaxSqr = 0.0f;
            for (;;) {
                const int b = workCursor.fetch_add(1, std::memory_order_relaxed);
                if (b >= blocks) break;
                const int lo = b * blockSize;
                const int hi = std::min(lo + blockSize, n);
                forcesAndIntegrateRange(lo, hi, dt, localMaxSqr);
            }
            threadMaxSpeedSqr[t] = localMaxSqr;
        });

        float maxSqr = 0.0f;
        for (float v : threadMaxSpeedSqr) if (v > maxSqr) maxSqr = v;
        maxSpeed = std::sqrt(maxSqr);
    }

    void forcesAndIntegrateRange(int lo, int hi, float dt, float& maxRelSpeedSqr) {
        const float* __restrict sx = sorted.x;
        const float* __restrict sy = sorted.y;
        const float* __restrict svx = sorted.vx;
        const float* __restrict svy = sorted.vy;

        const int cols = gridCols;
        const int rows = gridRows;
        const float wallRight = (float)screenWidth - cfg::WallMargin;
        const float wallBottom = (float)screenHeight - cfg::WallMargin;
        const float boundaryFriction = 0.5f;
        const float bounce = -0.2f;

        const __m256 vRadius = _mm256_set1_ps(cfg::CollisionRadius);
        const __m256 vRadiusSqr = _mm256_set1_ps(cfg::CollisionRadius * cfg::CollisionRadius);
        const __m256 vEpsilon = _mm256_set1_ps(0.0001f);
        const __m256 vRepulsion = _mm256_set1_ps(cfg::RepulsionForce);
        const __m256 vDamping = _mm256_set1_ps(cfg::DampingFactor);
        const __m256 vOne = _mm256_set1_ps(1.0f);

        for (int i = lo; i < hi; ++i) {
            const float myX = sx[i];
            const float myY = sy[i];
            const float myVX = svx[i];
            const float myVY = svy[i];

            float forceX = 0.0f;
            float forceY = 0.0f;

            if (myX < cfg::WallMargin)
                forceX += (cfg::WallMargin - myX) * cfg::WallForce;
            else if (myX > wallRight)
                forceX -= (myX - wallRight) * cfg::WallForce;

            if (myY < cfg::WallMargin)
                forceY += (cfg::WallMargin - myY) * cfg::WallForce;
            else if (myY > wallBottom)
                forceY -= (myY - wallBottom) * cfg::WallForce;

            int cx = (int)(myX * cfg::InvGridCellSize);
            int cy = (int)(myY * cfg::InvGridCellSize);
            if (cx < 0) cx = 0; else if (cx >= cols) cx = cols - 1;
            if (cy < 0) cy = 0; else if (cy >= rows) cy = rows - 1;

            const int startX = cx > 0 ? cx - 1 : 0;
            const int endX = cx < cols - 1 ? cx + 1 : cols - 1;
            const int startY = cy > 0 ? cy - 1 : 0;
            const int endY = cy < rows - 1 ? cy + 1 : rows - 1;

            const __m256 vMyX = _mm256_set1_ps(myX);
            const __m256 vMyY = _mm256_set1_ps(myY);
            const __m256 vMyVX = _mm256_set1_ps(myVX);
            const __m256 vMyVY = _mm256_set1_ps(myVY);
            __m256 accForceX = _mm256_setzero_ps();
            __m256 accForceY = _mm256_setzero_ps();
            // What can tunnel is relative motion between neighbours, not
            // absolute speed - a block of particles falling together is in no
            // danger however fast it drops.
            __m256 maxRelSqr = _mm256_setzero_ps();

            for (int y = startY; y <= endY; ++y) {
                const int rowBase = y * cols;
                const int begin = cellStart[rowBase + startX];
                const int end = cellStart[rowBase + endX + 1];

                for (int j = begin; j < end; j += 8) {
                    const int remaining = end - j;
                    // Overreading up to 7 floats is safe - the buffers carry
                    // slack - and the surplus lanes are masked out below.
                    const __m256 laneMask = remaining >= 8 ? _mm256_castsi256_ps(_mm256_set1_epi32(-1))
                                                           : maskFor(remaining);

                    const __m256 nx = _mm256_loadu_ps(sx + j);
                    const __m256 ny = _mm256_loadu_ps(sy + j);
                    const __m256 offX = _mm256_sub_ps(vMyX, nx);
                    const __m256 offY = _mm256_sub_ps(vMyY, ny);
                    const __m256 distSqr = _mm256_fmadd_ps(offX, offX, _mm256_mul_ps(offY, offY));

                    // distSqr > epsilon also excludes the particle itself, so
                    // no index comparison is needed - same result as the
                    // reference i != neighborIdx check.
                    __m256 hit = _mm256_and_ps(_mm256_cmp_ps(distSqr, vRadiusSqr, _CMP_LT_OQ),
                                               _mm256_cmp_ps(distSqr, vEpsilon, _CMP_GT_OQ));
                    hit = _mm256_and_ps(hit, laneMask);
                    if (_mm256_testz_ps(hit, hit)) continue;

                    const __m256 invDist = rsqrtRefined(distSqr);
                    // (R - d) / d * F  ==  (R * (1/d) - 1) * F
                    __m256 factor = _mm256_mul_ps(_mm256_fmsub_ps(vRadius, invDist, vOne), vRepulsion);
                    // Bitwise AND also scrubs the inf/NaN that rsqrt produces
                    // for the masked-out lanes where distSqr is zero.
                    factor = _mm256_and_ps(factor, hit);

                    accForceX = _mm256_fmadd_ps(offX, factor, accForceX);
                    accForceY = _mm256_fmadd_ps(offY, factor, accForceY);

                    const __m256 relVX = _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(svx + j), vMyVX), hit);
                    const __m256 relVY = _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(svy + j), vMyVY), hit);
                    accForceX = _mm256_fmadd_ps(relVX, vDamping, accForceX);
                    accForceY = _mm256_fmadd_ps(relVY, vDamping, accForceY);

                    // Already masked, so non-neighbours contribute zero.
                    maxRelSqr = _mm256_max_ps(maxRelSqr,
                        _mm256_fmadd_ps(relVX, relVX, _mm256_mul_ps(relVY, relVY)));
                }
            }

            forceX += horizontalSum(accForceX);
            forceY += horizontalSum(accForceY);

            float accX = forceX / cfg::ParticleMass;
            float accY = cfg::GravityY + (forceY / cfg::ParticleMass);

            if (leftDown) {
                const float tmX = mouseX - myX;
                const float tmY = mouseY - myY;
                const float dSq = tmX * tmX + tmY * tmY;
                if (dSq < cfg::MouseRadius * cfg::MouseRadius) {
                    const float dist = std::sqrt(dSq);
                    const float f = cfg::MouseForce / (dist + 1.0f);
                    accX += tmX * f;
                    accY += tmY * f;
                }
            }

            float vx = myVX + accX * dt;
            float vy = myVY + accY * dt;
            float px = myX + vx * dt;
            float py = myY + vy * dt;

            if (px < 0.0f) { px = 0.0f; vx *= bounce; }
            if (py < 0.0f) { py = 0.0f; vy *= bounce; }
            if (px > (float)screenWidth) { px = (float)screenWidth; vx *= bounce; }
            if (py > (float)screenHeight) { py = (float)screenHeight; vy *= bounce; vx *= boundaryFriction; }

            const float relSqr = horizontalMax(maxRelSqr);
            if (relSqr > maxRelSpeedSqr) maxRelSpeedSqr = relSqr;

            live.x[i] = px;
            live.y[i] = py;
            live.vx[i] = vx;
            live.vy[i] = vy;
        }
    }

    ThreadPool& pool;
    std::atomic<int> workCursor{ 0 };
    std::vector<float> threadMaxSpeedSqr;
    float maxSpeed = 0.0f;
    Buffers live;      // current state, also the destination of the integration
    Buffers sorted;    // cell-ordered snapshot the force kernel reads from
    std::vector<int> cellStart;
    std::vector<int> cellOf;
    std::vector<int> histograms;
    int cellCount = 0;
    int particleCount;
    int screenWidth;
    int screenHeight;
    int gridCols;
    int gridRows;
    float mouseX;
    float mouseY;
    bool leftDown;
    bool rightDown;
    std::mt19937 rng;
};

// ---------------------------------------------------------------------------
// Win32 application
// ---------------------------------------------------------------------------

struct AppOptions {
    bool useScalar = false;
    bool benchmark = false;
    int benchParticles = 65000;
    int benchSteps = 500;
    int benchWarmup = 100;
    int threads = 0;            // 0 = auto
    float fixedHz = 0.0f;       // 0 = derive from density
    int prespawn = 0;
    int width = 1200;
    int height = 720;
    const wchar_t* dumpPath = nullptr;
};

static AppOptions g_options;
static ISimulation* g_sim = nullptr;
static ThreadPool* g_pool = nullptr;
static HDC g_windowDC = nullptr;
static HDC g_memDC = nullptr;
static HBITMAP g_bitmap = nullptr;
static HGDIOBJ g_oldBitmap = nullptr;
static uint32_t* g_bits = nullptr;
static int g_bbWidth = 0;
static int g_bbHeight = 0;
static int g_clientWidth = 1200;
static int g_clientHeight = 720;
static float g_mouseX = 0.0f;
static float g_mouseY = 0.0f;
static bool g_leftDown = false;
static bool g_rightDown = false;

// Live timing shown in the overlay, so tuning is measured rather than guessed.
static double g_physicsMs = 0.0;
static double g_renderMs = 0.0;
static int g_lastSubSteps = 0;
static double g_stepsPerSecond = 0.0;   // measured, shows whether physics keeps up
static double g_rateWindowTime = 0.0;
static int g_rateWindowSteps = 0;

static double g_qpcPeriod = 0.0;
static float g_physicsHz = cfg::BasePhysicsHz;
static ThreadTopology g_topology{ 1, 1 };

// Reacts to a rising requirement immediately - a step that is already too
// large produces stacking on the spot. Comes back down slowly, so a brief calm
// patch cannot leave the next disturbance under-resolved.
static void updatePhysicsRate(int particleCount, int width, int height, float maxSpeed) {
    if (g_options.fixedHz > 0.0f) { g_physicsHz = g_options.fixedHz; return; }

    float target = physicsRateFor(particleCount, width, height);
    const float cfl = cflRateFor(maxSpeed);
    if (cfl > target) target = cfl;
    if (target > cfg::MaxPhysicsHz) target = cfg::MaxPhysicsHz;

    if (target > g_physicsHz) g_physicsHz = target;
    else g_physicsHz += (target - g_physicsHz) * 0.02f;
}

static inline double nowSeconds() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * g_qpcPeriod;
}

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

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    g_memDC = CreateCompatibleDC(g_windowDC);
    g_bitmap = CreateDIBSection(g_windowDC, &bmi, DIB_RGB_COLORS, (void**)&g_bits, nullptr, 0);
    g_oldBitmap = SelectObject(g_memDC, g_bitmap);

    g_bbWidth = w;
    g_bbHeight = h;
}

static void drawOverlay(float fps, int particleCount) {
    SetBkMode(g_memDC, TRANSPARENT);
    HGDIOBJ oldFont = SelectObject(g_memDC, GetStockObject(DEFAULT_GUI_FONT));
    HBRUSH dark = CreateSolidBrush(RGB(20, 20, 20));

    wchar_t buf[128];

    RECT left = { 5, 5, 210, 100 };
    FillRect(g_memDC, &left, dark);
    SetTextColor(g_memDC, RGB(0, 255, 0));
    swprintf(buf, 128, L"FPS: %.1f", fps);
    TextOutW(g_memDC, 12, 8, buf, (int)wcslen(buf));

    SetTextColor(g_memDC, RGB(128, 128, 128));
    swprintf(buf, 128, L"PhysSteps: %.0f Hz%s", g_physicsHz,
             g_options.fixedHz > 0.0f ? L" (fixed)" : L" (auto)");
    TextOutW(g_memDC, 12, 26, buf, (int)wcslen(buf));
    swprintf(buf, 128, L"Physics: %.2f ms/step", g_physicsMs);
    TextOutW(g_memDC, 12, 44, buf, (int)wcslen(buf));
    swprintf(buf, 128, L"Render: %.2f ms", g_renderMs);
    TextOutW(g_memDC, 12, 62, buf, (int)wcslen(buf));
    swprintf(buf, 128, L"Steps/s: %.0f of %.0f   Sub: %d   Thr: %d/%d",
             g_stepsPerSecond, g_physicsHz, g_lastSubSteps,
             g_options.useScalar ? 1 : (g_pool ? g_pool->size() : 1),
             g_pool ? g_pool->capacity() : 1);
    TextOutW(g_memDC, 12, 80, buf, (int)wcslen(buf));

    RECT right = { g_bbWidth - 180, 5, g_bbWidth - 5, 46 };
    FillRect(g_memDC, &right, dark);
    SetTextColor(g_memDC, RGB(255, 255, 255));
    swprintf(buf, 128, L"Particles: %d", particleCount);
    TextOutW(g_memDC, g_bbWidth - 173, 8, buf, (int)wcslen(buf));
    SetTextColor(g_memDC, RGB(160, 160, 160));
    swprintf(buf, 128, L"Path: %s", g_options.useScalar ? L"scalar reference" : L"AVX2 + threads");
    TextOutW(g_memDC, g_bbWidth - 173, 26, buf, (int)wcslen(buf));

    DeleteObject(dark);
    SelectObject(g_memDC, oldFont);
}

static void renderFrame(HWND hwnd, float fps) {
    (void)hwnd;
    if (!g_bits || !g_sim) return;

    const double t0 = nowSeconds();
    g_sim->render(g_bits, g_bbWidth, g_bbHeight);
    g_renderMs = (nowSeconds() - t0) * 1000.0;

    drawOverlay(fps, g_sim->count());
    BitBlt(g_windowDC, 0, 0, g_bbWidth, g_bbHeight, g_memDC, 0, 0, SRCCOPY);
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

// ---------------------------------------------------------------------------
// Command line, console, benchmark
// ---------------------------------------------------------------------------

static void parseCommandLine(AppOptions& opt) {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return;

    auto intArg = [&](int i, int fallback) {
        return (i < argc) ? _wtoi(argv[i]) : fallback;
    };

    for (int i = 1; i < argc; ++i) {
        if (_wcsicmp(argv[i], L"-scalar") == 0) {
            opt.useScalar = true;
        } else if (_wcsicmp(argv[i], L"-bench") == 0) {
            opt.benchmark = true;
            opt.benchParticles = intArg(++i, opt.benchParticles);
        } else if (_wcsicmp(argv[i], L"-steps") == 0) {
            opt.benchSteps = intArg(++i, opt.benchSteps);
        } else if (_wcsicmp(argv[i], L"-warmup") == 0) {
            opt.benchWarmup = intArg(++i, opt.benchWarmup);
        } else if (_wcsicmp(argv[i], L"-threads") == 0) {
            opt.threads = intArg(++i, opt.threads);
        } else if (_wcsicmp(argv[i], L"-hz") == 0) {
            opt.fixedHz = (float)intArg(++i, 0);
        } else if (_wcsicmp(argv[i], L"-n") == 0) {
            opt.prespawn = intArg(++i, opt.prespawn);
        } else if (_wcsicmp(argv[i], L"-size") == 0) {
            opt.width = intArg(++i, opt.width);
            opt.height = intArg(++i, opt.height);
        } else if (_wcsicmp(argv[i], L"-dump") == 0) {
            if (i + 1 < argc) opt.dumpPath = _wcsdup(argv[++i]);
        }
    }

    LocalFree(argv);
}

// A /SUBSYSTEM:WINDOWS binary has no console of its own. If stdout was already
// redirected to a file or pipe the CRT handle is usable as-is and must be left
// alone - AllocConsole would replace it and swallow the redirect. Only when
// there is no usable handle is a console attached or created.
static void ensureConsole() {
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE && GetFileType(out) != FILE_TYPE_UNKNOWN) {
        return;
    }
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
        AllocConsole();
    }
    FILE* dummy = nullptr;
    freopen_s(&dummy, "CONOUT$", "w", stdout);
    freopen_s(&dummy, "CONOUT$", "w", stderr);
}

// Re-evaluates the thread count for the current particle count. A no-op when
// -threads pinned it, and only touches the pool when the decision actually
// changes.
static void applyThreadPolicy(int particleCount, int width, int height) {
    if (!g_pool || g_options.threads > 0) return;
    const int want = threadCountFor(particleCount, width, height,
                                    g_topology.physical, g_topology.logical,
                                    g_pool->size());
    if (want != g_pool->size()) g_pool->resize(want);
}

static ISimulation* createSimulation(int width, int height) {
    if (g_options.useScalar) return new ReferenceSimulation(width, height);
    return new FastSimulation(width, height, *g_pool);
}

static int runBenchmark() {
    ensureConsole();

    const int width = g_options.width;
    const int height = g_options.height;

    std::unique_ptr<ISimulation> sim(createSimulation(width, height));
    sim->seedUniform(g_options.benchParticles, 12345u);
    sim->setMouse(0.0f, 0.0f, false, false);

    updatePhysicsRate(sim->count(), width, height, 0.0f);
    applyThreadPolicy(sim->count(), width, height);

    printf("FluidSimCpp benchmark\n");
    printf("  path       : %s\n", g_options.useScalar ? "scalar reference" : "AVX2 + threads");
    printf("  threads    : %d of %d logical (%d physical)%s\n",
           g_options.useScalar ? 1 : (g_pool ? g_pool->size() : 1),
           g_topology.logical, g_topology.physical,
           g_options.threads > 0 ? " [fixed]" : " [auto]");
    printf("  particles  : %d\n", sim->count());
    printf("  window     : %dx%d\n", width, height);
    printf("  rate       : %.0f Hz (%s)\n", g_physicsHz,
           g_options.fixedHz > 0.0f ? "fixed" : "density derived");
    printf("  warmup     : %d steps\n", g_options.benchWarmup);
    printf("  measured   : %d steps\n", g_options.benchSteps);
    fflush(stdout);

    for (int i = 0; i < g_options.benchWarmup; ++i) {
        updatePhysicsRate(sim->count(), width, height, sim->lastMaxSpeed());
        sim->step(1.0f / g_physicsHz);
    }

    std::vector<double> samples;
    samples.reserve(g_options.benchSteps);
    for (int i = 0; i < g_options.benchSteps; ++i) {
        updatePhysicsRate(sim->count(), width, height, sim->lastMaxSpeed());
        const double t0 = nowSeconds();
        sim->step(1.0f / g_physicsHz);
        samples.push_back((nowSeconds() - t0) * 1000.0);
    }

    double total = 0.0;
    for (double s : samples) total += s;
    const double mean = samples.empty() ? 0.0 : total / (double)samples.size();

    std::vector<double> sortedSamples = samples;
    std::sort(sortedSamples.begin(), sortedSamples.end());
    auto pick = [&](double q) {
        if (sortedSamples.empty()) return 0.0;
        size_t idx = (size_t)(q * (double)(sortedSamples.size() - 1));
        return sortedSamples[idx];
    };

    printf("\n  mean       : %.3f ms/step\n", mean);
    printf("  p50        : %.3f ms/step\n", pick(0.50));
    printf("  p99        : %.3f ms/step\n", pick(0.99));
    const double sustained = mean > 0.0 ? 1000.0 / mean : 0.0;
    printf("  sustained  : %.1f steps/s  (need %.0f for realtime, headroom %.2fx)\n",
           sustained, (double)g_physicsHz, sustained / (double)g_physicsHz);

    // Stability check - a diverging simulation would otherwise look fast.
    std::vector<float> positions;
    sim->collectPositions(positions);
    int outOfBounds = 0;
    int notFinite = 0;
    for (size_t i = 0; i + 1 < positions.size(); i += 2) {
        const float x = positions[i];
        const float y = positions[i + 1];
        if (!std::isfinite(x) || !std::isfinite(y)) { ++notFinite; continue; }
        if (x < -1.0f || y < -1.0f || x > (float)width + 1.0f || y > (float)height + 1.0f) ++outOfBounds;
    }
    printf("  final rate : %.0f Hz\n", g_physicsHz);
    printf("  max approach: %.1f px/s (closes %.2f px per step)\n",
           sim->lastMaxSpeed(), sim->lastMaxSpeed() / g_physicsHz);
    printf("  non-finite : %d\n", notFinite);
    printf("  out of box : %d\n", outOfBounds);

    if (g_options.dumpPath) {
        FILE* f = nullptr;
        if (_wfopen_s(&f, g_options.dumpPath, L"wb") == 0 && f) {
            const int32_t n = (int32_t)(positions.size() / 2);
            fwrite(&n, sizeof(n), 1, f);
            fwrite(positions.data(), sizeof(float), positions.size(), f);
            fclose(f);
            wprintf(L"  dumped     : %s (%d particles)\n", g_options.dumpPath, n);
        } else {
            wprintf(L"  dump failed: %s\n", g_options.dumpPath);
        }
    }

    fflush(stdout);
    return 0;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_qpcPeriod = 1.0 / (double)freq.QuadPart;

    parseCommandLine(g_options);

    if (!g_options.useScalar && !hasAvx2Support()) {
        MessageBoxW(nullptr,
            L"This build requires a CPU with AVX2 and FMA (Intel Haswell / AMD Excavator or newer).\n\n"
            L"Run with -scalar to use the reference path instead.",
            L"Fluid Simulation", MB_ICONERROR | MB_OK);
        return 1;
    }

    g_topology = detectTopology();

    // The pool owns one worker per logical processor and parks the surplus;
    // applyThreadPolicy decides how many actually run each frame.
    ThreadPool pool(g_options.threads > 0 ? g_options.threads : g_topology.logical);
    pool.resize(g_options.threads > 0 ? g_options.threads : g_topology.physical);
    g_pool = &pool;

    if (g_options.benchmark) {
        return runBenchmark();
    }

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

    g_clientWidth = g_options.width;
    g_clientHeight = g_options.height;

    RECT rc = { 0, 0, g_clientWidth, g_clientHeight };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowExW(
        0, className, L"Fluid Simulation",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd) return 0;

    // CS_OWNDC gives this window a private DC, so it can be held for the
    // lifetime of the app instead of fetched and released every frame.
    g_windowDC = GetDC(hwnd);

    std::unique_ptr<ISimulation> sim(createSimulation(g_clientWidth, g_clientHeight));
    g_sim = sim.get();
    if (g_options.prespawn > 0) sim->seedUniform(g_options.prespawn, 12345u);

    createBackbuffer(hwnd, g_clientWidth, g_clientHeight);

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    double prev = nowSeconds();
    double accumulator = 0.0;
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

        const double now = nowSeconds();
        float dt = (float)(now - prev);
        prev = now;

        if (dt > 0.0f) {
            const float instFps = 1.0f / dt;
            if (smoothedFps <= 0.0f) smoothedFps = instFps;
            else smoothedFps = smoothedFps * 0.95f + instFps * 0.05f;
        }
        if (dt > 0.25f) dt = 0.25f;

        sim->setMouse(g_mouseX, g_mouseY, g_leftDown, g_rightDown);

        // Spawning raises the density, which tightens the stability limit and
        // shifts the best thread count, so both are re-derived every frame.
        // Safe here: no job is in flight between frames.
        updatePhysicsRate(sim->count(), g_clientWidth, g_clientHeight, sim->lastMaxSpeed());
        applyThreadPolicy(sim->count(), g_clientWidth, g_clientHeight);
        const float physicsStep = 1.0f / g_physicsHz;
        const int maxSubSteps = maxSubStepsFor(g_physicsHz);

        accumulator += dt;
        int subSteps = 0;
        const double physicsStart = nowSeconds();
        while (accumulator >= physicsStep && subSteps < maxSubSteps) {
            sim->step(physicsStep);
            accumulator -= physicsStep;
            ++subSteps;
        }
        if (subSteps == maxSubSteps) accumulator = 0.0;
        if (subSteps > 0) {
            g_physicsMs = (nowSeconds() - physicsStart) * 1000.0 / (double)subSteps;
        }
        g_lastSubSteps = subSteps;
        // Counted over a window rather than averaged per frame: most frames
        // run zero steps and a few run one, so a per-frame mean of steps/dt
        // would be biased high.
        g_rateWindowTime += dt;
        g_rateWindowSteps += subSteps;
        if (g_rateWindowTime >= 0.25) {
            g_stepsPerSecond = (double)g_rateWindowSteps / g_rateWindowTime;
            g_rateWindowTime = 0.0;
            g_rateWindowSteps = 0;
        }

        renderFrame(hwnd, smoothedFps);
    }

    destroyBackbuffer();
    if (g_windowDC) ReleaseDC(hwnd, g_windowDC);
    g_sim = nullptr;
    sim.reset();
    g_pool = nullptr;
    UnregisterClassW(className, hInstance);
    return 0;
}
