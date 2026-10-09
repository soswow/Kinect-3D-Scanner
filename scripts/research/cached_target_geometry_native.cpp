// Research-only cached target search, not an Open3D evaluator replacement.
// Search layout/sequence follows Open3D v0.20.0 KDTreeFlann (MIT),
// https://github.com/isl-org/Open3D/blob/v0.20.0/cpp/open3d/geometry/KDTreeFlann.cpp
// Pinned Eigen (MPL-2.0) and nanoflann 1.5.0 (BSD) remain external dependencies.
// Their archive/header/license bytes must be verified by the companion helper.
#include <Eigen/Core>
#include <nanoflann.hpp>
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>

#if !defined(_WIN32) || !defined(_M_X64)
#error This source-only prototype declares the Windows x64 MSVC build route.
#endif
#if NANOFLANN_VERSION != 0x150
#error Require the pinned nanoflann 1.5.0 header.
#endif
#ifdef NANOFLANN_FIRST_MATCH
#error A lowest-index tie policy is not the declared Open3D build policy.
#endif

namespace {
using Tree = nanoflann::KDTreeEigenMatrixAdaptor<
        const Eigen::MatrixXd, -1, nanoflann::metric_L2, false>;
static_assert(sizeof(double) == 8 && sizeof(Eigen::Index) == 8,
              "Require original FP64 and x64 Eigen index layout");
static_assert(sizeof(Tree::index_t::Node) <= 64, "Revisit reservation for changed layout");
constexpr std::uint64_t kMaxRows = 1000000;
constexpr std::uint64_t kMaxHandles = 16;
constexpr std::uint64_t kMaxReservation = 512ULL * 1024 * 1024;
constexpr double kCoordinateLimit = 1048576.0;
std::mutex registry_mutex;
thread_local char last_error[512] = {};

// This is a conservative array/index/pool reservation, not process RSS.
// Pinned binary tree has at most 2*N-1 <=64-byte nodes. Include pool padding,
// index capacity, matrix storage and 64 KiB fixed allowance; stack/CRT opaque.
std::uint64_t reservation(std::uint64_t rows) { return 320 * rows + 65536; }
std::uint64_t fp_environment() {
    return (static_cast<std::uint64_t>(std::fegetround()) << 32) | _mm_getcsr();
}
bool fp_valid(std::uint64_t expected) {
    // Bind current rounding, FTZ/DAZ and control bits, excluding accrued flags.
    return std::fegetround() == FE_TONEAREST &&
           (fp_environment() & ~std::uint64_t(0x3f)) == expected;
}
bool points_valid(const double* values, std::uint64_t rows) {
    if (rows > kMaxRows || (rows && !values)) return false;
    for (std::uint64_t i = 0; i < 3 * rows; ++i)
        if (!std::isfinite(values[i]) || std::abs(values[i]) > kCoordinateLimit)
            return false;
    return true;
}
int error(int status, const char* message) noexcept {
    std::size_t i = 0;
    for (; i + 1 < sizeof(last_error) && message[i]; ++i) last_error[i] = message[i];
    last_error[i] = '\0'; return status;
}

struct Target {
    Eigen::MatrixXd data;
    std::unique_ptr<Tree> tree;
    const std::uint64_t rows;
    const std::uint64_t reserved;
    const std::uint64_t fp;
    Target(const double* values, std::uint64_t n, std::uint64_t environment)
        : data(Eigen::Map<const Eigen::MatrixXd>(values, 3, Eigen::Index(n))),
          rows(n), reserved(reservation(n)), fp(environment) {
        tree = std::make_unique<Tree>(data.rows(), data, 15);
        // Keep Open3D's explicit build, including the adaptor constructor's build.
        tree->index_->buildIndex();
        const auto& index = *tree->index_;
        const auto counted = 24 * rows +
            index.vAcc_.capacity() * sizeof(Eigen::Index) +
            index.pool_.usedMemory + index.pool_.wastedMemory +
            8192 + 16 * rows + sizeof(Target) + sizeof(Tree) + sizeof(Tree::index_t);
        if (counted > reserved) throw std::runtime_error("Index reservation exceeded");
    }
};
std::unordered_map<std::uint64_t, std::unique_ptr<Target>> targets;
std::uint64_t next_handle = 1;
std::uint64_t total_reserved = 0;
}  // namespace

extern "C" {
__declspec(dllexport) std::uint32_t ctgn_abi_version() { return 1; }
__declspec(dllexport) std::uint64_t ctgn_fp_environment() {
    return fp_environment() & ~std::uint64_t(0x3f);
}
__declspec(dllexport) const char* ctgn_last_error() { return last_error; }

// Status: 0 success, 1 domain, 2 reservation, 3 stale handle, 4 native fault,
// 5 floating environment drift. A failure never returns usable partial results.
__declspec(dllexport) int ctgn_create(const double* values, std::uint64_t rows,
        std::uint64_t limit, std::uint64_t expected_fp,
        std::uint64_t* handle, std::uint64_t* reserved) noexcept {
    if (!handle || !reserved) return error(1, "Missing output pointers");
    *handle = 0; *reserved = 0;
    try {
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (!rows || !points_valid(values, rows)) return error(1, "Target domain");
        if (!fp_valid(expected_fp)) return error(5, "Floating environment");
        const auto needed = reservation(rows);
        if (needed > limit || needed > kMaxReservation - total_reserved ||
                targets.size() >= kMaxHandles || !next_handle)
            return error(2, "Target reservation");
        auto owner = std::make_unique<Target>(values, rows, expected_fp);
        if (!fp_valid(expected_fp)) return error(5, "Floating environment drift");
        const auto key = next_handle++;
        targets.emplace(key, std::move(owner));
        total_reserved += needed;
        *handle = key; *reserved = needed; last_error[0] = '\0'; return 0;
    } catch (const std::exception& e) { return error(4, e.what()); }
      catch (...) { return error(4, "Unknown create fault"); }
}

__declspec(dllexport) int ctgn_query(std::uint64_t handle, const double* queries,
        std::uint64_t rows, double radius, std::uint32_t threads,
        std::int64_t* ids, double* squared) noexcept {
    try {
        // Serial ownership deliberately covers searches and destruction.
        std::lock_guard<std::mutex> lock(registry_mutex);
        const auto found = targets.find(handle);
        if (found == targets.end()) return error(3, "Stale target handle");
        const auto& target = *found->second;
        if (!points_valid(queries, rows) || (rows && (!ids || !squared)) ||
                !std::isfinite(radius) || radius < 0x1p-20 || radius > 1.0 ||
                !(threads == 1 || threads == 2 || threads == 4 || threads == 8 || threads == 20))
            return error(1, "Query domain");
        if (!fp_valid(target.fp)) return error(5, "Floating environment drift");
        const double radius2 = radius * radius;
        std::atomic<int> failure{0};
        const auto count_workers = std::max<std::uint64_t>(1, std::min<std::uint64_t>(threads, rows));
        auto work = [&](std::uint64_t worker) noexcept {
            try {
                if (!fp_valid(target.fp)) { failure.store(5); return; }
                const auto first = rows * worker / count_workers;
                const auto last = rows * (worker + 1) / count_workers;
                for (std::uint64_t row = first; row < last && !failure.load(); ++row) {
                    Eigen::Index index = -1;
                    double d2 = std::numeric_limits<double>::infinity();
                    const auto count = target.tree->index_->knnSearch(queries + 3 * row, 1, &index, &d2);
                    if (count != 1 || index < 0 || std::uint64_t(index) >= target.rows ||
                            !std::isfinite(d2) || d2 < 0) { failure.store(4); return; }
                    // SearchHybrid(max_nn=1) uses lower_bound, hence strict radius.
                    const bool hit = std::lower_bound(&d2, &d2 + 1, radius2) == &d2 + 1;
                    ids[row] = hit ? std::int64_t(index) : -1;
                    squared[row] = hit ? d2 : std::numeric_limits<double>::infinity();
                }
                if (!fp_valid(target.fp)) failure.store(5);
            } catch (...) { failure.store(4); }
        };
        std::vector<std::thread> workers;
        workers.reserve(count_workers - 1);
        try {
            for (std::uint64_t worker = 1; worker < count_workers; ++worker)
                workers.emplace_back(work, worker);
            work(0);
        } catch (...) {
            failure.store(4);
            for (auto& worker : workers) worker.join();
            throw;
        }
        for (auto& worker : workers) worker.join();
        if (failure.load()) return error(failure.load(), "Query worker/environment fault");
        if (!fp_valid(target.fp)) return error(5, "Floating environment drift");
        last_error[0] = '\0'; return 0;
    } catch (const std::exception& e) { return error(4, e.what()); }
      catch (...) { return error(4, "Unknown query fault"); }
}

__declspec(dllexport) int ctgn_destroy(std::uint64_t handle) noexcept {
    try {
        std::lock_guard<std::mutex> lock(registry_mutex);
        const auto found = targets.find(handle);
        if (found == targets.end()) return error(3, "Stale target handle");
        total_reserved -= found->second->reserved;
        targets.erase(found); last_error[0] = '\0'; return 0;
    } catch (const std::exception& e) { return error(4, e.what()); }
      catch (...) { return error(4, "Unknown destroy fault"); }
}
}
