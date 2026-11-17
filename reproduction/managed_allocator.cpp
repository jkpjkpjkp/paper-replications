// CUDA storage-only fallback for full-batch experiments that exceed free VRAM.
// No model, optimizer, dtype, batching or RNG changes. cudaFree synchronizes.
// Uses the documented CUDAPluggableAllocator ABI, verified against torch 2.4.1 headers.
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <sys/types.h>

// CUDA runtime wheel lacks the nvcc crt headers. These are the public runtime
// ABI declarations/constants from its cuda_runtime_api.h and driver_types.h.
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr unsigned cudaMemAttachGlobal = 0x01;
constexpr int cudaDevAttrConcurrentManagedAccess = 89;
extern "C" int cudaSetDevice(int);
extern "C" int cudaMallocManaged(void**, size_t, unsigned);
extern "C" int cudaFree(void*);
extern "C" const char* cudaGetErrorString(int);
extern "C" int cudaDeviceGetAttribute(int*, int, int);

static std::atomic<size_t> live_bytes{0}, peak_bytes{0};
static constexpr size_t LIMIT = size_t(10) * 1024 * 1024 * 1024;
extern "C" {
void* sit_managed_malloc(ssize_t size, int device, cudaStream_t stream) {
    if (size <= 0) return nullptr;
    auto now = live_bytes.fetch_add(size) + size;
    if (now > LIMIT) {
        live_bytes.fetch_sub(size);
        std::fprintf(stderr, "SIT managed-memory safety cap exceeded\n");
        return nullptr;
    }
    cudaSetDevice(device);
    void* ptr = nullptr;
    auto err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        live_bytes.fetch_sub(size);
        std::fprintf(stderr, "SIT cudaMallocManaged: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    auto old = peak_bytes.load();
    while (old < now && !peak_bytes.compare_exchange_weak(old, now)) {}
    return ptr;
}
void sit_managed_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (!ptr) return;
    cudaSetDevice(device);
    auto err = cudaFree(ptr);
    if (err != cudaSuccess) std::fprintf(stderr, "SIT cudaFree: %s\n", cudaGetErrorString(err));
    live_bytes.fetch_sub(size);
}
size_t sit_managed_peak_bytes() { return peak_bytes.load(); }
int sit_managed_supported() {
    int supported = 0;
    cudaDeviceGetAttribute(&supported, cudaDevAttrConcurrentManagedAccess, 0);
    return supported;
}
}
