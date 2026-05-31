/*
 * GPU-accelerated mesh voxelization via CUDA (Schwarz & Seidel 2010).
 * Self-contained — no external dependencies beyond CUDA runtime.
 * C-callable API for ctypes / pybind11 / cffi.
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_80 -o _gpu.so _gpu.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// ── Missing float3/float2 operators (CUDA 12.x removed these) ──────────────

__host__ __device__ __inline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __inline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__host__ __device__ __inline__ float2 operator-(const float2& a) {
    return make_float2(-a.x, -a.y);
}
__host__ __device__ __inline__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __inline__ float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__host__ __device__ __inline__ float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__host__ __device__ __inline__ int3 clamp(const int3& v, const int3& lo, const int3& hi) {
    return make_int3(
        v.x < lo.x ? lo.x : (v.x > hi.x ? hi.x : v.x),
        v.y < lo.y ? lo.y : (v.y > hi.y ? hi.y : v.y),
        v.z < lo.z ? lo.z : (v.z > hi.z ? hi.z : v.z));
}

#define checkCudaErrors(call) do {                              \
    cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(_e));    \
        return -1;                                              \
    }                                                           \
} while(0)

// ── Types ──────────────────────────────────────────────────────────────────

template <typename T>
struct AABox {
    T min, max;
    __host__ __device__ AABox() : min(T()), max(T()) {}
    __host__ __device__ AABox(T _min, T _max) : min(_min), max(_max) {}
};

struct voxinfo {
    AABox<float3> bbox;
    uint3 gridsize;
    size_t n_triangles;
    float3 unit;

    __host__ __device__ voxinfo() {}
    __host__ __device__ voxinfo(const AABox<float3>& b, const uint3& gs, size_t nt)
        : gridsize(gs), bbox(b), n_triangles(nt) {
        unit.x = (bbox.max.x - bbox.min.x) / float(gridsize.x);
        unit.y = (bbox.max.y - bbox.min.y) / float(gridsize.y);
        unit.z = (bbox.max.z - bbox.min.z) / float(gridsize.z);
    }
};

__host__ __device__ __inline__ int3 float3_to_int3(const float3 a) {
    return make_int3(static_cast<int>(a.x),
                     static_cast<int>(a.y),
                     static_cast<int>(a.z));
}

__device__ __inline__ void setBit(unsigned int* vtable, size_t index) {
    size_t int_location = index / size_t(32);
    unsigned int bit_pos = size_t(31) - (index % size_t(32));
    atomicOr(&(vtable[int_location]), 1U << bit_pos);
}

// ── Voxelization kernel ─────────────────────────────────────────────────────

__global__ void voxelize_kernel(voxinfo info, const float* triangle_data,
                                unsigned int* voxel_table)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    float3 delta_p = make_float3(info.unit.x, info.unit.y, info.unit.z);
    int3 grid_max = make_int3(info.gridsize.x - 1,
                               info.gridsize.y - 1,
                               info.gridsize.z - 1);

    while (tid < info.n_triangles) {
        size_t t = tid * 9;
        float3 v0 = make_float3(triangle_data[t+0], triangle_data[t+1],
                                triangle_data[t+2]) - info.bbox.min;
        float3 v1 = make_float3(triangle_data[t+3], triangle_data[t+4],
                                triangle_data[t+5]) - info.bbox.min;
        float3 v2 = make_float3(triangle_data[t+6], triangle_data[t+7],
                                triangle_data[t+8]) - info.bbox.min;

        float3 e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
        float3 n = make_float3(0, 0, 0);
        {
            float3 cr = make_float3(e0.y*e1.z - e0.z*e1.y,
                                    e0.z*e1.x - e0.x*e1.z,
                                    e0.x*e1.y - e0.y*e1.x);
            float len = sqrtf(cr.x*cr.x + cr.y*cr.y + cr.z*cr.z);
            if (len > 1e-20f) {
                n.x = cr.x / len; n.y = cr.y / len; n.z = cr.z / len;
            }
        }

        AABox<float3> t_world(fminf(v0, fminf(v1, v2)),
                              fmaxf(v0, fmaxf(v1, v2)));
        AABox<int3> t_grid;
        t_grid.min = clamp(float3_to_int3(
            make_float3(t_world.min.x / info.unit.x,
                        t_world.min.y / info.unit.y,
                        t_world.min.z / info.unit.z)),
            make_int3(0,0,0), grid_max);
        t_grid.max = clamp(float3_to_int3(
            make_float3(t_world.max.x / info.unit.x,
                        t_world.max.y / info.unit.y,
                        t_world.max.z / info.unit.z)),
            make_int3(0,0,0), grid_max);

        // Plane test
        float3 c = make_float3(0.0f, 0.0f, 0.0f);
        if (n.x > 0.0f) c.x = info.unit.x;
        if (n.y > 0.0f) c.y = info.unit.y;
        if (n.z > 0.0f) c.z = info.unit.z;
        float d1 = n.x*(c.x-v0.x) + n.y*(c.y-v0.y) + n.z*(c.z-v0.z);
        float d2 = n.x*(delta_p.x-c.x-v0.x) + n.y*(delta_p.y-c.y-v0.y) + n.z*(delta_p.z-c.z-v0.z);

        // Projection tests – XY
        float2 n_xy_e0 = make_float2(-e0.y, e0.x);
        float2 n_xy_e1 = make_float2(-e1.y, e1.x);
        float2 n_xy_e2 = make_float2(-e2.y, e2.x);
        if (n.z < 0.0f) { n_xy_e0 = -n_xy_e0; n_xy_e1 = -n_xy_e1; n_xy_e2 = -n_xy_e2; }
        float d_xy_e0 = -(n_xy_e0.x*v0.x + n_xy_e0.y*v0.y) + fmaxf(0.f, info.unit.x*n_xy_e0.x) + fmaxf(0.f, info.unit.y*n_xy_e0.y);
        float d_xy_e1 = -(n_xy_e1.x*v1.x + n_xy_e1.y*v1.y) + fmaxf(0.f, info.unit.x*n_xy_e1.x) + fmaxf(0.f, info.unit.y*n_xy_e1.y);
        float d_xy_e2 = -(n_xy_e2.x*v2.x + n_xy_e2.y*v2.y) + fmaxf(0.f, info.unit.x*n_xy_e2.x) + fmaxf(0.f, info.unit.y*n_xy_e2.y);

        // YZ
        float2 n_yz_e0 = make_float2(-e0.z, e0.y);
        float2 n_yz_e1 = make_float2(-e1.z, e1.y);
        float2 n_yz_e2 = make_float2(-e2.z, e2.y);
        if (n.x < 0.0f) { n_yz_e0 = -n_yz_e0; n_yz_e1 = -n_yz_e1; n_yz_e2 = -n_yz_e2; }
        float d_yz_e0 = -(n_yz_e0.x*v0.y + n_yz_e0.y*v0.z) + fmaxf(0.f, info.unit.y*n_yz_e0.x) + fmaxf(0.f, info.unit.z*n_yz_e0.y);
        float d_yz_e1 = -(n_yz_e1.x*v1.y + n_yz_e1.y*v1.z) + fmaxf(0.f, info.unit.y*n_yz_e1.x) + fmaxf(0.f, info.unit.z*n_yz_e1.y);
        float d_yz_e2 = -(n_yz_e2.x*v2.y + n_yz_e2.y*v2.z) + fmaxf(0.f, info.unit.y*n_yz_e2.x) + fmaxf(0.f, info.unit.z*n_yz_e2.y);

        // ZX
        float2 n_zx_e0 = make_float2(-e0.x, e0.z);
        float2 n_zx_e1 = make_float2(-e1.x, e1.z);
        float2 n_zx_e2 = make_float2(-e2.x, e2.z);
        if (n.y < 0.0f) { n_zx_e0 = -n_zx_e0; n_zx_e1 = -n_zx_e1; n_zx_e2 = -n_zx_e2; }
        float d_xz_e0 = -(n_zx_e0.x*v0.z + n_zx_e0.y*v0.x) + fmaxf(0.f, info.unit.x*n_zx_e0.x) + fmaxf(0.f, info.unit.z*n_zx_e0.y);
        float d_xz_e1 = -(n_zx_e1.x*v1.z + n_zx_e1.y*v1.x) + fmaxf(0.f, info.unit.x*n_zx_e1.x) + fmaxf(0.f, info.unit.z*n_zx_e1.y);
        float d_xz_e2 = -(n_zx_e2.x*v2.z + n_zx_e2.y*v2.x) + fmaxf(0.f, info.unit.x*n_zx_e2.x) + fmaxf(0.f, info.unit.z*n_zx_e2.y);

        for (int z = t_grid.min.z; z <= t_grid.max.z; z++) {
            for (int y = t_grid.min.y; y <= t_grid.max.y; y++) {
                for (int x = t_grid.min.x; x <= t_grid.max.x; x++) {
                    float3 p = make_float3(x*info.unit.x, y*info.unit.y, z*info.unit.z);
                    float nDOTp = n.x*p.x + n.y*p.y + n.z*p.z;
                    if (((nDOTp + d1) * (nDOTp + d2)) > 0.0f) continue;

                    float2 p_xy = make_float2(p.x, p.y);
                    if ((n_xy_e0.x*p_xy.x + n_xy_e0.y*p_xy.y + d_xy_e0) < 0.0f) continue;
                    if ((n_xy_e1.x*p_xy.x + n_xy_e1.y*p_xy.y + d_xy_e1) < 0.0f) continue;
                    if ((n_xy_e2.x*p_xy.x + n_xy_e2.y*p_xy.y + d_xy_e2) < 0.0f) continue;

                    float2 p_yz = make_float2(p.y, p.z);
                    if ((n_yz_e0.x*p_yz.x + n_yz_e0.y*p_yz.y + d_yz_e0) < 0.0f) continue;
                    if ((n_yz_e1.x*p_yz.x + n_yz_e1.y*p_yz.y + d_yz_e1) < 0.0f) continue;
                    if ((n_yz_e2.x*p_yz.x + n_yz_e2.y*p_yz.y + d_yz_e2) < 0.0f) continue;

                    float2 p_zx = make_float2(p.z, p.x);
                    if ((n_zx_e0.x*p_zx.x + n_zx_e0.y*p_zx.y + d_xz_e0) < 0.0f) continue;
                    if ((n_zx_e1.x*p_zx.x + n_zx_e1.y*p_zx.y + d_xz_e1) < 0.0f) continue;
                    if ((n_zx_e2.x*p_zx.x + n_zx_e2.y*p_zx.y + d_xz_e2) < 0.0f) continue;

                    size_t loc = (size_t)x + ((size_t)y)*info.gridsize.x
                               + ((size_t)z)*info.gridsize.x*info.gridsize.y;
                    setBit(voxel_table, loc);
                }
            }
        }
        tid += stride;
    }
}

// ── Sparse extraction kernel ───────────────────────────────────────────────

__global__ void extract_kernel(const unsigned int* vtable, uint3 gridsize,
                               int* out_coords, int* out_counter)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridsize.x * gridsize.y * gridsize.z;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total; i += stride) {
        size_t int_loc = i / 32;
        unsigned int bit = 31 - (i % 32);
        if (vtable[int_loc] & (1U << bit)) {
            int pos = atomicAdd(out_counter, 1);
            out_coords[pos * 3 + 0] = (int)(i % gridsize.x);
            out_coords[pos * 3 + 1] = (int)((i / gridsize.x) % gridsize.y);
            out_coords[pos * 3 + 2] = (int)(i / ((size_t)gridsize.x * gridsize.y));
        }
    }
}

// ── Public C API ───────────────────────────────────────────────────────────

extern "C" {

int mesh_to_voxels_gpu(
    const float* vertices, int n_verts,
    const int*   faces,    int n_faces,
    int resolution,
    int** out_coords, int* out_count)
{
    *out_coords = nullptr;
    *out_count  = 0;

    // Build triangle_data (9 floats per triangle)
    float *h_tri = (float*)malloc((size_t)n_faces * 9 * sizeof(float));
    if (!h_tri) return -1;
    for (int i = 0; i < n_faces; i++) {
        int i0 = faces[i*3], i1 = faces[i*3+1], i2 = faces[i*3+2];
        float *t = h_tri + i * 9;
        t[0]=vertices[i0*3];   t[1]=vertices[i0*3+1];   t[2]=vertices[i0*3+2];
        t[3]=vertices[i1*3];   t[4]=vertices[i1*3+1];   t[5]=vertices[i1*3+2];
        t[6]=vertices[i2*3];   t[7]=vertices[i2*3+1];   t[8]=vertices[i2*3+2];
    }
    float *d_tri;
    checkCudaErrors(cudaMalloc(&d_tri, (size_t)n_faces * 9 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_tri, h_tri,
                                (size_t)n_faces * 9 * sizeof(float),
                                cudaMemcpyHostToDevice));
    free(h_tri);

    // Voxel bit grid
    size_t grid_bits  = (size_t)resolution * resolution * resolution;
    size_t grid_words = (grid_bits + 31) / 32;
    unsigned int *d_vtable;
    checkCudaErrors(cudaMalloc(&d_vtable, grid_words * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_vtable, 0, grid_words * sizeof(unsigned int)));

    // voxinfo: bbox = [0,1] exactly (match CPU normalisation)
    AABox<float3> bbox(make_float3(0.f,0.f,0.f), make_float3(1.f,1.f,1.f));
    uint3 gs = make_uint3(resolution, resolution, resolution);
    voxinfo info(bbox, gs, (size_t)n_faces);

    // Launch voxelization
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       voxelize_kernel, 0, 0);
    int gridSize = (n_faces + blockSize - 1) / blockSize;
    voxelize_kernel<<<gridSize, blockSize>>>(info, d_tri, d_vtable);
    cudaDeviceSynchronize();
    cudaFree(d_tri);

    // Extract sparse coords
    int *d_counter, *d_coords;
    checkCudaErrors(cudaMalloc(&d_counter, sizeof(int)));
    checkCudaErrors(cudaMemset(d_counter, 0, sizeof(int)));
    int max_voxels = (int)(grid_bits / 15);
    checkCudaErrors(cudaMalloc(&d_coords, (size_t)max_voxels * 3 * sizeof(int)));

    int extract_blocks = (int)((grid_bits + 255) / 256);
    if (extract_blocks > 65535) extract_blocks = 65535;
    extract_kernel<<<extract_blocks, 256>>>(d_vtable, gs, d_coords, d_counter);
    cudaDeviceSynchronize();

    int n_voxels;
    checkCudaErrors(cudaMemcpy(&n_voxels, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    int *h_coords = (int*)malloc((size_t)n_voxels * 3 * sizeof(int));
    if (!h_coords) { cudaFree(d_vtable); cudaFree(d_counter); cudaFree(d_coords); return -1; }
    checkCudaErrors(cudaMemcpy(h_coords, d_coords, (size_t)n_voxels * 3 * sizeof(int),
                                cudaMemcpyDeviceToHost));

    cudaFree(d_vtable); cudaFree(d_counter); cudaFree(d_coords);

    *out_coords = h_coords;
    *out_count  = n_voxels;
    return 0;
}

void mesh_to_voxels_gpu_free(int* coords) { free(coords); }

} // extern "C"
