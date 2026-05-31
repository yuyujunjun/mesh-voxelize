/*
 * mesh-voxelize: Fast CPU mesh voxelization based on TRELLIS.2 / o-voxel.
 *
 * Stripped-down intersect scanline algorithm — only voxel occupancy.
 * No dual grid, no QEF, no external dependencies beyond C++17 STL.
 *
 * Original: o-voxel/src/convert/flexible_dual_grid.cpp (MIT License)
 * Author: Jianfeng XIANG <belljig@outlook.com>
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ── Minimal vector types (replace Eigen to stay dependency-free) ──────────

struct Vec2d {
    double x, y;
    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }
};

struct Vec3d {
    double x, y, z;
    Vec3d() = default;
    Vec3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    double& operator[](int i) { return (&x)[i]; }
    const double& operator[](int i) const { return (&x)[i]; }
};

struct Vec3f {
    float x, y, z;
    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};

struct Vec3i {
    int x, y, z;
    int& operator[](int i) { return (&x)[i]; }
    const int& operator[](int i) const { return (&x)[i]; }
};

struct VoxelCoord {
    int x, y, z;
    int& operator[](int i) { return (&x)[i]; }
    const int& operator[](int i) const { return (&x)[i]; }
    bool operator==(const VoxelCoord& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

namespace std {
template <> struct hash<VoxelCoord> {
    size_t operator()(const VoxelCoord& v) const {
        const size_t p1 = 73856093, p2 = 19349663, p3 = 83492791;
        return (size_t)v.x * p1 ^ (size_t)v.y * p2 ^ (size_t)v.z * p3;
    }
};
}  // namespace std

// ── lerp ──────────────────────────────────────────────────────────────────

template <typename T, typename U>
static inline U lerp(const T& a, const T& b, const T& t,
                     const U& val_a, const U& val_b) {
    if (a == b) return val_a;
    T alpha = (t - a) / (b - a);
    U result;
    result[0] = (1 - alpha) * val_a[0] + alpha * val_b[0];
    result[1] = (1 - alpha) * val_a[1] + alpha * val_b[1];
    return result;
}

// ── Intersect scanline (stripped — no QEF) ───────────────────────────────

static void intersect_scanline(
    const Vec3f& voxel_size,
    const Vec3i& grid_min,
    const Vec3i& grid_max,
    const std::vector<Vec3f>& triangles,  // 3 vertices per triangle
    std::unordered_map<VoxelCoord, size_t>& hash_table,
    std::vector<VoxelCoord>& voxels)
{
    const size_t N_tri = triangles.size() / 3;

    for (size_t i = 0; i < N_tri; ++i) {
        const Vec3f& v0 = triangles[i * 3 + 0];
        const Vec3f& v1 = triangles[i * 3 + 1];
        const Vec3f& v2 = triangles[i * 3 + 2];

        auto scan_line_fill = [&](const int ax2) {
            int ax0 = (ax2 + 1) % 3;
            int ax1 = (ax2 + 2) % 3;

            std::array<Vec3d, 3> t = {
                Vec3d(v0[ax0], v0[ax1], v0[ax2]),
                Vec3d(v1[ax0], v1[ax1], v1[ax2]),
                Vec3d(v2[ax0], v2[ax1], v2[ax2])
            };
            std::sort(t.begin(), t.end(),
                [](const Vec3d& a, const Vec3d& b) { return a.y < b.y; });

            int start = std::clamp(
                (int)(t[0].y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int mid   = std::clamp(
                (int)(t[1].y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);
            int end   = std::clamp(
                (int)(t[2].y / voxel_size[ax1]), grid_min[ax1], grid_max[ax1] - 1);

            auto scan_line_half = [&](
                int row_start, int row_end,
                const Vec3d& t0, const Vec3d& t1, const Vec3d& t2)
            {
                for (int y_idx = row_start; y_idx < row_end; ++y_idx) {
                    double y = (y_idx + 1) * voxel_size[ax1];
                    Vec2d t3 = lerp(t0.y, t1.y, y,
                                    Vec2d{t0.x, t0.z},
                                    Vec2d{t1.x, t1.z});
                    Vec2d t4 = lerp(t0.y, t2.y, y,
                                    Vec2d{t0.x, t0.z},
                                    Vec2d{t2.x, t2.z});
                    if (t3.x > t4.x) std::swap(t3, t4);

                    int line_start = std::clamp(
                        (int)(t3.x / voxel_size[ax0]),
                        grid_min[ax0], grid_max[ax0] - 1);
                    int line_end = std::clamp(
                        (int)(t4.x / voxel_size[ax0]),
                        grid_min[ax0], grid_max[ax0] - 1);

                    for (int x_idx = line_start; x_idx < line_end; ++x_idx) {
                        double x = (x_idx + 1) * voxel_size[ax0];
                        double z = lerp(t3.x, t4.x, x,
                                        Vec2d{t3.y, 0.0},
                                        Vec2d{t4.y, 0.0}).x;
                        int z_idx = (int)(z / voxel_size[ax2]);

                        if (z_idx >= grid_min[ax2] && z_idx < grid_max[ax2]) {
                            for (int dx = 0; dx < 2; ++dx) {
                                for (int dy = 0; dy < 2; ++dy) {
                                    VoxelCoord coord;
                                    coord[ax0] = x_idx + dx;
                                    coord[ax1] = y_idx + dy;
                                    coord[ax2] = z_idx;
                                    if (hash_table.find(coord) == hash_table.end()) {
                                        hash_table[coord] = voxels.size();
                                        voxels.push_back(coord);
                                    }
                                }
                            }
                        }
                    }
                }
            };
            scan_line_half(start, mid, t[0], t[1], t[2]);
            scan_line_half(mid, end, t[2], t[1], t[0]);
        };
        scan_line_fill(0);
        scan_line_fill(1);
        scan_line_fill(2);
    }
}

// ── pybind11 module ───────────────────────────────────────────────────────

py::array_t<int> mesh_to_voxels(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> faces,
    int grid_size)
{
    // Read input arrays
    auto v_buf = vertices.unchecked<2>();
    auto f_buf = faces.unchecked<2>();

    int num_vertices = v_buf.shape(0);
    int num_faces = f_buf.shape(0);

    // Build triangles vector
    std::vector<Vec3f> triangles;
    triangles.reserve(num_faces * 3);
    for (int fi = 0; fi < num_faces; ++fi) {
        for (int vi = 0; vi < 3; ++vi) {
            int idx = f_buf(fi, vi);
            triangles.push_back(Vec3f{
                v_buf(idx, 0), v_buf(idx, 1), v_buf(idx, 2)});
        }
    }

    // Setup grid
    float vs = 1.0f / grid_size;
    Vec3f voxel_size{vs, vs, vs};
    Vec3i grid_min{0, 0, 0};
    Vec3i grid_max{grid_size, grid_size, grid_size};

    // Voxelize
    std::unordered_map<VoxelCoord, size_t> hash_table;
    std::vector<VoxelCoord> voxels;
    intersect_scanline(voxel_size, grid_min, grid_max, triangles,
                       hash_table, voxels);

    // Return as numpy array (N, 3) int32
    auto result = py::array_t<int>({static_cast<ssize_t>(voxels.size()), ssize_t(3)});
    auto r_buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < voxels.size(); ++i) {
        r_buf(i, 0) = voxels[i].x;
        r_buf(i, 1) = voxels[i].y;
        r_buf(i, 2) = voxels[i].z;
    }

    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast CPU mesh voxelization from TRELLIS.2 / o-voxel";
    m.def("mesh_to_voxels", &mesh_to_voxels,
          py::arg("vertices"), py::arg("faces"), py::arg("grid_size"),
          R"(
Convert a triangle mesh to occupied voxel indices.

Args:
    vertices: float32 numpy array of shape (N, 3), in [0, 1] range.
    faces: int32 numpy array of shape (M, 3), indexing into vertices.
    grid_size: int, number of voxels per dimension (e.g. 512, 1024).

Returns:
    int32 numpy array of shape (K, 3), occupied voxel grid coordinates.

Example:
    >>> import mesh_voxelize
    >>> import numpy as np
    >>> verts = np.random.rand(100, 3).astype(np.float32)
    >>> faces = np.random.randint(0, 100, (50, 3)).astype(np.int32)
    >>> voxels = mesh_voxelize.mesh_to_voxels(verts, faces, 512)
)");
}
