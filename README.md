# mesh-voxelize

**Fast CPU mesh-to-voxel conversion** — the scanline voxelization algorithm from [TRELLIS.2](https://github.com/JeffreyXiang/TRELLIS.2) / o-voxel, stripped down for maximum speed.

> 🚀 **4× faster** than Open3D &nbsp;|&nbsp; **3× faster** than o-voxel &nbsp;|&nbsp;
> 🧠 **Zero external C++ dependencies** &nbsp;|&nbsp;
> 📦 **One-command install from GitHub**

## Install

```bash
pip install git+https://github.com/yuyujunjun/mesh-voxelize.git
```

That's it. No system packages, no CUDA, no conda. Just a C++17 compiler (gcc/clang/MSVC — you almost certainly already have one). `pybind11` is pulled in automatically by pip.

## Quickstart

```python
import numpy as np
import mesh_voxelize

# vertices: (N, 3) float32, in [0, 1] range
# faces:    (M, 3) int32
verts = np.random.rand(1000, 3).astype(np.float32)
faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)

# Voxelize at 512³ resolution
voxels = mesh_voxelize.mesh_to_voxels(verts, faces, grid_size=512)
# → int32 array of shape (K, 3), one row per occupied voxel
```

### Normalizing your mesh

The function expects vertices in `[0, 1]` with grid `[0, grid_size]`. To normalize an arbitrary mesh:

```python
v_min = verts.min(axis=0)
v_max = verts.max(axis=0)
extent = (v_max - v_min).max()
center = (v_min + v_max) / 2
scale = 0.999 / extent
verts_norm = (verts - center) * scale + 0.5
```

## API

```python
mesh_voxelize.mesh_to_voxels(vertices, faces, grid_size)
```

| Arg | Type | Description |
|-----|------|-------------|
| `vertices` | `np.ndarray` (N, 3) float32 | Vertex positions in [0, 1] |
| `faces` | `np.ndarray` (M, 3) int32 | Triangle indices into vertices |
| `grid_size` | `int` | Voxels per dimension (e.g. 512, 1024) |

Returns: `np.ndarray` (K, 3) int32 — grid coordinates of occupied voxels.

## Performance

PegInsertionSide-v1 mesh (281K verts, 561K faces), Intel Xeon:

| Resolution | Voxels | mesh-voxelize | Open3D | o-voxel |
|-----------|--------|---------------|--------|---------|
| 512³ | 554,681 | **0.26s** | 1.32s (5.1×) | 1.14s (4.4×) |
| 1024³ | 2,294,410 | **1.35s** | 5.41s (4.0×) | 3.95s (2.9×) |

Identical voxel positions to o-voxel (100.00% agreement). Open3D produces 41 extra boundary voxels at 1024³.

### Where the speedup comes from

o-voxel's `mesh_to_flexible_dual_grid` does four steps. Only **step 1** discovers which voxels are occupied — steps 2–4 compute dual vertices needed for surface reconstruction. This library runs only step 1:

| Step | 1024³ time | What it does | We keep? |
|------|-----------|--------------|----------|
| 1. Intersect scanline | 1.35s (34%) | Find occupied voxels | ✅ |
| 2. Face QEF | 1.03s (26%) | Per-face coverage tests, accumulate QEF | ✂️ |
| 3. Boundary QEF | 0.17s (4%) | DDA edge traversal, accumulate QEF | ✂️ |
| 4. Solve QEF | 0.90s (23%) | Solve linear systems for dual vertices | ✂️ |
| **Total o-voxel** | **3.95s** | | |
| **mesh-voxelize** | **1.35s** | | |

If you need surface reconstruction (dual vertices / remeshing), use full o-voxel. If you only need voxel occupancy — for sparse 3D convolutions, occupancy grids, or neural field training — this library gives you the same result 3× faster with zero dependencies.

Equivalent Open3D code for reference:
```python
import open3d as o3d
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh, voxel_size=1.0/1024, min_bound=(0,0,0), max_bound=(1,1,1))
voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
```

## How it works

The algorithm is a **triple-axis scanline fill** (`intersect_qef` from o-voxel):

1. For each triangle, project onto 3 orthogonal planes (XY, YZ, ZX)
2. Scanline-rasterize the projected triangle
3. Mark the 4 voxels around each intersection point as occupied

This is fundamentally different from ray-casting or flood-fill — it guarantees watertight coverage without an interior test, and naturally handles open surfaces. The original o-voxel then spends ~66% of runtime computing QEF dual vertices, which we skip entirely.

## License

MIT. Core algorithm extracted from [o-voxel](https://github.com/JeffreyXiang/TRELLIS.2) by Jianfeng Xiang.

## Citation

If you use this in research, please cite TRELLIS.2:

```bibtex
@article{xiang2024trellis2,
  title={TRELLIS 2: Structured 3D Latents for Versatile 3D Generation},
  author={Xiang, Jianfeng and others},
  journal={arXiv preprint},
  year={2024}
}
```
