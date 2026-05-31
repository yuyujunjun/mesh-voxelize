# mesh-voxelize

CPU + GPU mesh voxelization extracted from [TRELLIS.2](https://github.com/JeffreyXiang/TRELLIS.2) / o-voxel. GPU variant (CUDA) gives 3× speedup vs CPU at 99.99% IoU.

## Install

```bash
pip install git+https://github.com/yuyujunjun/mesh-voxelize.git
```

Requires a C++17 compiler. `pybind11` is pulled in automatically.

### GPU variant (optional)

The GPU backend requires CUDA 12+ and an NVIDIA GPU (compute capability ≥ 8.0).

```bash
cd mesh_voxelize && make CUDAARCHS=80
```

This builds `_gpu.so` alongside the Python package. No extra Python dependencies.

## Usage

### CPU

```python
import numpy as np
import mesh_voxelize

verts = np.random.rand(1000, 3).astype(np.float32)   # (N, 3) in [0, 1]
faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)  # (M, 3)

voxels = mesh_voxelize.mesh_to_voxels(verts, faces, grid_size=512)
# -> (K, 3) int32, occupied voxel grid coordinates
```

### GPU

```python
voxels = mesh_voxelize.mesh_to_voxels_gpu(verts, faces, grid_size=512)
# -> same interface, same result, 3× faster
```

`mesh_to_voxels_gpu` is available when `_gpu.so` has been built. If missing, the import is silently skipped — CPU `mesh_to_voxels` remains available.

To normalize an arbitrary mesh to [0, 1]:

```python
v_min, v_max = verts.min(axis=0), verts.max(axis=0)
extent = (v_max - v_min).max()
center = (v_min + v_max) / 2
verts_norm = (verts - center) * (0.999 / extent) + 0.5
```

## API

```python
mesh_voxelize.mesh_to_voxels(vertices, faces, grid_size)
mesh_voxelize.mesh_to_voxels_gpu(vertices, faces, grid_size)  # requires CUDA build
```

| Arg | Type | Description |
|-----|------|-------------|
| `vertices` | `np.ndarray` (N, 3) float32 | Vertex positions in [0, 1] |
| `faces` | `np.ndarray` (M, 3) int32 | Triangle indices |
| `grid_size` | `int` | Voxels per dimension |

Returns `np.ndarray` (K, 3) int32 of occupied voxel coordinates. Both functions produce identical results (IoU ≥ 99.99%).

## Performance

PegInsertionSide-v1 mesh (281K verts, 561K faces):

| Resolution | Voxels | CPU | GPU (A100) | Open3D | o-voxel (full) |
|-----------|--------|-----|------------|--------|----------------|
| 512³ | 554,681 | 0.26s | 0.25s | 1.32s | 1.14s |
| 1024³ | 2,294,410 | 1.35s | 0.56s | 5.41s | 3.95s |

CPU: Intel Xeon. GPU: NVIDIA A100-SXM4-80GB. GPU time includes host↔device transfer.

Voxel positions identical to o-voxel.

## Relation to o-voxel

This library runs only the first step of o-voxel's `mesh_to_flexible_dual_grid` — the intersect scanline that discovers occupied voxels. Steps 2–4 (face QEF, boundary QEF, dual vertex solve) are omitted:

| Step | 1024³ | Purpose |
|------|-------|---------|
| Intersect scanline | 1.35s (34%) | Find occupied voxels ← this library |
| Face QEF | 1.03s (26%) | Per-face QEF accumulation |
| Boundary QEF | 0.17s (4%) | Edge traversal QEF |
| Dual vertex solve | 0.90s (23%) | Linear system solve |
| **Total** | **3.95s** | |

If you need surface reconstruction via dual vertices, use full o-voxel. For occupancy grids or sparse 3D convolution inputs, this library produces the same voxel set.

## License

MIT. Algorithm from [o-voxel](https://github.com/JeffreyXiang/TRELLIS.2) by Jianfeng Xiang.

## Citation

```bibtex
@article{xiang2024trellis2,
  title={TRELLIS 2: Structured 3D Latents for Versatile 3D Generation},
  author={Xiang, Jianfeng and others},
  journal={arXiv preprint},
  year={2024}
}
```
