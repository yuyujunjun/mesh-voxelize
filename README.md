# mesh-voxelize

CPU mesh voxelization extracted from [TRELLIS.2](https://github.com/JeffreyXiang/TRELLIS.2) / o-voxel.

## Install

```bash
pip install git+https://github.com/yuyujunjun/mesh-voxelize.git
```

Requires a C++17 compiler. `pybind11` is pulled in automatically.

## Usage

```python
import numpy as np
import mesh_voxelize

verts = np.random.rand(1000, 3).astype(np.float32)   # (N, 3) in [0, 1]
faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)  # (M, 3)

voxels = mesh_voxelize.mesh_to_voxels(verts, faces, grid_size=512)
# -> (K, 3) int32, occupied voxel grid coordinates
```

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
```

| Arg | Type | Description |
|-----|------|-------------|
| `vertices` | `np.ndarray` (N, 3) float32 | Vertex positions in [0, 1] |
| `faces` | `np.ndarray` (M, 3) int32 | Triangle indices |
| `grid_size` | `int` | Voxels per dimension |

Returns `np.ndarray` (K, 3) int32 of occupied voxel coordinates.

## Performance

PegInsertionSide-v1 mesh (281K verts, 561K faces), Intel Xeon:

| Resolution | Voxels | mesh-voxelize | Open3D | o-voxel (full) |
|-----------|--------|---------------|--------|----------------|
| 512³ | 554,681 | 0.26s | 1.32s | 1.14s |
| 1024³ | 2,294,410 | 1.35s | 5.41s | 3.95s |

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
