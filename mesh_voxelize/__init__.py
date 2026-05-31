from ._core import mesh_to_voxels

__all__ = ["mesh_to_voxels"]
__version__ = "0.2.0"

# GPU variant — optional, requires _gpu.so (make -C mesh_voxelize)
try:
    from .gpu import mesh_to_voxels as mesh_to_voxels_gpu
    __all__.append("mesh_to_voxels_gpu")
except (FileNotFoundError, OSError):
    pass
