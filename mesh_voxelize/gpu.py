"""GPU-accelerated mesh voxelization (CUDA).

Usage:
    from mesh_voxelize.gpu import mesh_to_voxels

    voxels = mesh_to_voxels(vertices, faces, resolution)
    # vertices: float32 (V, 3)  – must be in [0, 1]
    # faces:    int32   (F, 3)
    # returns:  int32   (N, 3)  – voxel indices in [0, resolution)
"""

import ctypes
import numpy as np
import os

_SO_DIR = os.path.dirname(os.path.abspath(__file__))
_lib = None  # lazy load


def _get_lib():
    global _lib
    if _lib is None:
        so_path = os.path.join(_SO_DIR, "_gpu.so")
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"_gpu.so not found at {so_path}. "
                "Build it with: make -C mesh_voxelize"
            )
        _lib = ctypes.CDLL(so_path)
        _lib.mesh_to_voxels_gpu.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),   ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        _lib.mesh_to_voxels_gpu.restype = ctypes.c_int
        _lib.mesh_to_voxels_gpu_free.argtypes = [ctypes.c_void_p]
        _lib.mesh_to_voxels_gpu_free.restype = None
    return _lib


def mesh_to_voxels(vertices: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    """GPU voxelize a mesh. Returns (N,3) int32 voxel indices."""
    lib = _get_lib()
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces    = np.ascontiguousarray(faces,    dtype=np.int32)

    out_ptr = ctypes.c_void_p()
    out_count = ctypes.c_int(0)
    ret = lib.mesh_to_voxels_gpu(
        vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(vertices),
        faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(faces),
        resolution,
        ctypes.byref(out_ptr),
        ctypes.byref(out_count),
    )
    if ret != 0:
        raise RuntimeError(f"mesh_to_voxels_gpu failed with code {ret}")

    n = out_count.value
    buf = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int * (n * 3))).contents
    arr = np.frombuffer(buf, dtype=np.int32).reshape(n, 3).copy()
    lib.mesh_to_voxels_gpu_free(out_ptr)
    return arr
