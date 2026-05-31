# Build GPU voxelization shared library.
# Requires: nvcc, CUDA toolkit >= 12.x
#
# Usage:
#   make                   # build _gpu.so
#   make clean             # remove _gpu.so
#   CUDAARCHS="80" make   # target specific arch (A100 = 80)
#
# To build during pip install:
#   pip install . && make -C mesh_voxelize

CUDAARCHS ?= 80
NVCC ?= nvcc
NVCCFLAGS = -shared -Xcompiler -fPIC -O3 -arch=sm_$(CUDAARCHS)

_gpu.so: _gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f _gpu.so
