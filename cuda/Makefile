NVCCFLAGS ?= -O3 -std=c++17
CUTLASS_PATH ?= /usr/local/cutlass
CUTLASS_INCLUDES = -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

all: build

build: matrix_add.out vector_add.out vector_add_simple.out flash_attention_cuda.out cutlass_attention.out

matrix_add.out: matrix_add.cu
	nvcc $(NVCCFLAGS) -o matrix_add.out matrix_add.cu

vector_add.out: vector_add.cu
	nvcc $(NVCCFLAGS) -o vector_add.out vector_add.cu

vector_add_simple.out: vector_add_simple.cu
	nvcc $(NVCCFLAGS) -o vector_add_simple.out vector_add_simple.cu

flash_attention_cuda.out: flash_attention_cuda.cu
	nvcc $(NVCCFLAGS) -o flash_attention_cuda.out flash_attention_cuda.cu

cutlass_attention.out: cutlass_attention.cu
	@test -f "$(CUTLASS_PATH)/include/cutlass/gemm/device/gemm.h" || (echo "CUTLASS not found. Set CUTLASS_PATH=/path/to/cutlass"; exit 1)
	nvcc $(NVCCFLAGS) $(CUTLASS_INCLUDES) -o cutlass_attention.out cutlass_attention.cu

clean:
	rm -f vector_add.out
	rm -f matrix_add.out
	rm -f vector_add_simple.out
	rm -f flash_attention_cuda.out
	rm -f cutlass_attention.out

run: build
	./vector_add.out
	./matrix_add.out
	./vector_add_simple.out
	./flash_attention_cuda.out
	./cutlass_attention.out