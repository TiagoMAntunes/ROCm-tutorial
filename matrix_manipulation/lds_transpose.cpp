#include "hip/hip_runtime.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define THREADS_PER_BLOCK 256

const static int tile_dim = 32;

__global__ void transpose_lds_kernel(float * in, float * out, int width, int height) {
    
    __shared__ float tile[tile_dim][tile_dim];
    
    int x_tile_index = blockIdx.x * tile_dim;
    int y_tile_index = blockIdx.y * tile_dim;

    int in_index = (y_tile_index + threadIdx.y) * width + (x_tile_index + threadIdx.x);
    int out_index = (x_tile_index + threadIdx.y) * width + (y_tile_index + threadIdx.x);
    
    if (in_index >= width * height || out_index >= width * height) return;

    tile[threadIdx.y][threadIdx.x] = in[in_index];

    __syncthreads();

    out[out_index] = tile[threadIdx.x][threadIdx.y];
}


int main() {
    float * h_in, * h_out;
    float * d_in, * d_out;

    int WIDTH = 4096;
    int HEIGHT = 4096;
    size_t DATA_SIZE = WIDTH * HEIGHT * sizeof(float);
    
    // allocate host memory
    h_in  = (float *) malloc(DATA_SIZE);
    h_out = (float *) malloc(DATA_SIZE);

    // initialize host memory
    for (int i = 0; i < WIDTH * HEIGHT; i++) 
        h_in[i] = i;

    // debugging
    for (int i = 0; i < WIDTH * HEIGHT; i++) 
        h_out[i] = 0;

    // allocate device memory
    HIP_ASSERT(hipMalloc(&d_in, DATA_SIZE));
    HIP_ASSERT(hipMalloc(&d_out, DATA_SIZE));

    // initialize device memory
    HIP_ASSERT(hipMemcpy(d_in, h_in, DATA_SIZE, hipMemcpyHostToDevice));
    HIP_ASSERT(hipDeviceSynchronize());

    // GPU kernel launch
    // We tile the matrix into blocks (x0,y0,x1,y1) which have an horizontal dimension and vertical dimension
    int x_tile = tile_dim;
    int y_tile = tile_dim; // each tile has 64 elements

    hipLaunchKernelGGL(transpose_lds_kernel,
                    dim3(WIDTH / x_tile + 1, HEIGHT / y_tile + 1),
                    dim3(x_tile, y_tile),
                    0, 0,
                    d_in, d_out, WIDTH, HEIGHT
    );
    HIP_ASSERT(hipDeviceSynchronize());

    // copy memory back into host
    HIP_ASSERT(hipMemcpy(h_out, d_out, DATA_SIZE, hipMemcpyDeviceToHost));

    // validate data
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
        if (abs(h_out[j * WIDTH + i] - h_in[i * WIDTH + j]) >= 1e-5)
            std::cout << "Error at position " << i * WIDTH + j << std::endl;

    // release device memory
    HIP_ASSERT(hipFree(d_in));
    HIP_ASSERT(hipFree(d_out));

    // release host memory
    free(h_in);
    free(h_out);

    return 0;
}