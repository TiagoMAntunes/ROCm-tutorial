#include "hip/hip_runtime.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <chrono>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define THREADS_PER_BLOCK 256


__global__ void vector_add_float(float * a, float * b, float * c, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    float * hostA, * hostB, * hostC;
    float * deviceA, * deviceB, * deviceC;

    unsigned long int NUM = 268435456;
    size_t size = NUM * sizeof(float);
    printf("Allocating %lu elements, total size %lu bytes\n", NUM, size);
    // initialize local data
    hostA = (float *) malloc(size);
    hostB = (float *) malloc(size);
    hostC = (float *) malloc(size);

    for (int i = 0; i < NUM; i++) {
        hostA[i] = i;
        hostB[i] = i;        
    }

    // initialize device data
    
    HIP_ASSERT(hipMalloc(&deviceA, size));
    HIP_ASSERT(hipMalloc(&deviceB, size));
    HIP_ASSERT(hipMalloc(&deviceC, size));

    // move data into device
    HIP_ASSERT(hipMemcpy(deviceA, hostA, size, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceB, hostB, size, hipMemcpyHostToDevice));

    // launch function
    hipLaunchKernelGGL(vector_add_float, 
                dim3(NUM/THREADS_PER_BLOCK + 1), // n blocks
                dim3(THREADS_PER_BLOCK),     // n threads per block
                0, 0,
                deviceA, deviceB, deviceC, NUM
    );

    hipDeviceSynchronize();

    // copy data back
    HIP_ASSERT(hipMemcpy(hostC, deviceC, size, hipMemcpyDeviceToHost));

    // print data
    for (int i = 0; i < NUM; i++)
        if (hostC[i] - i*2 > 1e-5)
            printf("Error at %d with value %f\n", i, hostC[i]);

    HIP_ASSERT(hipFree(deviceA));
    HIP_ASSERT(hipFree(deviceB));
    HIP_ASSERT(hipFree(deviceC));

    free(hostA);
    free(hostB);
    free(hostC);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    printf("Ran in %lu ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

}