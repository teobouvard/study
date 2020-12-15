#include "wb.hpp"
#include <algorithm>
#include <stdlib.h>

constexpr int BLOCK_SIZE = 128;

__global__ void sum_reduce(float *vec, float *sum) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // each block partially sum in shared array
  __shared__ float partialSum[BLOCK_SIZE];
  // each thread loads an element of the input vector
  partialSum[threadIdx.x] = vec[idx];

  // using tree reduction with sequential adressing
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      partialSum[threadIdx.x] += partialSum[threadIdx.x + stride];
    }
  }

  // each block contributes to the partial sum with its reduced sum
  if (threadIdx.x == 0) {
    sum[blockIdx.x] = partialSum[0];
  }
}

int main(int argc, char **argv) {
  // read arguments
  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  int numElements;
  float *hostVec = wbImport(wbArg_getInputFile(args, 0), &numElements);
  float hostPartialSum[BLOCK_SIZE] = {};
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "Number of elements : ", numElements);

  wbTime_start(GPU, "Allocating GPU memory");
  float *deviceVec;
  cudaMalloc(&deviceVec, numElements * sizeof(float));
  float *devicePartialSum;
  cudaMalloc(&devicePartialSum, BLOCK_SIZE * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory");

  wbTime_start(GPU, "Copying input memory to the GPU");
  cudaMemcpy(deviceVec, hostVec, numElements * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(devicePartialSum, 0, BLOCK_SIZE * sizeof(float));
  wbTime_stop(GPU, "Copying input memory to the GPU");

  // initialize grid and block dimensions
  int block = BLOCK_SIZE;
  int grid = (numElements + block - 1) / block;

  wbTime_start(Compute, "Performing CUDA computation");
  sum_reduce<<<grid, block>>>(deviceVec, devicePartialSum);
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostPartialSum, devicePartialSum, BLOCK_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(devicePartialSum);
  cudaFree(deviceVec);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // final reduction
  float result = 0.0;
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    result += hostPartialSum[i];
  }

  // check matrix product solution
  float cpuResult = 0.0;
  for (int i = 0; i < numElements; ++i) {
    cpuResult += hostVec[i];
  }

  if (!wbInternal::wbFPCloseEnough(result, cpuResult)) {
    fprintf(stderr, "Expecting %f but got %f\n", cpuResult, result);
  } else {
    fprintf(stderr, "Solution is correct (sum=%f)\n", result);
  }

  // free host memory
  free(hostVec);

  return EXIT_SUCCESS;
}
