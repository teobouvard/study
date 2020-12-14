#include "wb.hpp"
#include <stdlib.h>
#include <sys/types.h>

constexpr int NUM_BINS = 128;

__global__ void char_histogram(int *vec, int size, int *histogram) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx;
  int stride = blockDim.x * gridDim.x;

  // initialize shared histogram
  __shared__ int shared_histogram[NUM_BINS];

  // make sure all threads have an initial value of 0
  if (threadIdx.x < NUM_BINS) {
    shared_histogram[threadIdx.x] = 0;
  }
  __syncthreads();

  // count occurences in shared histogram
  while (i < size) {
    atomicAdd(&(shared_histogram[vec[i]]), 1);
    i += stride;
  }

  __syncthreads();
  // combine results on DRAM histogram
  if (threadIdx.x < NUM_BINS) {
    atomicAdd(&(histogram[threadIdx.x]), shared_histogram[threadIdx.x]);
  }
}

int main(int argc, char **argv) {
  // read arguments
  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  int numElements;
  int *hostVec = wbImport(wbArg_getInputFile(args, 0), &numElements, "integer");
  int hostHistogram[NUM_BINS] = {};
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "Number of elements : ", numElements);

  wbTime_start(GPU, "Allocating GPU memory");
  int *deviceVec;
  cudaMalloc(&deviceVec, numElements * sizeof(int));
  int *deviceHistogram;
  cudaMalloc(&deviceHistogram, NUM_BINS * sizeof(int));
  wbTime_stop(GPU, "Allocating GPU memory");

  wbTime_start(GPU, "Copying input memory to the GPU");
  cudaMemcpy(deviceVec, hostVec, numElements * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemset(deviceHistogram, 0, NUM_BINS * sizeof(int));
  wbTime_stop(GPU, "Copying input memory to the GPU");

  // initialize grid and block dimensions
  int block = 256;
  int grid = (numElements + block - 1) / block;

  wbTime_start(Compute, "Performing CUDA computation");
  char_histogram<<<grid, block>>>(deviceVec, numElements, deviceHistogram);
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostHistogram, deviceHistogram, NUM_BINS * sizeof(int),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceHistogram);
  cudaFree(deviceVec);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // check matrix product solution
  wbSolutionEqual(args, hostHistogram, NUM_BINS);

  // free host memory
  free(hostVec);

  return EXIT_SUCCESS;
}
