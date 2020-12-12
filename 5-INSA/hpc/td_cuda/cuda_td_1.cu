#include "wb.hpp"
#include <stdlib.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("blockDim=%d, blockIdx=%d, threadIdx=%d\n", blockDim.x, blockIdx.x,
  //       threadIdx.x);
  if (i < len) {
    out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv) {
  // read arguments
  wbArg_t args = wbArg_read(argc, argv);

  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  // read input data
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
  wbLog(TRACE, "The input length is ", inputLength);

  // allocate GPU memory
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc(&deviceInput1, inputLength * sizeof(float));
  cudaMalloc(&deviceInput2, inputLength * sizeof(float));
  cudaMalloc(&deviceOutput, inputLength * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  // copy memory to the GPU
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // initialize grid and block dimensions
  int numBlocks = 1;
  int threadsPerBlock = inputLength;

  // launch GPU kernel
  wbTime_start(Compute, "Performing CUDA computation");
  vecAdd<<<numBlocks, threadsPerBlock>>>(deviceInput1, deviceInput2,
                                         deviceOutput, inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  // copy GPU memory back to the CPU
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // free GPU memory
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // check solution
  wbSolution(args, hostOutput, inputLength);

  // free host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return EXIT_SUCCESS;
}
