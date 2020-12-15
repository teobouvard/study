#include "wb.hpp"
#include <stdlib.h>

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < numCRows * numCColumns) {
    int row = idx / numCColumns;
    int column = idx % numCColumns;
    float result = 0.0;
    for (int i = 0; i < numAColumns; ++i) {
      result += A[row * numAColumns + i] * B[i * numBColumns + column];
      // TODO why do both lines compute the same result ?
    }
    C[row * numCColumns + column] = result;
  }
}

int main(int argc, char **argv) {
  // read arguments
  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  // allocating input matrices
  int numARows;
  int numAColumns;
  int numBRows;
  int numBColumns;
  float *hostA =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  float *hostB =
      (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

  // allocating output matrix
  int numCRows = numARows;
  int numCColumns = numBColumns;
  float *hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory");
  float *deviceA;
  float *deviceB;
  float *deviceC;
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory");

  wbTime_start(GPU, "Copying input memory to the GPU");
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU");

  // initialize grid and block dimensions
  int threadsPerBlock = 128;
  int numElements = numCRows * numCColumns;
  int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  wbLog(TRACE, "Using ", numBlocks, " blocks");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply<<<numBlocks, threadsPerBlock>>>(
      deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns,
      numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // check matrix product solution
  wbSolution(args, hostC, numCRows, numCColumns);

  // free host memory
  free(hostA);
  free(hostB);
  free(hostC);

  return EXIT_SUCCESS;
}
