#include "wb.hpp"
#include <stdlib.h>

constexpr int TILE_WIDTH = 16;

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  // tiles of A and B in shared memory
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  // retrieve element of C to be computed by current thread
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  // C[i, j] to be computed by current thread
  float cij = 0;

  for (int step = 0; step < numCColumns; ++step) {
    // each thread loads an element of A and an element of B in shared memory
    tileA[ty][tx] =
        A[row * numAColumns + step * TILE_WIDTH + tx]; // not sure of col/row
    tileB[ty][tx] =
        B[(step * TILE_WIDTH + ty) * numBRows + col]; // not sure of col/row

    // make sure all elements of tile are loaded
    __syncthreads();

    // compute partial dot product
    for (int i = 0; i < TILE_WIDTH; ++i) {
      cij += tileA[ty][i] * tileB[i][tx];
    }

    // make sure all elements of tile were used to compute cij
    __syncthreads();
  }

  C[row * numCColumns + col] = cij;
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
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid(numCColumns / block.x, numCRows / block.y);

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiplyShared<<<grid, block>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows, numBColumns,
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
