#include "wb.hpp"
#include <stdlib.h>

__global__ void grayscale(float *inputImage, float *outputImage,
                          int imageSize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < imageSize) {
    float r = inputImage[3 * i];
    float g = inputImage[3 * i + 1];
    float b = inputImage[3 * i + 2];
    outputImage[i] = 0.21 * r + 0.71 * g + 0.07 * b;
  }
}

int main(int argc, char *argv[]) {
  // parse input arguments
  wbArg_t args = wbArg_read(argc, argv);

  // read input image
  char *inputImageFile = wbArg_getInputFile(args, 0);
  wbImage_t inputImage = wbImport(inputImageFile);
  int imageWidth = wbImage_getWidth(inputImage);
  int imageHeight = wbImage_getHeight(inputImage);
  int imageChannels = wbImage_getChannels(inputImage);

  // create output image
  // monochromatic so it only contains a single channel
  wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // retreive image data to perform computations
  float *hostInputImageData = wbImage_getData(inputImage);
  float *hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  float *deviceInputImageData;
  float *deviceOutputImageData;
  cudaMalloc(&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  // initialize grid and block dimensions
  // threadsPerBlock should be a multiple of the number of threads per warp (32)
  // to maximize efficiency and should be less than the maximum number of
  // threads per block which can be retrieved with the deviceQuery script (1024)
  // still not sure of how high it should be to be the most efficient?
  // trial&error + occupancy calculator
  int threadsPerBlock = 256;
  int numPixels = imageWidth * imageHeight;
  int numBlocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
  wbLog(TRACE, "Using ", numBlocks, " blocks");

  wbTime_start(Compute, "Doing the computation on the GPU");
  grayscale<<<numBlocks, threadsPerBlock>>>(deviceInputImageData,
                                            deviceOutputImageData, numPixels);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // scale image from (0.0, 1.0) to (0, 255)
  unsigned char grayScale[imageHeight][imageWidth];
  for (int j = 0; j < imageHeight; ++j) {
    for (int i = 0; i < imageWidth; ++i) {
      grayScale[j][i] = ceil(hostOutputImageData[i + j * imageWidth] * 255.0);
    }
  }

  // write output image to PPM file
  char *outputImageFile = wbArg_getInputFile(args, 1);
  FILE *fp = fopen(outputImageFile, "wb"); /* b - binary mode */
  fprintf(fp, "P5\n%d %d\n255\n", imageWidth, imageHeight);
  fwrite(grayScale, sizeof(grayScale), 1, fp);
  fclose(fp);

  // free GPU memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  // free host memory
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return EXIT_SUCCESS;
}
