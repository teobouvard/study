#include "wb.hpp"
#include <stdlib.h>

#define BLUR_SIZE 10

__global__ void blur(float *inputImage, float *outputImage, int imageSize) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < imageSize) {
    float blurredPixel = 0.0;
    for (int s = -BLUR_SIZE; s < BLUR_SIZE; ++s) {
      blurredPixel += inputImage[i + 3 * s];
    }
    blurredPixel /= 2 * BLUR_SIZE;

    outputImage[i] = blurredPixel;
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
  wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  // retreive image data to perform computations
  float *hostInputImageData = wbImage_getData(inputImage);
  float *hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  float *deviceInputImageData;
  float *deviceOutputImageData;
  cudaMalloc(&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
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
  int threadsPerBlock = 256;
  int numPixels = imageWidth * imageHeight * imageChannels;
  int numBlocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
  wbLog(TRACE, "Using ", numBlocks, " blocks");

  wbTime_start(Compute, "Doing the computation on the GPU");
  blur<<<numBlocks, threadsPerBlock>>>(deviceInputImageData,
                                       deviceOutputImageData, numPixels);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // write output image to PPM file
  char *outputImageFile = wbArg_getInputFile(args, 1);
  FILE *fp = fopen(outputImageFile, "wb"); /* b - binary mode */
  fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
  for (int i = 0; i < imageHeight; ++i) {
    for (int j = 0; j < imageWidth; ++j) {
      unsigned char color[3];
      color[0] = hostOutputImageData[(i * imageWidth + j) * 3] * 255; /* red */
      color[1] =
          hostOutputImageData[(i * imageWidth + j) * 3 + 1] * 255; /* green */
      color[2] =
          hostOutputImageData[(i * imageWidth + j) * 3 + 2] * 255; /* blue */

      fwrite(color, 1, 3, fp);
    }
  }
  fclose(fp);

  // free GPU memory
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  // free host memory
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return EXIT_SUCCESS;
}
