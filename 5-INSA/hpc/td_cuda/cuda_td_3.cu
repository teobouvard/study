#include "wb.hpp"
#include <cmath>
#include <stdlib.h>

void gaussian_kernel_fill(float sigma, float K, int kernelSize, float *kernel) {
  // 1 hour of undefined behaviour debugging later : without initializing sum to
  // 0, it starts with the value which was at this address
  float sum = 0.0;
  for (int i = 0; i < kernelSize; i++) {
    for (int j = 0; j < kernelSize; j++) {
      float x = i - (kernelSize - 1) / 2.0;
      float y = j - (kernelSize - 1) / 2.0;
      kernel[i * kernelSize + j] =
          K * std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
      sum += kernel[i * kernelSize + j];
    }
  }
  for (int i = 0; i < kernelSize * kernelSize; i++) {
    kernel[i] /= sum;
  }
  // print kernel for debugging purposes
  // for (int i = 0; i < kernelSize; i++) {
  //  for (int j = 0; j < kernelSize; j++) {
  //    printf("%f ", kernel[i * kernelSize + j]);
  //  }
  //  printf("\n");
  //}
}

__global__ void blur(float *inputImage, float *outputImage, int imageWidth,
                     int numElements, float *gaussianKernel, int kernelSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    float blurredElement = 0.0;
    float kernelElement;
    float imageElement;
    int imageIndex;
    /// TODO handle even kernel sizes
    int halfKernelSize = (kernelSize - 1) / 2.0;
    for (int i = 0; i < kernelSize; ++i) {
      for (int j = 0; j < kernelSize; ++j) {
        kernelElement = gaussianKernel[i * kernelSize + j];
        imageIndex = idx + 3 * (i - halfKernelSize) * imageWidth +
                     3 * (j - halfKernelSize);
        if (imageIndex > 0 && imageIndex < numElements) {
          imageElement = inputImage[imageIndex];
        } else {
          imageElement = 0.0;
        }
        blurredElement += kernelElement * imageElement;
      }
    }

    outputImage[idx] = blurredElement;
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

  // create gaussian kernel for filtering
  int kernelSize = 50;
  float sigma = 10;
  float K = 1.0;
  float *hostGaussianKernel =
      (float *)malloc(kernelSize * kernelSize * sizeof(float));
  if (hostGaussianKernel == NULL) {
    fprintf(stderr, "Could not allocate gaussian kernel memory");
    return EXIT_FAILURE;
  }
  gaussian_kernel_fill(sigma, K, kernelSize, hostGaussianKernel);
  printf("kernel_element=%f\n", hostGaussianKernel[1250]);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceGaussianKernel;
  cudaMalloc(&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceGaussianKernel, kernelSize * kernelSize * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceGaussianKernel, hostGaussianKernel,
             kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  // initialize grid and block dimensions
  // threadsPerBlock should be a multiple of the number of threads per warp (32)
  // to maximize efficiency and should be less than the maximum number of
  // threads per block which can be retrieved with the deviceQuery script (1024)
  // still not sure of how high it should be to be the most efficient?
  int threadsPerBlock = 256;
  int numElements = imageWidth * imageHeight * imageChannels;
  int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  wbLog(TRACE, "Using ", numBlocks, " blocks");

  wbTime_start(Compute, "Doing the computation on the GPU");
  blur<<<numBlocks, threadsPerBlock>>>(
      deviceInputImageData, deviceOutputImageData, imageWidth, numElements,
      deviceGaussianKernel, kernelSize);
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
  free(hostGaussianKernel);

  return EXIT_SUCCESS;
}
