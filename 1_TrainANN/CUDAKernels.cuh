#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <cassert>

#ifndef CUDA_KERNELS
#define CUDA_KERNELS

using func_lrate = float(*) (int);
using func_t = float(*) (float);
using func2_t = float(*) (float, float);
using func3_t = float(*) (float, float, float);

const float alpha = 1.0f; //aparte del producto entre los elementos, puedes multiplicar esto
const float beta_sum = 1.0f; //aparte del producto final, puedes sumar esto
const float beta_nosum = 0.0f; //aparte del producto final, puedes sumar esto

void edu_shuffle(int arr[], int n);

void manageCUDAError(cudaError_t status, char* description);

unsigned long long nextMultiple(unsigned long long val, int mod);

unsigned long long nextFourMultiple(unsigned long long val);

const void matrizTraspuestaDevice(cublasHandle_t handle, float* odata, float* idata, int m, int n);

const void productoMatricesDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n);

const void productoMatricesTrasposedBDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n);

const void productoMatricesTrasposedADevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n);

const void productoMatricesBatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr);

const void productoMatricesTrasposedBBatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr);

const void productoMatricesTrasposedABatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr);

const void productoMatricesBatchDeviceSumC(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr);

__global__ void applyFunctionVectorial(float* arr, func_t func);

__global__ void applyFunctionScalar(float* arr, func_t func);

__global__ void applyLossFunctionVectorial(float* pred, float* real, float* dst, func2_t func);

__global__ void applyLossFunctionScalar(float* pred, float* real, float* dst, func2_t func);

__global__ void multiplyAllElementsByConstantVectorial(float* arr, float ct);

__global__ void multiplyAllElementsByConstantScalar(float* arr, float ct);

__global__ void sumVectorsSameDimensionsVectorial(float* dst, float* src);

__global__ void sumVectorsSameDimensionsScalar(float* dst, float* src);

__global__ void multiplyMatricesSameDimensionsVectorial(float* dst, float* src);

__global__ void multiplyMatricesSameDimensionsScalar(float* dst, float* src);

__global__ void sumStdDev(float* matrix, float* mean_vector, float* var_vector, int nrows, int ncols);

__global__ void applyLayerNormalization(float* matrix_forward, float* mean_vector, float* var_vector, int nrows, int ncols);

const void managedApplyFunction(cudaStream_t stream, int max_num_threads, int num_elems, float* arr, func_t func);

const void managedApplyLossFunction(cudaStream_t stream, int max_num_threads, int num_elems, float* pred, float* real, float* dst, func2_t func);

const void managedMultiplyAllElementsByConstant(cudaStream_t stream, int max_num_threads, int num_elems, float* arr, float ct);

const void managedSumVectorsSameDimensions(cudaStream_t stream, int max_num_threads, int num_elems, float* dst, float* src);

const void managedMultiplyMatricesSameDimensions(cudaStream_t stream, int max_num_threads, int num_elems, float* dst, float* src);

const void managedSumStdDev(cudaStream_t stream, int max_num_threads, float* matrix, float* mean_vector, float* var_vector, int nrows, int ncols);

const void managedApplyLayerNormalization(cudaStream_t stream, int max_num_threads, float* matrix_forward, float* mean_vector, float* var_vector, int nrows, int ncols);

#endif