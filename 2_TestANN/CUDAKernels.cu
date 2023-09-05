#include "CUDAKernels.cuh"

using namespace std;

void edu_shuffle(int arr[], int n) {
    unsigned semilla = rand() % 10000;
    shuffle(arr, arr + n, default_random_engine(semilla));
}

void manageCUDAError(cudaError_t status, char* description) {
    if (status != cudaSuccess) {
        fprintf(stderr, "\n\nError de CUDA %s: %s\n\n", description, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

unsigned long long nextMultiple(unsigned long long val, int mod) {
    if (val == 0) { return mod; }
    if (val % mod == 0) { return val; }
    return val + (mod - (val % mod));
}

unsigned long long nextFourMultiple(unsigned long long val) {
    if (val == 0) { return 4; }
    if (val % 4 == 0) { return val; }
    return val + (4 - (val % 4));
}

const void matrizTraspuestaDevice(cublasHandle_t handle, float* odata, float* idata, int m, int n) {
    cublasSgeam_64(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, idata, n, &beta_nosum, idata, m, odata, m);
}

const void productoMatricesDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_nosum, c, n);
}

const void productoMatricesTrasposedBDevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, a, k, &beta_nosum, c, n);
}

const void productoMatricesTrasposedADevice(cublasHandle_t handle, const float* a, const float* b, float* c, int m, int k, int n) {
    cublasSgemm_v2_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, n, a, m, &beta_nosum, c, n);
}

const void productoMatricesBatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_nosum, c, n, num_matr);
}

const void productoMatricesTrasposedBBatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k, a, k, &beta_nosum, c, n, num_matr);
}

const void productoMatricesTrasposedABatchDevice(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b, n, a, m, &beta_nosum, c, n, num_matr);
}

const void productoMatricesBatchDeviceSumC(cublasHandle_t handle, float** a, float** b, float** c, int m, int k, int n, int num_matr) {
    cublasSgemmBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta_sum, c, n, num_matr);
}

__global__ void applyFunctionVectorial(float* arr, func_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = reinterpret_cast<float4*>(arr)[idx];
    val.x = func(val.x);
    val.y = func(val.y);
    val.z = func(val.z);
    val.w = func(val.w);
    reinterpret_cast<float4*>(arr)[idx] = val;
}

__global__ void applyFunctionScalar(float* arr, func_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = func(arr[idx]);
}

__global__ void applyLossFunctionVectorial(float* pred, float* real, float* dst, func2_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 vpred = reinterpret_cast<float4*>(pred)[idx];
    float4 vreal = reinterpret_cast<float4*>(real)[idx];
    vpred.x = func(vpred.x, vreal.x);
    vpred.y = func(vpred.y, vreal.y);
    vpred.z = func(vpred.z, vreal.z);
    vpred.w = func(vpred.w, vreal.w);
    reinterpret_cast<float4*>(dst)[idx] = vpred;
}

__global__ void applyLossFunctionScalar(float* pred, float* real, float* dst, func2_t func) {
    //https://forums.developer.nvidia.com/t/the-float-and-float4-types-in-cuda/65061
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = func(pred[idx], real[idx]);
}

__global__ void multiplyAllElementsByConstantVectorial(float* arr, float ct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = reinterpret_cast<float4*>(arr)[idx];
    val.x = val.x * ct;
    val.y = val.y * ct;
    val.z = val.z * ct;
    val.w = val.w * ct;
    reinterpret_cast<float4*>(arr)[idx] = val;
}

__global__ void multiplyAllElementsByConstantScalar(float* arr, float ct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = arr[idx] * ct;
}

__global__ void sumVectorsSameDimensionsVectorial(float* dst, float* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val_src = reinterpret_cast<float4*>(src)[idx];
    float4 val_dst = reinterpret_cast<float4*>(dst)[idx];
    val_src.x = val_src.x + val_dst.x;
    val_src.y = val_src.y + val_dst.y;
    val_src.z = val_src.z + val_dst.z;
    val_src.w = val_src.w + val_dst.w;
    reinterpret_cast<float4*>(dst)[idx] = val_src;
}

__global__ void sumVectorsSameDimensionsScalar(float* dst, float* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = dst[idx] + src[idx];
}

__global__ void multiplyMatricesSameDimensionsVectorial(float* dst, float* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val_src = reinterpret_cast<float4*>(src)[idx];
    float4 val_dst = reinterpret_cast<float4*>(dst)[idx];
    val_src.x = val_src.x * val_dst.x;
    val_src.y = val_src.y * val_dst.y;
    val_src.z = val_src.z * val_dst.z;
    val_src.w = val_src.w * val_dst.w;
    reinterpret_cast<float4*>(dst)[idx] = val_src;
}

__global__ void multiplyMatricesSameDimensionsScalar(float* dst, float* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = dst[idx] + src[idx];
}

__global__ void sumStdDev(float* matrix, float* mean_vector, float* var_vector, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float vals[32][32];
    if (row < nrows && col < ncols) {
        vals[threadIdx.x][threadIdx.y] = powf(matrix[row * ncols + col]-mean_vector[row], 2.0f);
    } else {
        vals[threadIdx.x][threadIdx.y] = 0;
    }
    __syncthreads();

    if ((threadIdx.y + 1) % 2 == 0) { vals[threadIdx.x][threadIdx.y] += vals[threadIdx.x][threadIdx.y - 1]; }
    __syncthreads();
    if ((threadIdx.y + 1) % 4 == 0) { vals[threadIdx.x][threadIdx.y] += vals[threadIdx.x][threadIdx.y - 2]; }
    __syncthreads();
    if ((threadIdx.y + 1) % 8 == 0) { vals[threadIdx.x][threadIdx.y] += vals[threadIdx.x][threadIdx.y - 4]; }
    __syncthreads();
    if ((threadIdx.y + 1) % 16 == 0) { vals[threadIdx.x][threadIdx.y] += vals[threadIdx.x][threadIdx.y - 8]; }
    __syncthreads();
    if ((threadIdx.y + 1) % 32 == 0) { vals[threadIdx.x][threadIdx.y] += vals[threadIdx.x][threadIdx.y - 16]; }
    __syncthreads();

    //if ((threadIdx.y + 1) % 32 == 0 && row < nrows) { printf("\n:)");  stddev_vector[row] = 3; }
    if ((threadIdx.y + 1) % 32 == 0 && row < nrows) { atomicAdd(&var_vector[row], vals[threadIdx.x][threadIdx.y]); }
}

__global__ void applyLayerNormalization(float* matrix_forward, float* mean_vector, float* var_vector, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nrows && col < ncols) {
        matrix_forward[row * ncols + col] = (matrix_forward[row * ncols + col] - mean_vector[row]) / sqrtf(var_vector[row]);
    }
}

const void managedApplyFunction(cudaStream_t stream, int max_num_threads, int num_elems, float* arr, func_t func) {
    int nblocks = (int)(num_elems / (4 * max_num_threads));
    applyFunctionVectorial <<< nblocks, max_num_threads, 0, stream >> > (arr, func);
    num_elems -= (nblocks * 4 * max_num_threads);
    int offset = (nblocks * 4 * max_num_threads);
    applyFunctionVectorial << < 1, (int) (num_elems/4), 0, stream >> > (arr + offset, func);
    offset += ((num_elems / 4) * 4);
    num_elems -= (num_elems / 4) * 4;
    applyFunctionScalar << < 1, num_elems%4, 0, stream >> > (arr + offset, func);
}

const void managedApplyLossFunction(cudaStream_t stream, int max_num_threads, int num_elems, float* pred, float* real, float* dst, func2_t func) {
    int nblocks = (int)(num_elems / (4 * max_num_threads));
    applyLossFunctionVectorial << < nblocks, max_num_threads, 0, stream >> > (pred, real, dst, func);
    num_elems -= (nblocks * 4 * max_num_threads);
    int offset = (nblocks * 4 * max_num_threads);
    applyLossFunctionVectorial << < 1, (int)(num_elems / 4), 0, stream >> > (pred+ offset, real+ offset, dst+ offset, func);
    offset += ((num_elems / 4) * 4);
    num_elems -= (num_elems / 4) * 4;
    applyLossFunctionScalar << < 1, num_elems % 4, 0, stream >> > (pred + offset, real + offset, dst + offset, func);
}

const void managedMultiplyAllElementsByConstant(cudaStream_t stream, int max_num_threads, int num_elems, float* arr, float ct) {
    int nblocks = (int)(num_elems / (4 * max_num_threads));
    multiplyAllElementsByConstantVectorial << < nblocks, max_num_threads, 0, stream >> > (arr, ct);
    num_elems -= (nblocks * 4 * max_num_threads);
    int offset = (nblocks * 4 * max_num_threads);
    multiplyAllElementsByConstantVectorial << < 1, (int)(num_elems / 4), 0, stream >> > (arr + offset, ct);
    offset += ((num_elems / 4) * 4);
    num_elems -= (num_elems / 4) * 4;
    multiplyAllElementsByConstantScalar << < 1, num_elems % 4, 0, stream >> > (arr + offset, ct);
}

const void managedSumVectorsSameDimensions(cudaStream_t stream, int max_num_threads, int num_elems, float* dst, float* src) {
    int nblocks = (int)(num_elems / (4 * max_num_threads));
    sumVectorsSameDimensionsVectorial << < nblocks, max_num_threads, 0, stream >> > (dst, src);
    num_elems -= (nblocks * 4 * max_num_threads);
    int offset = (nblocks * 4 * max_num_threads);
    sumVectorsSameDimensionsVectorial << < 1, (int)(num_elems / 4), 0, stream >> > (dst + offset, src + offset);
    offset += ((num_elems / 4) * 4);
    num_elems -= (num_elems / 4) * 4;
    sumVectorsSameDimensionsScalar << < 1, num_elems % 4, 0, stream >> > (dst + offset, src + offset);
}

const void managedMultiplyMatricesSameDimensions(cudaStream_t stream, int max_num_threads, int num_elems, float* dst, float* src) {
    int nblocks = (int)(num_elems / (4 * max_num_threads));
    multiplyMatricesSameDimensionsVectorial << < nblocks, max_num_threads, 0, stream >> > (dst, src);
    num_elems -= (nblocks * 4 * max_num_threads);
    int offset = (nblocks * 4 * max_num_threads);
    multiplyMatricesSameDimensionsVectorial << < 1, (int)(num_elems / 4), 0, stream >> > (dst + offset, src + offset);
    offset += ((num_elems / 4) * 4);
    num_elems -= (num_elems / 4) * 4;
    multiplyMatricesSameDimensionsScalar << < 1, num_elems % 4, 0, stream >> > (dst + offset, src + offset);
}

const void managedSumStdDev(cudaStream_t stream, int max_num_threads, float* matrix, float* mean_vector, float* stddev_vector, int nrows, int ncols) {
    int sqrt_dim = (int)sqrt(max_num_threads);
    dim3 grid((int)ceil(nrows / (float)sqrt_dim), (int)ceil(ncols / (float)sqrt_dim));
    dim3 block(sqrt_dim, sqrt_dim);
    sumStdDev << < grid, block, 0, stream >> > (matrix, mean_vector, stddev_vector, nrows, ncols);
}

const void managedApplyLayerNormalization(cudaStream_t stream, int max_num_threads, float* matrix_forward, float* mean_vector, float* var_vector, int nrows, int ncols) {
    int sqrt_dim = (int)sqrt(max_num_threads);
    dim3 grid((int)ceil(nrows / (float)sqrt_dim), (int)ceil(ncols / (float)sqrt_dim));
    dim3 block(sqrt_dim, sqrt_dim);
    applyLayerNormalization << < grid, block, 0, stream >> > (matrix_forward, mean_vector, var_vector, nrows, ncols);
}