#include "BackpropFunctions.cuh"

__global__ void actualizarValoresMatrizMomentoAdam(const float* grad, float* mdst, float b1, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //if (mdst[idx * ncols + idy]!= 0) { printf("\nmomento distinto de 0"); }
        mdst[idx * ncols + idy] = b1 * mdst[idx * ncols + idy] + (1 - b1) * grad[idx * ncols + idy];
        //if (isnan(mdst[idx * ncols + idy]) || isinf(mdst[idx * ncols + idy])) { printf("\ncaca..."); }
    }
}

//con la matriz gradiente grad se aplica velocidad en la matriz destino mdst 
__global__ void actualizarValoresMatrizVelocidadAdam(const float* grad, float* mdst, float b2, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //if (mdst[idx * ncols + idy] != 0) { printf("\nvelocidad distinto de 0"); }
        //if (mdst[idx * ncols + idy] < 0) { mdst[idx * ncols + idy] = -mdst[idx * ncols + idy]; }
        mdst[idx * ncols + idy] = b2 * mdst[idx * ncols + idy] + (1 - b2) * (grad[idx * ncols + idy] * grad[idx * ncols + idy]);
        //if (isnan(mdst[idx * ncols + idy]) || isinf(mdst[idx * ncols + idy])) { printf("\ncaca..."); }
    }
}

//se aplica el aprendizaje adam en la matriz destino mdest 
__global__ void calcularVectorGradienteAdam(float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        float t1 = tapren * (mom[idx * ncols + idy] / (1 - b1));
        float t2 = epsilon + sqrtf(vel[idx * ncols + idy] / (float)(1 - b2));
        mdst[idx * ncols + idy] = -t1 / t2;
    }
}

void managedActualizarValoresMatrizMomentoAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b1, int nrows, int ncols) {
    int sqrt_dim = (int) sqrt(max_threads_block);
    dim3 grid(sqrt_dim, sqrt_dim);
    dim3 block((int)ceil(nrows/ (float)sqrt_dim), (int)ceil(ncols / (float)sqrt_dim));
    actualizarValoresMatrizMomentoAdam << < grid, block, 0, stream >> > (grad, mdst, b1, nrows, ncols);
}

void managedActualizarValoresMatrizVelocidadAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b2, int nrows, int ncols) {
    int sqrt_dim = (int)sqrt(max_threads_block);
    dim3 grid(sqrt_dim, sqrt_dim);
    dim3 block((int)ceil(nrows / (float)sqrt_dim), (int)ceil(ncols / (float)sqrt_dim));
    actualizarValoresMatrizVelocidadAdam << < grid, block, 0, stream >> > (grad, mdst, b2, nrows, ncols);
}

void managedCalcularVectorGradienteAdam(cudaStream_t stream, int max_threads_block, float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols) {
    int sqrt_dim = (int)sqrt(max_threads_block);
    dim3 grid(sqrt_dim, sqrt_dim);
    dim3 block((int)ceil(nrows / (float)sqrt_dim), (int)ceil(ncols / (float)sqrt_dim));
    calcularVectorGradienteAdam << < grid, block, 0, stream >> > (tapren, b1, b2, epsilon, mdst, mom, vel, nrows, ncols);
}