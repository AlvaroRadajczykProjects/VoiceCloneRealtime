#include "CUDAKernels.cuh"

__global__ void actualizarValoresMatrizMomentoAdam(const float* grad, float* mdst, float b1, int nrows, int ncols);

__global__ void actualizarValoresMatrizVelocidadAdam(const float* grad, float* mdst, float b2, int nrows, int ncols);

__global__ void calcularVectorGradienteAdam(float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols);

void managedActualizarValoresMatrizMomentoAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b1, int nrows, int ncols);

void managedActualizarValoresMatrizVelocidadAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b2, int nrows, int ncols);

void managedCalcularVectorGradienteAdam(cudaStream_t stream, int max_threads_block, float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols);