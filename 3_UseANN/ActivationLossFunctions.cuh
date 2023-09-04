#include "CUDAKernels.cuh"

__device__ float kernel_watelu(float x);

/* ============================== ACTIVATION FUNCTIONS AND THEIR DERIVATIVES ============================== */

//ELU
__device__ float kernel_ELU(float x);
__device__ float kernel_dELU(float x);

//Linear
__device__ float kernel_Linear(float x);
__device__ float kernel_dLinear(float x);

/* ================================= LOSS FUNCTIONS AND THEIR DERIVATIVES ================================= */

//MSE
__device__ float kernel_MSE(float predy, float realy);
__device__ float kernel_dMSE(float predy, float realy);

/* ======================================================================================================== */