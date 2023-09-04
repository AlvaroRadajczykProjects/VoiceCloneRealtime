#include "ActivationLossFunctions.cuh"

__device__ float kernel_watelu(float x) {
    return x * 666;
}
extern __device__ func_t d_watelu = kernel_watelu;

/* ============================== ACTIVATION FUNCTIONS AND THEIR DERIVATIVES ============================== */

//ELU
__device__ float kernel_ELU(float x) {
    if (x < 0) { return 0.01 * (expf(x) - 1); }
    else { return x; }
}
extern __device__ func_t d_ELU = kernel_ELU;

__device__ float kernel_dELU(float x) {
    if (x < 0) { return 0.01 * ((x * 100) + 1); }
    else { return 1; }
}
extern __device__ func_t d_dELU = kernel_dELU;

//Linear
__device__ float kernel_Linear(float x) {
    return x;
}
extern __device__ func_t d_Linear = kernel_Linear;

__device__ float kernel_dLinear(float x) {
    return 1;
}
extern __device__ func_t d_dLinear = kernel_dLinear;

/* ================================= LOSS FUNCTIONS AND THEIR DERIVATIVES ================================= */

//MSE
__device__ float kernel_MSE(float predy, float realy) {
    return powf(predy - realy, 2.0);
}
extern __device__ func2_t d_MSE = kernel_MSE;

__device__ float kernel_dMSE(float predy, float realy) {
    return 2 * (predy - realy);
}
extern __device__ func2_t d_dMSE = kernel_dMSE;

/* ======================================================================================================== */