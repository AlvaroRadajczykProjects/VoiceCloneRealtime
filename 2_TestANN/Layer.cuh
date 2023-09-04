#include "CUDAKernels.cuh"
#include "BackpropFunctions.cuh"
#include "PrintUtils.h"

#ifndef LAYER
#define LAYER

class Layer {
    private:
        int max_num_threads;
        int num_blocks_needed_apply_function;
        int num_threads_needed_apply_function;

        int input_size = 0;
        int size = 0;
        int number_networks = 0;
        int number_input_examples = 0;
        bool is_training = false;
        bool is_first_layer = false;

        cublasHandle_t* handle;

        func_t activation_function = NULL;
        func_t activation_derivative_function = NULL;
        //func2_t each_output_sum = NULL;
        //func2_t activation_function2 = NULL;
        //func2_t activation_derivative_function2 = NULL;

        float* d_auxiliar_expand_reduce_matrix;
        float** hd_expand_reduce_matrix_pointers;
        float** d_expand_reduce_matrix_pointers;

        float* d_array_weight_matrix = NULL;
        float* d_array_bias_vector = NULL;
        float** hd_weight_matrices_pointers = NULL;
        float** hd_bias_vectors_pointers = NULL;
        float** d_weight_matrices_pointers = NULL;
        float** d_bias_vectors_pointers = NULL;

        //sólo hace falta al, este se puede deshacer vuelta a zl haciendo las operaciones opuestas al revés en la función de la derivada
        float* d_forward = NULL;
        float** hd_forward_pointers = NULL;
        float** d_forward_pointers = NULL;

        float* d_error_array_weight_matrix = NULL;
        float* d_error_array_bias_vector = NULL;
        float** hd_error_weight_matrices_pointers = NULL;
        float** hd_error_bias_vectors_pointers = NULL;
        float** d_error_weight_matrices_pointers = NULL;
        float** d_error_bias_vectors_pointers = NULL;

        //será la matriz de device de tamaño max(nelems_entrada+nelems_salida, nelems_mayor_capa_salida)
        float* d_auxiliar_error_forward_layer = NULL;
        float** d_auxiliar_error_forward_layer_pointers = NULL;
        float** hd_auxiliar_error_forward_layer_pointers = NULL;
        float** hd_auxiliar2_error_forward_layer_pointers = NULL;
        float* d_auxiliar2_error_forward_layer = NULL;
        float** d_auxiliar2_error_forward_layer_pointers = NULL;

        float* d_weight_matrix_momentum = NULL;
        float* d_bias_vector_momentum = NULL;
        float* d_weight_matrix_velocity = NULL;
        float* d_bias_vector_velocity = NULL;
        
    public:
        //dev_act_func and dev_act_der_func are __device__ float func(), need to be casted to (const void*)
        Layer( int sz, func_t dev_act_func, func_t dev_act_der_func );
        ~Layer();

        void showInfo();
        void showWeightBias();
        void showErrorWeightBias();
        void showAuxiliarExpandReduce();
        void showForward();

        int getSize();
        float* getDeviceForward();
        float** getDeviceForwardPointers();
        float** getAuxiliarExpandReduceMatrixPointers();
        float** getDeviceAuxiliarErrorForwardLayerPointers();
        float** getDeviceAuxiliar2ErrorForwardLayerPointers();
        int getTotalElementsBiasVectors();
        int getTotalElementsWeightMatrices();

        void setMaxNumThreads(int set);
        void setInputSize(int is);
        void setNumberInputExamples(int set);
        void setAuxiliarExpandReduceMatrix(float* set);
        void setNumberNetworks(int nn);
        void setIsFirstLayer(bool set);
        void setIsTraining(bool set);
        void setCublasHandle(cublasHandle_t* h);

        void initWeightBiasValues(curandGenerator_t curandGenerator);

        void forward(cudaStream_t stream, float** d_input_pointers, int num_inputs);
        void forward(cudaStream_t stream, Layer* previous_layer, int num_inputs);

        void backward(cudaStream_t stream, Layer* previous_layer, int num_outputs);
        void backward(cudaStream_t stream, float** input_pointers, int num_outputs);

        void applyGradientSGD(cudaStream_t stream, float lrate);
        void applyGradientAdam(cudaStream_t stream, float lrate, float* params, int nparams);

        void allocWeightMatricesMemory();
        void freeWeightMatricesMemory();

        void allocForwardMemory();
        void freeForwardMemory();

        void allocBackwardMemory(int batch_size, float* d_aux_error_matrix, float* d_aux2_error_matrix);
        void freeBackwardMemory();

        void copyWeightBias( float* h_weight, float* h_bias );
        void storeBiasVectorsWeightMatrices( float* h_weight, float* h_bias );

        /*int getInputSize();
        
        int getNumberNetwors();
        float* getHostWeightMatrix();
        float* getHostBiasVector();
        float* getDeviceForward();

        void setInputSize(int sz);
        void setNumberNetwors(int nn);
        void setAuxiliarTransposeMatrix(float* d_aux_mtr);
        void setIsTraining(bool new_training);

        void forward(float* d_input_values);
        void forward(Layer* previous_layer);
        void copyForwardInHostPagedMemory(float* h_paged_mem_pointer);

        float obtainCostFunction();

        void applyLossFunction();*/
};

#endif