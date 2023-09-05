#include "Layer.cuh"
#include "HostPinnedDeviceMatrix.cuh"

#ifndef NETWORK
#define NETWORK

class Network;

using func_backprop = void(*) (Network* n, float, float*, int);

void applyVGradSGD(Network* n, float lrate, float* params, int nparams);
void applyVGradAdam(Network* n, float lrate, float* params, int nparams);

class Network {
    private:
        int max_num_threads;

        int input_size;
        int output_size;
        int number_networks;
        int number_layers;
        Layer** layers;
        func2_t loss_function;
        func2_t derivative_loss_function;

        int max_train_number_examples;
        int max_validation_number_examples;
        int max_batch_size;
        int max_layer_size = 0;

        int current_train_number_examples;
        int current_validation_number_examples;
        int current_batch_size;

        bool check_errors_CUDA = false;

        cudaStream_t stream_principal;
        cudaStream_t stream_transferencia_output;

        HostPinnedDeviceMatrix* input_train = NULL;
        HostPinnedDeviceMatrix* output_train = NULL;

        HostPinnedDeviceMatrix* input_validation = NULL;
        HostPinnedDeviceMatrix* output_validation = NULL;

        //AUXILIAR_POINTERS
        //vector of 1's for repeating vector multiple times in matrix or sum all matrix cols in vector
        float* d_auxiliar_expand_reduce_matrix = NULL;
        //matrix of next four multiple of max_number_examples rows and output_size cols, when using more than one network, here the average of each network prediction is calculated
        //also is used for cost function summatory, perfectly fits in
        float* d_output_forward_multiple_nn_sum = NULL;
        float** d_output_forward_multiple_nn_sum_pointers = NULL;
        //matrix where changed-order output is copied and loss function is calculated, and also is stored backpropagated error of current layer
        float* d_auxiliar_matrix_loss_function_error_backprop = NULL;
        float* d_auxiliar2_matrix_loss_function_error_backprop = NULL;

        //also works as output pointers
        float** d_input_train_pointers = NULL;
        float** d_input_validation_pointers = NULL;

        cublasHandle_t handle;

        /*func2_t loss_function = NULL;
        */

    public:
        Network( int is, int nn, int nl, Layer** ls, func2_t ls_fn, func2_t dls_fn, bool norm_input);
        ~Network();

        void showInfoAboutNetwork();
        void showWeightsBiasesLayers();
        void showErrorWeightsBiasesLayers();
        void showAuxiliarExpandReduceMatrices();
        void showForwardMatrices();

        int getNumberNetwors();
        int getNumberLayers();
        Layer** getLayers();
        cudaStream_t getStreamPrincipal();

        void initForward( int max_num_input_examples_expected );
        void initForwardTrain(int max_train_examples, int max_validation_examples, int m_batch_size);

        void initWeightBiasValues();

        const void copyInputOutputTrain(int num_examples, float* input_data, float* output_data);
        const void copyInputOutputValidation(int num_examples, float* input_data, float* output_data);

        const void forward( int num_examples, float* input_data, float* output_pointer_dest);
        
        const void forwardTrain(int num_examples);
        const void forwardTrain(int num_examples, int batch_size, float** d_pointers);

        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id);
        float* trainGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int offset_id, int* batch_ids);

        float* validationGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id);
        float* validationGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int offset_id, int* batch_ids);

        float* backwardPhase(int num_examples, int offset_id, int* early_counters);
        float* backwardPhase(int num_examples, int batch_size, int offset_id, int* batch_ids, int* early_counters);

        void noBackwardNetworksOutCounter(int batch_size, int* early_counters);

        void epochAllExamples(float lrate, float* params, int nparams, func_backprop backprop_function, int number_train_batches, int number_remainder_train_examples, int repeat_train_arr, int number_validation_batches, int number_remainder_validation_examples, int repeat_validation_arr, int* train_indices, int* val_indices, float* cost_train, float* cost_val, int* early_counters);

        //first func_lrate parameter value (number of first epoch) is 0, not 1
        void trainAllExamplesMaxBatch(func_lrate function_learning_rate, int nparams, float* params, func_backprop backprop_function, int nepochs, int show_per_epoch, float convergence, float min_err_start_early_stop, int count_early_stop);

        void finalizeForwardBackward();

        void storeNetworkInFile(char* name);
        void loadNetworkFromFile(char* name);



        //void initBackwardADAM();
        //void finalizeBackwardADAM();

        
        /*int getInputSize();
        int getOutputSize();
        
        void changeIsTraining(bool new_training);
        float* getWeightMatrix();
        float* getBiasVector();*/
        
};

#endif