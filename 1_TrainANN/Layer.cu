#include "Layer.cuh"

#define N 10

using namespace std;

Layer::Layer(int sz, func_t dev_act_func, func_t dev_act_der_func) {
	size = sz;
    activation_function = dev_act_func;
    activation_derivative_function = dev_act_der_func;
}

Layer::~Layer() {
    handle = NULL;
}

void Layer::showInfo() {
    printf("\n\t\tInput size: %d", input_size);
    printf("\n\t\tSize: %d", size);
    printf("\n\t\tNumber of networks: %d", number_networks);
    printf("\n\t\tFirst layer?: %s", is_first_layer ? "Yes" : "No");
    printf("\n\t\tIs training?: %s", is_training ? "Yes" : "No");
    printf("\n");
}

void Layer::showWeightBias() {
    float* h_weight_m = new float[input_size * size];
    float* h_bias_v = new float[size];
    for (int i = 0; i < number_networks; i++) {
        cudaMemcpy(h_weight_m, hd_weight_matrices_pointers[i], input_size * size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias_v, hd_bias_vectors_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tbias:", h_bias_v, 1, size);
        imprimirMatrizPorPantalla("\n\t\tweight:", h_weight_m, input_size, size); 
    }
    delete h_weight_m;
    delete h_bias_v;
}

void Layer::showErrorWeightBias() {
    float* h_weight_m = new float[input_size * size];
    float* h_bias_v = new float[size];
    for (int i = 0; i < number_networks; i++) {
        cudaMemcpy(h_weight_m, hd_error_weight_matrices_pointers[i], input_size * size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias_v, hd_error_bias_vectors_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tbias:", h_bias_v, 1, size);
        imprimirMatrizPorPantalla("\n\t\tweight:", h_weight_m, input_size, size);
    }
    delete h_weight_m;
    delete h_bias_v;
}

void Layer::showAuxiliarExpandReduce() {
    float* h_auxiliar_expand_reduce_matrix = new float[size];
    for (int i = 0; i < number_networks; i++) {
        cudaMemcpy(h_auxiliar_expand_reduce_matrix, hd_expand_reduce_matrix_pointers[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tauxiliar:", h_auxiliar_expand_reduce_matrix, 1, size);
        
    }
    delete h_auxiliar_expand_reduce_matrix;
}

void Layer::showForward() {
    float* h_forward = new float[number_input_examples * size];
    for (int i = 0; i < number_networks; i++) {
        cudaMemcpy(h_forward, hd_forward_pointers[i], number_input_examples * size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("\n\tNetwork %d:", i);
        imprimirMatrizPorPantalla("\n\t\tforward matrix:", h_forward, number_input_examples, size);
        
    }
    delete h_forward;
}

int Layer::getSize() {
    return size;
}

float* Layer::getDeviceForward() {
    return d_forward;
}

float** Layer::getDeviceForwardPointers() {
    return d_forward_pointers;
}


float** Layer::getAuxiliarExpandReduceMatrixPointers() {
    return d_expand_reduce_matrix_pointers;
}

float** Layer::getDeviceAuxiliarErrorForwardLayerPointers() {
    return d_auxiliar_error_forward_layer_pointers;
}

float** Layer::getDeviceAuxiliar2ErrorForwardLayerPointers() {
    return d_auxiliar2_error_forward_layer_pointers;
}

int Layer::getTotalElementsBiasVectors() {
    return size * number_networks;
}

int Layer::getTotalElementsWeightMatrices() {
    return input_size * size * number_networks;
}

void Layer::setMaxNumThreads(int set) {
    max_num_threads = set;
}

void Layer::setInputSize(int is) {
    input_size = is;
}

void Layer::setNumberInputExamples(int set) {
    number_input_examples = set;
}

void Layer::setAuxiliarExpandReduceMatrix(float* set) {
    d_auxiliar_expand_reduce_matrix = set;
    if (set != NULL) {
        hd_expand_reduce_matrix_pointers = new float* [number_networks];
        cudaMalloc(&d_expand_reduce_matrix_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) { hd_expand_reduce_matrix_pointers[i] = d_auxiliar_expand_reduce_matrix; }
        cudaMemcpy(d_expand_reduce_matrix_pointers, hd_expand_reduce_matrix_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    } else {
        cudaFree(d_expand_reduce_matrix_pointers);
        delete hd_expand_reduce_matrix_pointers;
    }
}

void Layer::setNumberNetworks(int nn) {
    number_networks = nn;
}

void Layer::setIsFirstLayer(bool set) {
    is_first_layer = set;
}

void Layer::setIsTraining(bool set) {
    is_training = set;
}

void Layer::setCublasHandle(cublasHandle_t* h) {
    handle = h;
}

void Layer::initWeightBiasValues(curandGenerator_t curandGenerator) {
    cudaMemset(d_array_bias_vector, 0, nextFourMultiple(size * number_networks) * sizeof(float));
    unsigned long long semilla = rand() % 10000;
    curandSetPseudoRandomGeneratorSeed(curandGenerator, semilla);
    curandGenerateNormal(curandGenerator, (float*)d_array_weight_matrix, nextFourMultiple(input_size * size * number_networks), 0.0, 2.0/(float)input_size );
    cudaDeviceSynchronize();
}

void Layer::forward(cudaStream_t stream, float** d_input_pointers, int num_inputs) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, num_inputs, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, d_input_pointers, d_weight_matrices_pointers, d_forward_pointers, num_inputs, input_size, size, number_networks);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size*num_inputs*number_networks), d_forward, activation_function);
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::forward(cudaStream_t stream, Layer* previous_layer, int num_inputs) {
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_bias_vectors_pointers, d_forward_pointers, num_inputs, 1, size, number_networks);
    productoMatricesBatchDeviceSumC(*handle, previous_layer->getDeviceForwardPointers(), d_weight_matrices_pointers, d_forward_pointers, num_inputs, input_size, size, number_networks);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_inputs * number_networks), d_forward, activation_function);
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_function);
}

void Layer::backward(cudaStream_t stream, Layer* previous_layer, int num_outputs) {

    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);

    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);

    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);

    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);

    //weight error
    productoMatricesTrasposedABatchDevice(*handle, previous_layer->getDeviceForwardPointers(), d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);

    //previous layer error
    productoMatricesTrasposedBBatchDevice(*handle, d_auxiliar_error_forward_layer_pointers, d_weight_matrices_pointers, previous_layer->getDeviceAuxiliar2ErrorForwardLayerPointers(), num_outputs, size, input_size, number_networks);
    cudaMemcpy(d_auxiliar_error_forward_layer, d_auxiliar2_error_forward_layer, nextFourMultiple(num_outputs * input_size * number_networks) * sizeof(float), cudaMemcpyDeviceToDevice);

    /*
    float* h_res = new float[nextFourMultiple(num_outputs * max(size, input_size) * number_networks)];

    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);
    
    cudaMemcpy(h_res, d_forward, nextFourMultiple(size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("aplico la funcion: ", h_res, num_outputs * number_networks, size);

    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);

    cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("multiplico aplicar funcion por error: ", h_res, num_outputs * number_networks, size);

    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);

    cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("divido todos los elementos entre la constante: ", h_res, num_outputs * number_networks, size);

    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);

    cudaMemcpy(h_res, d_error_array_bias_vector, nextFourMultiple(size * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("bias error: ", h_res, number_networks, size);

    //weight error
    productoMatricesTrasposedABatchDevice(*handle, previous_layer->getDeviceForwardPointers(), d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);

    cudaMemcpy(h_res, d_error_array_weight_matrix, nextFourMultiple(input_size * size * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("weight error: ", h_res, input_size* number_networks, size);

    //previous layer error
    productoMatricesTrasposedBBatchDevice(*handle, d_auxiliar_error_forward_layer_pointers, d_weight_matrices_pointers, previous_layer->getDeviceAuxiliar2ErrorForwardLayerPointers(), num_outputs, size, input_size, number_networks);
    cudaMemcpy(d_auxiliar_error_forward_layer, d_auxiliar2_error_forward_layer, nextFourMultiple(num_outputs * input_size * number_networks) * sizeof(float), cudaMemcpyDeviceToDevice);
    
    //printf("\nDIMENSIONES DE LAS MATRICES RESULTADO DE ERROR HACIA ATRAS: %d %d", num_outputs, input_size);
    //float** punteros = new float* [number_networks];
    //cudaMemcpy(punteros, d_auxiliar_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < number_networks; i++) { printf("\nPOS POINTER AUX1: %p", punteros[i]); }
    //cudaMemcpy(punteros, d_auxiliar2_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < number_networks; i++) { printf("\nPOS POINTER AUX2: %p", punteros[i]); }

    cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(input_size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("error a propagar hacia atras calculado: ", h_res, num_outputs * number_networks, input_size);
    */

}

void Layer::backward(cudaStream_t stream, float** input_pointers, int num_outputs) {
    
    /*float* h_res = new float[nextFourMultiple(num_outputs * input_size * number_networks)];
    cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    imprimirMatrizPorPantalla("error que tomo del calculo anterior: ", h_res, num_outputs * number_networks, size);*/

    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);

    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);

    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);

    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);

    //weight error
    productoMatricesTrasposedABatchDevice(*handle, input_pointers, d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);

    //float* h_res = new float[nextFourMultiple(max(input_size, size) * num_outputs * number_networks)];
    //cudaMemcpy(h_res, d_auxiliar_error_forward_layer, nextFourMultiple(input_size * num_outputs * number_networks) * sizeof(float), cudaMemcpyDeviceToHost);
    //imprimirMatrizPorPantalla("error de la capa posterior calculado: ", h_res, num_outputs * number_networks, input_size);
    
    //managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);

    /*
    //applyFunctionVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_forward, activation_derivative_function);
    managedApplyFunction(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_forward, activation_derivative_function);
    //multiplyMatricesSameDimensionsVectorial << < num_blocks_needed_apply_function, num_threads_needed_apply_function, 0, stream >> > (d_auxiliar_error_forward_layer, d_forward);
    managedMultiplyMatricesSameDimensions(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, d_forward);
    //multiplyAllElementsByConstantVectorial << < (int)ceil(nextFourMultiple(num_outputs) / ((float)max_num_threads * 4)), min(max_num_threads, (int)(nextFourMultiple(number_networks * size) / (float)4)), 0, stream >> > (d_error_array_bias_vector, 1 / (float)num_outputs);
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * num_outputs * number_networks), d_auxiliar_error_forward_layer, 1 / (float)num_outputs);
    //bias error
    productoMatricesBatchDevice(*handle, d_expand_reduce_matrix_pointers, d_auxiliar_error_forward_layer_pointers, d_error_bias_vectors_pointers, 1, num_outputs, size, number_networks);
    //weight error
    productoMatricesTrasposedABatchDevice(*handle, input_pointers, d_auxiliar_error_forward_layer_pointers, d_error_weight_matrices_pointers, input_size, num_outputs, size, number_networks);
    */
}

void Layer::applyGradientSGD(cudaStream_t stream, float lrate) {
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(input_size * size * number_networks), d_error_array_weight_matrix, -lrate);
    managedMultiplyAllElementsByConstant(stream, max_num_threads, nextFourMultiple(size * number_networks), d_error_array_bias_vector, -lrate);
    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(input_size * size * number_networks), d_array_weight_matrix, d_error_array_weight_matrix);
    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(size * number_networks), d_array_bias_vector, d_error_array_bias_vector);
}

void Layer::applyGradientAdam(cudaStream_t stream, float lrate, float* params, int nparams) {
    //void managedActualizarValoresMatrizMomentoAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b1, int nrows, int ncols);
    //void managedActualizarValoresMatrizVelocidadAdam(cudaStream_t stream, int max_threads_block, const float* grad, float* mdst, float b2, int nrows, int ncols);
    //void managedCalcularVectorGradienteAdam(cudaStream_t stream, int max_threads_block, float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols);
    managedActualizarValoresMatrizMomentoAdam(stream, max_num_threads, d_error_array_weight_matrix, d_weight_matrix_momentum, params[0], input_size * number_networks, size);
    managedActualizarValoresMatrizMomentoAdam(stream, max_num_threads, d_error_array_bias_vector, d_bias_vector_momentum, params[0], number_networks, size);

    managedActualizarValoresMatrizVelocidadAdam(stream, max_num_threads, d_error_array_weight_matrix, d_weight_matrix_velocity, params[1], input_size * number_networks, size);
    managedActualizarValoresMatrizVelocidadAdam(stream, max_num_threads, d_error_array_bias_vector, d_bias_vector_velocity, params[1], number_networks, size);

    managedCalcularVectorGradienteAdam(stream, max_num_threads, lrate, params[0], params[1], params[2], d_error_array_weight_matrix, d_weight_matrix_momentum, d_weight_matrix_velocity, input_size * number_networks, size);
    managedCalcularVectorGradienteAdam(stream, max_num_threads, lrate, params[0], params[1], params[2], d_error_array_bias_vector, d_bias_vector_momentum, d_bias_vector_velocity, number_networks, size);

    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(input_size * size * number_networks), d_array_weight_matrix, d_error_array_weight_matrix);
    managedSumVectorsSameDimensions(stream, max_num_threads, nextFourMultiple(size * number_networks), d_array_bias_vector, d_error_array_bias_vector);
}

void Layer::allocWeightMatricesMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMalloc( &d_array_weight_matrix, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
        cudaMalloc( &d_array_bias_vector, nextFourMultiple(size * number_networks) * sizeof(float));
        hd_weight_matrices_pointers = new float* [number_networks];
        hd_bias_vectors_pointers = new float* [number_networks];
        cudaMalloc(&d_weight_matrices_pointers, number_networks * sizeof(float*));
        cudaMalloc(&d_bias_vectors_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) {
            hd_weight_matrices_pointers[i] = d_array_weight_matrix + i*(input_size * size);
            hd_bias_vectors_pointers[i] = d_array_bias_vector + i * (size);
        }
        cudaMemcpy(d_weight_matrices_pointers, hd_weight_matrices_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_vectors_pointers, hd_bias_vectors_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    }
}

void Layer::freeWeightMatricesMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaFree(d_array_weight_matrix); d_array_weight_matrix = NULL;
        cudaFree(d_array_bias_vector); d_array_bias_vector = NULL;
        delete hd_weight_matrices_pointers;  hd_weight_matrices_pointers = NULL;
        delete hd_bias_vectors_pointers; hd_bias_vectors_pointers = NULL;
        cudaFree(d_weight_matrices_pointers); d_weight_matrices_pointers = NULL;
        cudaFree(d_bias_vectors_pointers); d_bias_vectors_pointers = NULL;
    }
    input_size = 0;
    size = 0;
    number_networks = 0;
}

void Layer::allocForwardMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0 && number_input_examples > 0) {
        num_blocks_needed_apply_function = (int)ceil((size * number_input_examples) / (float)(max_num_threads * 4));
        num_threads_needed_apply_function = min(max_num_threads, number_input_examples * size * 4);

        cudaMalloc(&d_forward, nextFourMultiple( number_input_examples * size * number_networks ) * sizeof(float));
        hd_forward_pointers = new float* [number_networks];
        cudaMalloc(&d_forward_pointers, number_networks * sizeof(float*));
        for (int i = 0; i < number_networks; i++) {
            hd_forward_pointers[i] = d_forward + ( i * number_input_examples * size);
        }
        cudaMemcpy(d_forward_pointers, hd_forward_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    }
}

void Layer::freeForwardMemory() {
    if (input_size > 0 && size > 0 && number_networks > 0 && number_input_examples > 0) {
        num_blocks_needed_apply_function = 0;
        num_threads_needed_apply_function = 0;

        cudaFree(d_forward); d_forward = NULL;
        delete hd_forward_pointers; hd_forward_pointers = NULL;
        cudaFree(d_forward_pointers); d_forward_pointers = NULL;
    }
    number_input_examples = 0;
}

void Layer::allocBackwardMemory(int batch_size, float* d_aux_error_matrix, float* d_aux2_error_matrix) {
    d_auxiliar_error_forward_layer = d_aux_error_matrix;
    d_auxiliar2_error_forward_layer = d_aux2_error_matrix;

    cudaMalloc( &d_error_array_weight_matrix, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMalloc( &d_error_array_bias_vector, nextFourMultiple(size * number_networks) * sizeof(float));

    cudaMalloc(&d_weight_matrix_momentum, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMalloc(&d_bias_vector_momentum, nextFourMultiple(size * number_networks) * sizeof(float));
    cudaMemset(d_weight_matrix_momentum, 0, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMemset(d_bias_vector_momentum, 0, nextFourMultiple(size * number_networks) * sizeof(float));

    cudaMalloc(&d_weight_matrix_velocity, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMalloc(&d_bias_vector_velocity, nextFourMultiple(size * number_networks) * sizeof(float));
    cudaMemset(d_weight_matrix_velocity, 0, nextFourMultiple(input_size * size * number_networks) * sizeof(float));
    cudaMemset(d_bias_vector_velocity, 0, nextFourMultiple(size * number_networks) * sizeof(float));

    hd_error_weight_matrices_pointers = new float* [number_networks];
    hd_error_bias_vectors_pointers = new float* [number_networks];
    hd_auxiliar_error_forward_layer_pointers = new float* [number_networks];
    hd_auxiliar2_error_forward_layer_pointers = new float* [number_networks];
    cudaMalloc(&d_auxiliar_error_forward_layer_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_auxiliar2_error_forward_layer_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_error_weight_matrices_pointers, number_networks * sizeof(float*));
    cudaMalloc(&d_error_bias_vectors_pointers, number_networks * sizeof(float*));
    for (int i = 0; i < number_networks; i++) {
        hd_error_weight_matrices_pointers[i] = d_error_array_weight_matrix + i*(input_size * size);
        hd_error_bias_vectors_pointers[i] = d_error_array_bias_vector + i * (size);
        hd_auxiliar_error_forward_layer_pointers[i] = d_auxiliar_error_forward_layer + i * (size * batch_size);
        hd_auxiliar2_error_forward_layer_pointers[i] = d_auxiliar2_error_forward_layer + i * (size * batch_size);
        //printf("\nDISTANCIA SUMADA RED %d: %d", i, i * (size * batch_size));
        //printf("\nPosicion hd_auxiliar_error_forward_layer_pointers: %p", hd_auxiliar_error_forward_layer_pointers[i]);
        //printf("\nPosicion hd_auxiliar2_error_forward_layer_pointers: %p", hd_auxiliar2_error_forward_layer_pointers[i]);
    }
    cudaMemcpy(d_error_weight_matrices_pointers, hd_error_weight_matrices_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_error_bias_vectors_pointers, hd_error_bias_vectors_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_auxiliar_error_forward_layer_pointers, hd_auxiliar_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_auxiliar2_error_forward_layer_pointers, hd_auxiliar2_error_forward_layer_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
}

void Layer::freeBackwardMemory() {
    d_auxiliar_error_forward_layer = NULL;
    d_auxiliar2_error_forward_layer = NULL;
    if (hd_auxiliar_error_forward_layer_pointers != NULL) { delete hd_auxiliar_error_forward_layer_pointers; hd_auxiliar_error_forward_layer_pointers = NULL; }
    if (hd_auxiliar2_error_forward_layer_pointers != NULL) { delete hd_auxiliar2_error_forward_layer_pointers; hd_auxiliar2_error_forward_layer_pointers = NULL; }
    if (d_auxiliar_error_forward_layer_pointers != NULL) { cudaFree(d_auxiliar_error_forward_layer_pointers); d_auxiliar_error_forward_layer_pointers = NULL; }
    if (d_auxiliar2_error_forward_layer_pointers != NULL) { cudaFree(d_auxiliar2_error_forward_layer_pointers); d_auxiliar2_error_forward_layer_pointers = NULL; }
    if (d_error_array_weight_matrix != NULL) { cudaFree(d_error_array_weight_matrix); d_error_array_weight_matrix = NULL; }
    if (d_error_array_bias_vector != NULL) { cudaFree(d_error_array_bias_vector); d_error_array_bias_vector = NULL; }
    if (hd_error_weight_matrices_pointers != NULL) { delete hd_error_weight_matrices_pointers;  hd_error_weight_matrices_pointers = NULL; }
    if (hd_error_bias_vectors_pointers != NULL) { delete hd_error_bias_vectors_pointers; hd_error_bias_vectors_pointers = NULL; }
    if (d_error_weight_matrices_pointers != NULL) { cudaFree(d_error_weight_matrices_pointers); d_error_weight_matrices_pointers = NULL; }
    if (d_error_bias_vectors_pointers != NULL) { cudaFree(d_error_bias_vectors_pointers); d_error_bias_vectors_pointers = NULL; }

    if (d_weight_matrix_momentum != NULL) { cudaFree(d_weight_matrix_momentum); d_weight_matrix_momentum = NULL; }
    if (d_bias_vector_momentum != NULL) { cudaFree(d_bias_vector_momentum); d_bias_vector_momentum = NULL; }
    if (d_weight_matrix_velocity != NULL) { cudaFree(d_weight_matrix_velocity); d_weight_matrix_velocity = NULL; }
    if (d_bias_vector_velocity != NULL) { cudaFree(d_bias_vector_velocity); d_bias_vector_velocity = NULL; }
}

void Layer::copyWeightBias(float* h_weight, float* h_bias) {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        cudaMemcpy(d_array_weight_matrix, h_weight, input_size * size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array_bias_vector, h_bias, size * number_networks * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void Layer::storeBiasVectorsWeightMatrices(float* h_weight, float* h_bias) {
    if (input_size > 0 && size > 0 && number_networks > 0) {
        //printf("\nPunteros: %p %p\n", d_array_weight_matrix, d_array_bias_vector);
        cudaMemcpy(h_weight, d_array_weight_matrix, input_size * size * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias, d_array_bias_vector, size * number_networks * sizeof(float), cudaMemcpyDeviceToHost);
    }
}