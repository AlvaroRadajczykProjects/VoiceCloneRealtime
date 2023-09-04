#include "Network.cuh"
#include "funciones_archivos.h"

using namespace std;

//no params
void applyVGradSGD(Network* n, float lrate, float* params, int nparams) {
	if (nparams != 0) {
		printf("\nNetwork applyVGradSGD: incorrect number of parameters (expected 0, received %d)", nparams);
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < n->getNumberLayers(); i++) {
		n->getLayers()[i]->applyGradientSGD(n->getStreamPrincipal(), lrate);
	}
}

//params: b1, b2, epsilon
void applyVGradAdam(Network* n, float lrate, float* params, int nparams) {
	if (nparams != 3) {
		printf("\nNetwork applyVGradSGD: incorrect number of parameters (expected 3, received %d)", nparams);
		exit(EXIT_FAILURE);
	}
	//printf("\nParams: ");
	//for (int i = 0; i < nparams; i++) { printf("%.16f, ", params[i]); }
	//printf("\n");
	for (int i = 0; i < n->getNumberLayers(); i++) {
		n->getLayers()[i]->applyGradientAdam(n->getStreamPrincipal(), lrate, params, nparams);
	}
}

Network::Network(int is, int nn, int nl, Layer** ls, func2_t ls_fn, func2_t dls_fn) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	max_num_threads = deviceProp.maxThreadsPerBlock;

	loss_function = ls_fn;
	derivative_loss_function = dls_fn;
	input_size = is;
	output_size = ls[nl - 1]->getSize();
	number_networks = nn;
	number_layers = nl;
	layers = ls;
	layers[0]->setIsFirstLayer(true);
	cublasCreate_v2(&handle);
	for (int i = 0; i < number_layers; i++) {
		max_layer_size = max(max_layer_size, layers[i]->getSize());
		layers[i]->setCublasHandle(&handle);
		layers[i]->setNumberNetworks(number_networks);
		layers[i]->setIsTraining(false);
		if (i == 0) { layers[i]->setInputSize(input_size); }
		else { layers[i]->setInputSize(layers[i - 1]->getSize()); }
		layers[i]->setMaxNumThreads(max_num_threads);
		layers[i]->allocWeightMatricesMemory();
	}
	cudaDeviceSynchronize();
}

Network::~Network() {
	for (int i = 0; i < number_layers; i++) {
		layers[i]->freeWeightMatricesMemory();
		delete layers[i];
	}
	delete layers;
	cublasDestroy_v2(handle);
	cudaDeviceSynchronize();
}

void Network::showInfoAboutNetwork() {
	printf("\n");
	printf("\nINFO ABOUT THE NETWORK");
	printf("\n======================");
	printf("\nInput size (number of each example attributes): %d", input_size);
	//printf("\nMax input examples (this network can forward training and/or predicting): %d", input_size);
	printf("\nOutput size (number of each prediction): %d", output_size);
	printf("\nNumber of networks (multiple networks can be trained for ensemble averaging with multiple similar neural networks in one device): %d", number_networks);
	printf("\nNumber of layers (all networks are similar, same shape, different initialization values): %d", number_layers);
	printf("\nMax layer size: %d", max_layer_size);
	printf("\nLayers dimensions:");
	for (int i = 0; i < number_layers; i++) {
		printf("\n\tLayer %d:", i);
		layers[i]->showInfo();
	}
	printf("\n");
}

void Network::showWeightsBiasesLayers() {
	printf("\n");
	printf("\nWEIGHTS AND BIASES");
	printf("\n==================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showWeightBias();
	}
	printf("\n");
}

void Network::showErrorWeightsBiasesLayers() {
	printf("\n");
	printf("\nERROR WEIGHTS AND BIASES");
	printf("\n========================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showErrorWeightBias();
	}
	printf("\n");
}

void Network::showAuxiliarExpandReduceMatrices() {
	printf("\n");
	printf("\nAUXILIAR EXPAND AND REDUCE VECTORS (is only one, but check all networks match the same)");
	printf("\n==================================------------------------------------------------------");
	printf("\n");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showAuxiliarExpandReduce();
	}
	printf("\n");
}

void Network::showForwardMatrices() {
	printf("\n");
	printf("\nFORWARD MATRICES");
	printf("\n================");
	for (int i = 0; i < number_layers; i++) {
		printf("\nLayer %d:", i);
		layers[i]->showForward();
	}
	printf("\n");
}

int Network::getNumberNetwors() {
	return number_networks;
}

int Network::getNumberLayers() {
	return number_layers;
}

Layer** Network::getLayers() {
	return layers;
}

cudaStream_t Network::getStreamPrincipal() {
	return stream_principal;
}

void Network::initForward(int max_num_input_examples_expected) {
	max_train_number_examples = max_num_input_examples_expected;

	cudaStreamCreate(&stream_principal);
	cudaStreamCreate(&stream_transferencia_output);
	cublasSetStream_v2(handle, stream_principal);

	input_train = new HostPinnedDeviceMatrix(max_train_number_examples, input_size, 4, cudaHostAllocWriteCombined);
	output_train = new HostPinnedDeviceMatrix(max_train_number_examples, output_size, 4, cudaHostAllocWriteCombined);

	int* hd_input_pointers = new int [number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_pointers[i] = 0; }
	d_input_train_pointers = input_train->generateDeviceRowsPointers(0, number_networks, hd_input_pointers);
	delete hd_input_pointers;

	int tam = nextFourMultiple(max(max(max_batch_size, number_networks), output_size));
	cudaMalloc(&d_auxiliar_expand_reduce_matrix, tam * sizeof(float));
	float* h_auxiliar_expand_reduce_matrix = new float[tam];
	for (int i = 0; i < tam; i++) { h_auxiliar_expand_reduce_matrix[i] = 1.0f; }
	cudaMemcpy(d_auxiliar_expand_reduce_matrix, h_auxiliar_expand_reduce_matrix, tam * sizeof(float), cudaMemcpyHostToDevice);
	delete h_auxiliar_expand_reduce_matrix;
	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(max_train_number_examples);
		layers[i]->setAuxiliarExpandReduceMatrix(d_auxiliar_expand_reduce_matrix);
		layers[i]->allocForwardMemory();
	}

	cudaMalloc(&d_output_forward_multiple_nn_sum, nextFourMultiple(number_networks * max_train_number_examples * output_size) * sizeof(float));

	//Cublas warmup
	productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);

	cudaDeviceSynchronize();
}

void Network::initForwardTrain(int max_train_examples, int max_validation_examples, int m_batch_size) {

	if (m_batch_size > max_train_examples || (m_batch_size > max_validation_examples && max_validation_examples != 0)) {
		printf("\nIn initForwardTrain: batch size cannot be bigger than max_train_examples and max_validation_examples if max_validation_examples != 0");
		return;
	}

	if (max_train_examples < max_validation_examples) {
		printf("\nIn initForwardTrain: max_train_examples must be equal or bigger than max_validation_examples");
		return;
	}

	max_batch_size = m_batch_size;
	max_train_number_examples = max_train_examples;
	max_validation_number_examples = max_validation_examples;

	cudaStreamCreate(&stream_principal);
	cudaStreamCreate(&stream_transferencia_output);
	cublasSetStream_v2(handle, stream_principal);

	input_train = new HostPinnedDeviceMatrix(max_train_number_examples, input_size, 4, cudaHostAllocWriteCombined);
	output_train = new HostPinnedDeviceMatrix(max_train_number_examples, output_size, 4, cudaHostAllocWriteCombined);

	input_validation = new HostPinnedDeviceMatrix(max_validation_number_examples, input_size, 4, cudaHostAllocWriteCombined);
	output_validation = new HostPinnedDeviceMatrix(max_validation_number_examples, output_size, 4, cudaHostAllocWriteCombined);

	int* hd_input_pointers = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_pointers[i] = 0; }
	d_input_train_pointers = input_train->generateDeviceRowsPointers(0, number_networks, hd_input_pointers);
	
	if (max_validation_number_examples > 0) {
		d_input_validation_pointers = input_validation->generateDeviceRowsPointers(0, number_networks, hd_input_pointers);
	}
	
	delete hd_input_pointers;

	/*float** hd_input_pointers = new float* [number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_pointers[i] = d_pinned_input_train_matrix + 0; }
	cudaMalloc(&d_input_train_pointers, number_networks * sizeof(float*));
	cudaMemcpy(d_input_train_pointers, hd_input_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_input_pointers;

	float** hd_input_val_pointers = new float* [number_networks];
	for (int i = 0; i < number_networks; i++) { hd_input_val_pointers[i] = d_pinned_input_validation_matrix + 0; }
	cudaMalloc(&d_input_validation_pointers, number_networks * sizeof(float*));
	cudaMemcpy(d_input_validation_pointers, hd_input_val_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_input_val_pointers;*/

	int tam = nextFourMultiple(max_batch_size * number_networks * output_size);
	cudaMalloc(&d_auxiliar_expand_reduce_matrix, tam * sizeof(float));
	float* h_auxiliar_expand_reduce_matrix = new float[tam];
	for (int i = 0; i < tam; i++) { h_auxiliar_expand_reduce_matrix[i] = 1.0f; }
	cudaMemcpy(d_auxiliar_expand_reduce_matrix, h_auxiliar_expand_reduce_matrix, tam * sizeof(float), cudaMemcpyHostToDevice);
	delete h_auxiliar_expand_reduce_matrix;

	//max_input_number_examples instead of max_batch_size * number_networks?
	cudaMalloc(&d_auxiliar_matrix_loss_function_error_backprop, nextFourMultiple(max_batch_size * number_networks * max_layer_size) * sizeof(float));
	cudaMalloc(&d_auxiliar2_matrix_loss_function_error_backprop, nextFourMultiple(max_batch_size * number_networks * max_layer_size) * sizeof(float));

	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(max_batch_size);
		layers[i]->setAuxiliarExpandReduceMatrix(d_auxiliar_expand_reduce_matrix);
		layers[i]->allocForwardMemory();
		layers[i]->allocBackwardMemory(m_batch_size, d_auxiliar_matrix_loss_function_error_backprop, d_auxiliar2_matrix_loss_function_error_backprop);
		layers[i]->setIsTraining(true);
	}

	cudaMalloc(&d_output_forward_multiple_nn_sum, nextFourMultiple(number_networks * max_batch_size * output_size) * sizeof(float));
	float** hd_output_forward_multiple_nn_sum_pointers = new float* [number_networks];
	cudaMalloc(&d_output_forward_multiple_nn_sum_pointers, number_networks * sizeof(float*));
	for (int i = 0; i < number_networks; i++) { hd_output_forward_multiple_nn_sum_pointers[i] = d_output_forward_multiple_nn_sum + i * output_size; }
	cudaMemcpy(d_output_forward_multiple_nn_sum_pointers, hd_output_forward_multiple_nn_sum_pointers, number_networks * sizeof(float*), cudaMemcpyHostToDevice);
	delete hd_output_forward_multiple_nn_sum_pointers;

	int first_max = max(max_layer_size, input_size);
	int second_max = 0;
	for (int i = 0; i < number_layers; i++) {
		if (max(second_max, layers[i]->getSize()) < first_max) { second_max = max(second_max, layers[i]->getSize()); }
	}

	//Cublas warmup
	productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);

	cudaDeviceSynchronize();
}

void Network::initWeightBiasValues() {
	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MT19937);
	curandSetGeneratorOrdering(curandGenerator, CURAND_ORDERING_PSEUDO_BEST);
	cudaDeviceSynchronize();
	for (int i = 0; i < number_layers; i++) {
		layers[i]->initWeightBiasValues(curandGenerator);
	}
	cudaDeviceSynchronize();
	curandDestroyGenerator(curandGenerator);
}

const void Network::copyInputOutputTrain(int num_examples, float* input_data, float* output_data) {
	if (num_examples <= max_train_number_examples) {
		input_train->copyHostToDevice(input_data, num_examples, stream_principal);
		output_train->copyHostToDevice(output_data, num_examples, stream_transferencia_output);
		//input_train->showDeviceMatrix("input train", stream_principal);
		//output_train->showDeviceMatrix("output train", stream_principal);
	}
	else {
		printf("\nCannot copy input and output, more examples than max number of examples defined in initForward");
	}
}

const void Network::copyInputOutputValidation(int num_examples, float* input_data, float* output_data) {
	if (num_examples <= max_validation_number_examples) {
		input_validation->copyHostToDevice(input_data, num_examples, stream_principal);
		output_validation->copyHostToDevice(output_data, num_examples, stream_transferencia_output);
		//input_validation->showDeviceMatrix("input validation", stream_principal);
		//output_validation->showDeviceMatrix("input validation", stream_principal);
	}
	else {
		printf("\nCannot copy input and output, more examples than max number of examples defined in initForward");
	}
}

const void Network::forward(int num_examples, float* input_data, float* output_pointer_dest) {
	if (num_examples <= max_train_number_examples) {
		//cudaMemcpyAsync(h_pinned_input_train_matrix, input_data, num_examples * input_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
		//cudaMemcpyAsync(d_pinned_input_train_matrix, h_pinned_input_train_matrix, num_examples * input_size * sizeof(float), cudaMemcpyHostToDevice, stream_principal);
		input_train->copyHostToDevice(input_data, num_examples, stream_principal);
		layers[0]->forward(stream_principal, d_input_train_pointers, num_examples);
		for (int i = 1; i < number_layers; i++) {
			layers[i]->forward(stream_principal, layers[i - 1], num_examples);
		}

		if (number_networks == 1) {
			//cudaMemcpyAsync(h_pinned_output_train_matrix, layers[number_layers - 1]->getDeviceForward(), num_examples * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_principal);
			//cudaMemcpyAsync(output_pointer_dest, h_pinned_output_train_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
			output_train->copyFromDevice(layers[number_layers - 1]->getDeviceForward(), num_examples, stream_principal);
			output_train->copyDeviceToHost(output_pointer_dest, num_examples, stream_principal);
		}
		else {
			if (max_train_number_examples == 1) {
				productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, layers[number_layers - 1]->getDeviceForward(), d_output_forward_multiple_nn_sum, 1, number_networks, output_size);

				managedMultiplyAllElementsByConstant(stream_principal, max_num_threads, nextFourMultiple(output_size), d_output_forward_multiple_nn_sum, 1 / (float)number_networks);
				//multiplyAllElementsByConstantVectorial << < (int)ceil(nextFourMultiple(num_examples * output_size) /(float)(max_num_threads*4)), min(max_num_threads, nextFourMultiple(num_examples * output_size) / 4), 0, stream_principal >> > (d_output_forward_multiple_nn_sum, 1 / (float)number_networks);
				output_train->copyFromDevice(d_output_forward_multiple_nn_sum, 1, stream_principal);
				output_train->copyDeviceToHost(output_pointer_dest, 1, stream_principal);
				//cudaMemcpyAsync(h_pinned_output_train_matrix, d_output_forward_multiple_nn_sum, num_examples * output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_principal);
				//cudaMemcpyAsync(output_pointer_dest, h_pinned_output_train_matrix, num_examples * output_size * sizeof(float), cudaMemcpyHostToHost, stream_principal);
			}
			else {
				//habrá que hacer el sumatorio de todas las matrices al de todas las redes, y multiplicarles 1/numero_redes
			}
		}
		cudaStreamSynchronize(stream_principal);
	}
	else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForward");
	}
}

const void Network::forwardTrain(int num_examples) {
	layers[0]->forward(stream_principal, d_input_train_pointers, num_examples);
	for (int i = 1; i < number_layers; i++) {
		layers[i]->forward(stream_principal, layers[i - 1], num_examples);
	}
	cudaStreamSynchronize(stream_principal);
	cudaStreamSynchronize(stream_transferencia_output);
}

//first batch_id = 0
const void Network::forwardTrain(int num_examples, int batch_size, float** d_pointers) {
	layers[0]->forward(stream_principal, d_pointers, batch_size);
	for (int i = 1; i < number_layers; i++) {
		layers[i]->forward(stream_principal, layers[i - 1], batch_size);
	}
	cudaStreamSynchronize(stream_principal);
	cudaStreamSynchronize(stream_transferencia_output);
}

void Network::noBackwardNetworksOutCounter(int batch_size, int* early_counters) {
	for (int i = 0; i < number_networks; i++) {
		if (early_counters[i] < 1) {
			cudaMemset(d_auxiliar_matrix_loss_function_error_backprop + i*(batch_size * output_size), 0, batch_size * output_size * sizeof(float));
		}
	}
}

float* Network::trainGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id) {
	int* pos = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { pos[i] = 0; }
	float* res = trainGetCostFunctionAndCalculateLossFunction(num_examples, num_examples, offset_id, pos);
	delete pos;
	return res;
}

//first batch_id = 0
float* Network::trainGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int offset_id, int* batch_ids) {
	if (num_examples <= max_train_number_examples && batch_size <= max_batch_size) {
		if (num_examples % batch_size == 0) {
			int num_elems_batch = batch_size * input_size;

			d_input_train_pointers = input_train->generateDeviceRowsPointers(offset_id, number_networks, batch_ids);
			forwardTrain(num_examples, batch_size, d_input_train_pointers);

			num_elems_batch = batch_size * output_size;

			//apply cost function
			for (int i = 0; i < number_networks; i++) {
				managedApplyLossFunction(stream_principal, max_num_threads, num_elems_batch,
					layers[number_layers - 1]->getDeviceForward() + (i * output_size),
					output_train->getDevicePointer() + (batch_ids[i] * output_size),
					d_auxiliar_matrix_loss_function_error_backprop + (i * output_size),
					loss_function
				);
			}

			//obtain cost function
			float* cost_function_result = new float[number_networks];
			float* d_res = 0;
			cudaMalloc(&d_res, number_networks * sizeof(float));
			productoMatricesBatchDevice(handle, layers[number_layers - 1]->getAuxiliarExpandReduceMatrixPointers(), layers[number_layers - 1]->getDeviceAuxiliarErrorForwardLayerPointers(), d_output_forward_multiple_nn_sum_pointers, 1, batch_size, output_size, number_networks);

			managedMultiplyAllElementsByConstant(stream_principal, max_num_threads, output_size * number_networks, d_output_forward_multiple_nn_sum, 1 / (float)(batch_size));

			productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, d_output_forward_multiple_nn_sum, d_res, 1, output_size, number_networks);
			cudaMemcpy(cost_function_result, d_res, number_networks * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < number_networks; i++) { cost_function_result[i] = cost_function_result[i] / (float)output_size; }
			cudaFree(d_res);

			//apply loss function
			for (int i = 0; i < number_networks; i++) {
				managedApplyLossFunction(stream_principal, max_num_threads, num_elems_batch,
					layers[number_layers - 1]->getDeviceForward() + (i * output_size),
					output_train->getDevicePointer() + (batch_ids[i] * output_size),
					d_auxiliar_matrix_loss_function_error_backprop + (i * output_size),
					derivative_loss_function
				);
			}

			return cost_function_result;
		}
		else {
			printf("\nwhen batch forwardTrain, num_examples % batch_size must be 0");
		}
	}
	else {
		printf("\nCannot make forward, more examples or bigger batch size than defined in initForwardTrain");
	}
	return NULL;
}

float* Network::validationGetCostFunctionAndCalculateLossFunction(int num_examples, int offset_id) {
	int* pos = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { pos[i] = 0; }
	//printf("\nOFFSET ID: %d; NUMBER EXAMPLES: %d", offset_id, num_examples);
	float* res = validationGetCostFunctionAndCalculateLossFunction(num_examples, num_examples, offset_id, pos);
	delete pos;
	return res;
}

//first batch_id = 0
float* Network::validationGetCostFunctionAndCalculateLossFunction(int num_examples, int batch_size, int offset_id, int* batch_ids) {
	if (num_examples <= max_validation_number_examples && batch_size <= max_batch_size) {
		if (num_examples % batch_size == 0) {
			int num_elems_batch = batch_size * input_size;

			d_input_validation_pointers = input_validation->generateDeviceRowsPointers(offset_id, number_networks, batch_ids);
			forwardTrain(num_examples, batch_size, d_input_validation_pointers);

			num_elems_batch = batch_size * output_size;

			//apply cost function
			for (int i = 0; i < number_networks; i++) {
				managedApplyLossFunction(stream_principal, max_num_threads, num_elems_batch,
					layers[number_layers - 1]->getDeviceForward() + (i * output_size),
					output_validation->getDevicePointer() + (batch_ids[i] * output_size),
					d_auxiliar_matrix_loss_function_error_backprop + (i * output_size),
					loss_function
				);
			}

			//obtain cost function
			float* cost_function_result = new float[number_networks];
			float* d_res = 0;
			cudaMalloc(&d_res, number_networks * sizeof(float));
			productoMatricesBatchDevice(handle, layers[number_layers - 1]->getAuxiliarExpandReduceMatrixPointers(), layers[number_layers - 1]->getDeviceAuxiliarErrorForwardLayerPointers(), d_output_forward_multiple_nn_sum_pointers, 1, batch_size, output_size, number_networks);

			managedMultiplyAllElementsByConstant(stream_principal, max_num_threads, output_size * number_networks, d_output_forward_multiple_nn_sum, 1 / (float)(batch_size));

			/*
			float* matriz_Cost = new float[number_networks * output_size];
			cudaMemcpy(matriz_Cost, d_output_forward_multiple_nn_sum, number_networks * output_size * sizeof(float), cudaMemcpyDeviceToHost);
			imprimirMatrizPorPantalla("Error de coste val agrupando batch con mul:", matriz_Cost, number_networks, output_size);
			delete matriz_Cost;
			*/

			productoMatricesDevice(handle, d_auxiliar_expand_reduce_matrix, d_output_forward_multiple_nn_sum, d_res, 1, output_size, number_networks);
			cudaMemcpy(cost_function_result, d_res, number_networks * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < number_networks; i++) { cost_function_result[i] = cost_function_result[i] / (float)output_size; }
			cudaFree(d_res);

			return cost_function_result;
		}
		else {
			printf("\nwhen batch forwardTrain, num_examples % batch_size must be 0");
		}
	}
	else {
		printf("\nCannot make forward, more examples or bigger batch size than defined in initForwardTrain");
	}
	return NULL;
}

float* Network::backwardPhase(int num_examples, int offset_id, int* early_counters) {
	int* pos = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { pos[i] = 0; }
	float* res = backwardPhase(num_examples, num_examples, offset_id, pos, early_counters);
	delete pos;
	return res;
}

float* Network::backwardPhase(int num_examples, int batch_size, int offset_id, int* batch_ids, int* early_counters) {
	if (num_examples <= max_train_number_examples && batch_size <= max_batch_size) {
		if (num_examples % batch_size == 0) {
			float* cost_function = trainGetCostFunctionAndCalculateLossFunction(num_examples, batch_size, offset_id, batch_ids);
			if(early_counters!= NULL){ noBackwardNetworksOutCounter(batch_size, early_counters); }
			for (int i = number_layers - 1; i > 0; i--) {
				layers[i]->backward(stream_principal, layers[i - 1], batch_size);
			}
			layers[0]->backward(stream_principal, d_input_train_pointers, batch_size);
			return cost_function;
		}
		else {
			printf("\nwhen batch backwardPhase, num_examples % batch_size must be 0");
		}
	}
	else {
		printf("\nCannot make forward, more examples than max number of examples defined in initForwardTrain");
	}
	return NULL;
}

void Network::epochAllExamples(float lrate, float* params, int nparams, func_backprop backprop_function, int number_train_batches, int number_remainder_train_examples, int repeat_train_arr, int number_validation_batches, int number_remainder_validation_examples, int repeat_validation_arr, int* train_indices, int* val_indices, float* cost_train, float* cost_val, int* early_counters) {

	memset(train_indices, 0, number_train_batches * repeat_train_arr * sizeof(float));
	for (int i = 0; i < number_train_batches; i++) { train_indices[i] = i * max_batch_size; }
	edu_shuffle(train_indices, number_train_batches);
	for (int i = 1; i < repeat_train_arr; i++) { memcpy(train_indices + i * number_train_batches, train_indices, number_train_batches * sizeof(int)); }

	/*printf("\nIndices entrenamiento: ");
	for (int i = 0; i < number_train_batches * repeat_train_arr; i++) { printf("%d, ", train_indices[i]); }
	printf("\n");*/

	memset(val_indices, 0, number_validation_batches * repeat_validation_arr * sizeof(float));
	for (int i = 0; i < number_validation_batches; i++) { val_indices[i] = i * max_batch_size; }
	edu_shuffle(val_indices, number_validation_batches);
	for (int i = 1; i < repeat_validation_arr; i++) { memcpy(val_indices + i * number_validation_batches, val_indices, number_validation_batches * sizeof(int)); }
	
	/*printf("\nIndices validacion: ");
	for (int i = 0; i < number_validation_batches * repeat_validation_arr; i++) { printf("%d, ", val_indices[i]); }
	printf("\n");*/

	for (int i = 0; i < max(1, number_train_batches - number_networks + 1); i++) {
		float* tmp_res_cost_train = backwardPhase(number_train_batches * max_batch_size, max_batch_size, 0, train_indices + i, early_counters);
		backprop_function(this, lrate, params, nparams); //applyVGradSGD(lrate);
		for (int j = 0; j < number_networks; j++) { cost_train[j] += tmp_res_cost_train[j] * max_batch_size / (float)max_train_number_examples; }
		delete tmp_res_cost_train;
	}
	if (number_remainder_train_examples > 0) {
		//printf("\nResto de entrenamiento en: %d\n", max_train_number_examples - number_remainder_train_examples);
		float* tmp_res_cost_train = backwardPhase(number_remainder_train_examples, max_train_number_examples - number_remainder_train_examples, early_counters);
		backprop_function(this, lrate, params, nparams);
		for (int j = 0; j < number_networks; j++) { cost_train[j] += tmp_res_cost_train[j] * number_remainder_train_examples / (float)max_train_number_examples; }
		delete tmp_res_cost_train;
	}

	for (int i = 0; i < max(1, number_validation_batches - number_networks + 1); i++) {
		float* tmp_res_cost_val = validationGetCostFunctionAndCalculateLossFunction(number_validation_batches * max_batch_size, max_batch_size, 0, val_indices + i);
		for (int j = 0; j < number_networks; j++) { cost_val[j] += tmp_res_cost_val[j] * max_batch_size / (float)max_validation_number_examples; }
		delete tmp_res_cost_val;
	}
	if (number_remainder_validation_examples > 0) {
		float* tmp_res_cost_val = validationGetCostFunctionAndCalculateLossFunction(number_remainder_validation_examples, max_validation_number_examples - number_remainder_validation_examples);
		for (int j = 0; j < number_networks; j++) { cost_val[j] += tmp_res_cost_val[j] * number_remainder_validation_examples / (float)max_validation_number_examples; }
		delete tmp_res_cost_val;
	}

}

void Network::trainAllExamplesMaxBatch(func_lrate function_learning_rate, int nparams, float* params, func_backprop backprop_function, int nepochs, int show_per_epoch, float convergence, float min_err_start_early_stop, int count_early_stop) {

	int number_train_batches = max_train_number_examples / max_batch_size;
	int number_remainder_train_examples = max_train_number_examples % max_batch_size;
	int repeat_train_arr = (int) ceil( (number_train_batches + number_networks - 1) / (float)number_train_batches);

	int number_validation_batches = max_validation_number_examples / max_batch_size;
	int number_remainder_validation_examples = max_validation_number_examples % max_batch_size;
	int repeat_validation_arr = (int) ceil( (number_validation_batches + number_networks - 1) / (float)number_validation_batches);

	/*printf("\nnumber_train_batches: %d", number_train_batches);
	printf("\nnumber_remainder_train_examples: %d", number_remainder_train_examples);
	printf("\nrepeat_train_arr: %d", repeat_train_arr);

	printf("\nnumber_validation_batches: %d", number_validation_batches);
	printf("\nnumber_remainder_validation_examples: %d", number_remainder_validation_examples);
	printf("\nrepeat_validation_arr: %d\n", repeat_validation_arr);*/

	int* train_indices = new int[number_train_batches * repeat_train_arr];
	int* val_indices = new int[number_validation_batches * repeat_validation_arr];

	float* prev_cost_train = new float[number_networks];
	float* prev_cost_val = new float[number_networks];

	float* cost_train = new float[number_networks];
	float* cost_val = new float[number_networks];

	int number_no_stopped_networks = number_networks;
	int* early_stop_counters = new int[number_networks];
	for (int i = 0; i < number_networks; i++) { early_stop_counters[i] = count_early_stop; }

	for (int i = 0; i < nepochs; i++) {
		memset(cost_train, 0, number_networks * sizeof(float));
		memset(cost_val, 0, number_networks * sizeof(float));
		float lrate = function_learning_rate(i);
		epochAllExamples(lrate, params, nparams, backprop_function, number_train_batches, number_remainder_train_examples, repeat_train_arr, number_validation_batches, number_remainder_validation_examples, repeat_validation_arr, train_indices, val_indices, cost_train, cost_val, early_stop_counters);

		bool all_zero = true;
		for (int j = 0; j < number_networks; j++) {
			if (cost_val[j] > prev_cost_val[j] && prev_cost_val[j] <= min_err_start_early_stop) { early_stop_counters[j]--; }
			if (early_stop_counters[j] > 0) { all_zero = false; }
		}

		/*printf("\n");
		for (int j = 0; j < number_networks; j++) { printf("%d, ", early_stop_counters[j]); }
		printf("\n");*/

		if (all_zero) {
			printf("\n\nTrain stopped because of early stopping (all networks)!");
			printf("\nEPOCH %d:", i + 1);
			for (int j = 0; j < number_networks; j++) {
				printf("\n\tRED %j-> err train: %.20f  err test: %.20f", cost_train[j], cost_val[j]);
			}
			printf("\n");
			break;
		}

		if (i == 0 || (i + 1) % show_per_epoch == 0) {
			printf("\nEPOCH %d:", i + 1);
			for (int j = 0; j < number_networks; j++) {
				printf("\n\tRED %j-> err train: %.20f  err test: %.20f", cost_train[j], cost_val[j]);
			}
			printf("\n");
		}
		memcpy(prev_cost_train, cost_train, number_networks * sizeof(float));
		memcpy(prev_cost_val, cost_val, number_networks * sizeof(float));
	}

	delete train_indices;
	delete val_indices;
	delete prev_cost_train;
	delete prev_cost_val;
	delete cost_train;
	delete cost_val;

}

void Network::finalizeForwardBackward() {
	cublasSetStream_v2(handle, 0);
	cudaStreamDestroy(stream_principal);
	cudaStreamDestroy(stream_transferencia_output);

	if (input_train != NULL) { delete input_train; input_train = NULL; }
	if (output_train != NULL) { delete output_train; output_train = NULL; }

	if (input_validation != NULL) { delete input_validation; input_validation = NULL; }
	if (output_validation != NULL) { delete output_validation; output_validation = NULL; }

	if (d_input_train_pointers != NULL) { cudaFree(d_input_train_pointers); d_input_train_pointers = NULL; }
	if (d_input_validation_pointers != NULL) { cudaFree(d_input_validation_pointers); d_input_validation_pointers = NULL; }

	for (int i = 0; i < number_layers; i++) {
		layers[i]->setNumberInputExamples(0);
		layers[i]->setAuxiliarExpandReduceMatrix(NULL);
		layers[i]->freeForwardMemory();
		layers[i]->freeBackwardMemory();
		layers[i]->setIsTraining(false);
	}
	cudaFree(d_auxiliar_expand_reduce_matrix);

	cudaFree(d_output_forward_multiple_nn_sum);
	if (d_output_forward_multiple_nn_sum_pointers != NULL) { cudaFree(d_output_forward_multiple_nn_sum_pointers);  d_output_forward_multiple_nn_sum_pointers = NULL; }

	if (d_auxiliar_matrix_loss_function_error_backprop != NULL) { cudaFree(d_auxiliar_matrix_loss_function_error_backprop); d_auxiliar_matrix_loss_function_error_backprop = NULL; }
	if (d_auxiliar2_matrix_loss_function_error_backprop != NULL) { cudaFree(d_auxiliar2_matrix_loss_function_error_backprop); d_auxiliar2_matrix_loss_function_error_backprop = NULL; }

	cudaDeviceSynchronize();

	max_batch_size = 0;
	max_train_number_examples = 0;
	max_validation_number_examples = 0;
	max_layer_size = 0;
}

void Network::storeNetworkInFile(char* name) {
	
	unsigned long long num_elems = 0;
	for (int i = 0; i < number_layers; i++) { num_elems += layers[i]->getTotalElementsBiasVectors() + layers[i]->getTotalElementsWeightMatrices(); }

	unsigned long long offset = 0;
	float* data = new float[num_elems];

	for (int i = 0; i < number_layers; i++) {
		layers[i]->storeBiasVectorsWeightMatrices(data + layers[i]->getTotalElementsBiasVectors() + offset, data + offset);
		offset += layers[i]->getTotalElementsBiasVectors() + layers[i]->getTotalElementsWeightMatrices();
	}

	char* buffer = (char*) data;
	crearArchivoEscribirYCerrar(name, num_elems * sizeof(float), buffer);

	delete data;
}

void Network::loadNetworkFromFile(char* name) {
	
	unsigned int nbytes = 0;

	char* cargar = leerArchivoYCerrar(name, &nbytes);

	unsigned int nnumeros = nbytes / 4;

	float* data = (float*)cargar;

	unsigned long long num_elems = 0;
	for (int i = 0; i < number_layers; i++) { num_elems += layers[i]->getTotalElementsBiasVectors() + layers[i]->getTotalElementsWeightMatrices(); }

	if (num_elems != nnumeros) { printf("\nError loading Network: not same number of elements (total numbers of biases and weights) %d %d\n", num_elems, nnumeros); }

	unsigned long long offset = 0;
	for (int i = 0; i < number_layers; i++) {
		layers[i]->copyWeightBias(data + layers[i]->getTotalElementsBiasVectors() + offset, data + offset);
		offset += layers[i]->getTotalElementsBiasVectors() + layers[i]->getTotalElementsWeightMatrices();
	}

}