#include "HostPinnedDeviceMatrix.cuh"

HostPinnedDeviceMatrix::HostPinnedDeviceMatrix(int nr, int nc, int mo, unsigned int flags) {
	number_rows = nr;
	number_cols = nc;
	total_accesible_elements = number_rows * number_cols;
	number_elements_multiple_of = mo;
	total_elements = nextMultiple(total_accesible_elements, number_elements_multiple_of);

	manageCUDAError(cudaHostAlloc(&host_pinned_data, total_elements * sizeof(float), flags), "HostPinnedDeviceMatrix constr: 1");
	manageCUDAError(cudaMalloc(&device_data, total_elements * sizeof(float)), "HostPinnedDeviceMatrix constr: 2");

	cudaDeviceSynchronize();
}

HostPinnedDeviceMatrix::~HostPinnedDeviceMatrix() {
	number_rows = 0;
	number_cols = 0;
	total_accesible_elements = 0;
	number_elements_multiple_of = 1;
	total_elements = 0;

	manageCUDAError(cudaFreeHost(host_pinned_data), "HostPinnedDeviceMatrix destr: 1");
	manageCUDAError(cudaFree(device_data), "HostPinnedDeviceMatrix destr: 2");

	cudaDeviceSynchronize();
}

int HostPinnedDeviceMatrix::getNumberRows() {
	return number_rows;
}

int HostPinnedDeviceMatrix::getNumberCols() {
	return number_cols;
}

float* HostPinnedDeviceMatrix::getDevicePointer() {
	return device_data;
}

void HostPinnedDeviceMatrix::copyHostToDevice(float* h_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(host_pinned_data, h_ptr, total_accesible_elements * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyHostToDevice: 1");
	manageCUDAError(cudaMemcpyAsync(device_data, host_pinned_data, total_accesible_elements * sizeof(float), cudaMemcpyHostToDevice, stream), "HostPinnedDeviceMatrix copyHostToDevice: 2");
}

void HostPinnedDeviceMatrix::copyHostToDevice(float* h_ptr, int nrows, cudaStream_t stream) {
	if (nrows > number_rows) {
		printf("\n\nError host HostPinnedDeviceMatrix copyHostToDevice(2): nrows argument bigger than number_rows\n\n");
		exit(EXIT_FAILURE);
	}
	manageCUDAError(cudaMemcpyAsync(host_pinned_data, h_ptr, total_accesible_elements * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyHostToDevice(2): 1");
	manageCUDAError(cudaMemcpyAsync(device_data, host_pinned_data, nrows * number_cols * sizeof(float), cudaMemcpyHostToDevice, stream), "HostPinnedDeviceMatrix copyHostToDevice(2): 2");
}

void HostPinnedDeviceMatrix::copyDeviceToHost(float* h_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(host_pinned_data, device_data, total_accesible_elements * sizeof(float), cudaMemcpyDeviceToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost: 1");
	manageCUDAError(cudaMemcpyAsync(h_ptr, host_pinned_data, total_accesible_elements * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost: 2");
}

void HostPinnedDeviceMatrix::copyDeviceToHost(float* h_ptr, int nrows, cudaStream_t stream) {
	if (nrows > number_rows) {
		printf("\n\nError host HostPinnedDeviceMatrix copyDeviceToHost(2): nrows argument bigger than number_rows\n\n");
		exit(EXIT_FAILURE);
	}
	manageCUDAError(cudaMemcpyAsync(host_pinned_data, device_data, total_accesible_elements * sizeof(float), cudaMemcpyDeviceToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost(2): 1");
	manageCUDAError(cudaMemcpyAsync(h_ptr, host_pinned_data, nrows * number_cols * sizeof(float), cudaMemcpyHostToHost, stream), "HostPinnedDeviceMatrix copyDeviceToHost(2): 2");
}

void HostPinnedDeviceMatrix::copyFromDevice(float* d_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(device_data, d_ptr, total_accesible_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream), "HostPinnedDeviceMatrix copyDeviceToDevice: 1");
}

void HostPinnedDeviceMatrix::copyFromDevice(float* d_ptr, int nrows, cudaStream_t stream) {
	if (nrows > number_rows) {
		printf("\n\nError host HostPinnedDeviceMatrix copyFromDevice(2): nrows argument bigger than number_rows\n\n");
		exit(EXIT_FAILURE);
	}
	manageCUDAError(cudaMemcpyAsync(device_data, d_ptr, nrows * number_cols * sizeof(float), cudaMemcpyDeviceToDevice, stream), "HostPinnedDeviceMatrix copyDeviceToDevice(2): 1");
}

void HostPinnedDeviceMatrix::copyToDevice(float* d_ptr, cudaStream_t stream) {
	manageCUDAError(cudaMemcpyAsync(d_ptr, device_data, total_accesible_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream), "HostPinnedDeviceMatrix copyDeviceToDevice: 1");
}

void HostPinnedDeviceMatrix::copyToDevice(float* d_ptr, int nrows, cudaStream_t stream) {
	if (nrows > number_rows) {
		printf("\n\nError host HostPinnedDeviceMatrix copyToDevice(2): nrows argument bigger than number_rows\n\n");
		exit(EXIT_FAILURE);
	}
	manageCUDAError(cudaMemcpyAsync(d_ptr, device_data, nrows * number_cols * sizeof(float), cudaMemcpyDeviceToDevice, stream), "HostPinnedDeviceMatrix copyDeviceToDevice(2): 1");
}

float** HostPinnedDeviceMatrix::generateDeviceRowsPointers(int row_offset, int num_rows, int* row_ids) {
	for (int i = 0; i < num_rows; i++) {
		row_ids[i] += row_offset;
		//printf("\nrow_ids[%d] = %d", i, row_ids[i]);
		if (row_ids[i] >= number_rows) {
			printf("\n\nError host HostPinnedDeviceMatrix generateDeviceRowsPointers: invalid row_id (in %d, %d >= %d)\n\n", i, row_ids[i], number_rows);
			exit(EXIT_FAILURE);
		}
	}
	float** hres = new float*[num_rows];
	for (int i = 0; i < num_rows; i++) { hres[i] = device_data + (row_ids[i] * number_cols); }

	float** res = 0;
	manageCUDAError(cudaMalloc(&res, num_rows * sizeof(float*)), "HostPinnedDeviceMatrix generateDeviceRowsPointers: 1");
	manageCUDAError(cudaMemcpy(res, hres, num_rows * sizeof(float*), cudaMemcpyHostToDevice), "HostPinnedDeviceMatrix generateDeviceRowsPointers: 2");
	delete hres;

	return res;
}

void HostPinnedDeviceMatrix::showDeviceMatrix(char* msg, cudaStream_t stream) {
	float* mtr = new float[total_accesible_elements];
	copyDeviceToHost(mtr, stream);
	cudaStreamSynchronize(stream);
	imprimirMatrizPorPantalla(msg, mtr, number_rows, number_cols);
	delete mtr;
}