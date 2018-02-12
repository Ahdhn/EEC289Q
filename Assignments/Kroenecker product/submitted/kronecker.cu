#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdint>
double maxBW;

void KroneckerGPUSmall(int M, int N, float* A, float* B, float* C);
void KroneckerGPU(int M, int N, float* A, float* B, float* C);

//Error handling micro, wrap it around function whenever possible
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void KroneckerCPU(int M, int N, float* A, float* B, float*C) {

	for (int rowA = 0; rowA < M; rowA++) {
		for (int colA = 0; colA < M; colA++) {
			float elemA = A[rowA * M + colA];

			for (int rowB = 0; rowB < N; rowB++) {
				int rowC = rowA * N + rowB;

				for (int colB = 0; colB < N; colB++) {
					int colC = colA * N + colB;
					float elemB = B[rowB * N + colB];

					C[rowC * (M * N) + colC] = elemA * elemB;
				}
			}
		}
	}
}


void PrintMatrix(float* matrix, int M, int N) {
	for (int row = 0; row<M; row++)
	{
		for (int columns = 0; columns<N; columns++)
		{
			printf("%7.3f ", matrix[row * N + columns]);
		}
		printf("\n");
	}
}

void checkResults(int N, int M, float* gpu_result, float* cpu_result){

	for (int i = 0; i < M * N * M * N; i++) {
		if (fabs(gpu_result[i] - cpu_result[i]) > 0.01) {
			printf("\n Mismatch at index %d: GPU: %f CPU %f\n", i, gpu_result[i], cpu_result[i]);
			exit(EXIT_FAILURE);
		}
	}

	printf("\n Ok!!!\n");
}

/***************************************************/
/***************************************************/
/***************************************************/
__global__ void KroneckerKernel(const int M, const int N, float* A, float* B, float* C){

	extern __shared__ float shrd_ptr[];
	float*sh_B = shrd_ptr;
	__shared__ float shrd_C[1024];
	__shared__ uint32_t shrd_C_id[1024];

	uint32_t tid = threadIdx.x;
	uint32_t sh_b_size = (N*N <= 64*64) ? N*N: 64*64;
	int b_jump = -sh_b_size;


	uint32_t numJumps = max(1,((M*M) / (blockDim.x * gridDim.x))); // at least do one run


	for (uint32_t jump_id = 0; jump_id < numJumps; jump_id++){ //how many different groups of elements a block will touch
		                                                        //groups of elements are elemnts next to one another in A


		uint32_t A_id = blockIdx.x * blockDim.x + tid + jump_id*gridDim.x*blockDim.x; //the element A id

		if(A_id < M*M){

			uint32_t A_i = A_id / M;
			uint32_t A_j = A_id % M;
			float A_ele = A[A_id];

			int id =0;

#pragma unroll
			for (uint32_t B_id = 0; B_id < N*N; B_id++){

				if(B_id%sh_b_size == 0 ){
					__syncthreads();
					b_jump += sh_b_size;
					for (uint32_t i = tid; i < sh_b_size; i += blockDim.x){
						sh_B[i] = B[i+b_jump];
					}
					__syncthreads();
				}


				float B_ele;
				B_ele = sh_B[B_id-b_jump];


				uint32_t B_i = B_id / N;
				uint32_t B_j = B_id % N;
				uint32_t C_i = A_i*N + B_i;
				uint32_t C_j = A_j*N + B_j;
				uint32_t C_id = C_i*N*M + C_j;
				//C[C_id] = A_ele*B_ele;

				shrd_C[tid*blockDim.x + id] = A_ele*B_ele;
				shrd_C_id[tid*blockDim.x + id] = C_id;
				id++;

				if(id == 32){
					__syncthreads();
#pragma unroll
					for (int i=0; i<1024; i+=32){
						C[shrd_C_id[tid+i]] = shrd_C[tid+i];
					}
					id = 0;
					__syncthreads();
				}







			}

		}
	}

}

/***************************************************/
/***************************************************/
/***************************************************/
__global__ void KroneckerKernelSmall(const int M, const int N, float* A, float* B, float* C){

	//Each thread will read it multiple entry in A, do N multiply and then jump
	// N*M (in C array) to do the same N mutliple but with different values in B. Keep jumpping
	//N times.
	//The above will be packed inside a loop that iterate number of times equal to N*M/numWarps

	//**** Move B to shared memory ***//
	extern __shared__ float shrd_ptr[];
	float*sh_B = shrd_ptr;
	__shared__ float shrd_C[1024];
	__shared__ uint32_t shrd_C_id[1024];

	uint32_t tid = threadIdx.x;

	for (uint32_t i = tid; i < N*N; i += blockDim.x){
		sh_B[i] = B[i];
	}
	__syncthreads();

	//the indcies
	uint32_t numJumps = max(1,((M*M) / (blockDim.x * gridDim.x))); // at least do one run

	for (uint32_t jump_id = 0; jump_id < numJumps; jump_id++){ //how many different groups of elements a block will touch
		                                                        //groups of elements are elemnts next to one another in A


		uint32_t A_id = blockIdx.x * blockDim.x + tid + jump_id*gridDim.x*blockDim.x; //the element A id

		if(A_id < M*M){

			uint32_t A_i = A_id / M;
			uint32_t A_j = A_id % M;
			float A_ele = A[A_id];

			int id =0;
#pragma unroll
			for (uint32_t B_id = 0; B_id < N*N; B_id++){
				float B_ele = sh_B[B_id];

				uint32_t B_i = B_id / N;
				uint32_t B_j = B_id % N;
				uint32_t C_i = A_i*N + B_i;
				uint32_t C_j = A_j*N + B_j;
				uint32_t C_id = C_i*N*M + C_j;
				//C[C_id] = A_ele*B_ele;

				shrd_C[tid*blockDim.x + id] = A_ele*B_ele;
				shrd_C_id[tid*blockDim.x + id] = C_id;
				id++;

				if(id == 32){
					__syncthreads();
#pragma unroll
					for (int i=0; i<1024; i+=32){
						C[shrd_C_id[tid+i]] = shrd_C[tid+i];
						//C[tid + i] = shrd_C[ tid + i ];

					}
					id = 0;
					__syncthreads();
				}
			}

		}
	}
}

void KroneckerGPUSmall(int M, int N, float* A, float* B, float*C){

	//The assumtion here is that B is at most 16kB
	//Prefer more L1-cache because we at max need 16kB shared memory
	//cudaFuncSetCacheConfig(KroneckerKernelSmall, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(KroneckerKernelSmall, cudaFuncCachePreferShared);

	float *d_A, *d_B, *d_C;
	const int M2 = M*M;
	const int N2 = N*N;

	//****allocate memory on device***//
	HANDLE_ERROR(cudaMallocManaged(&d_A, M2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_B, N2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_C, M2*N2*sizeof(float)));

//#define my_debug

	//****move data to device***//
	for (int i = 0; i<N2; i++){ d_B[i] = B[i]; }
	for (int i = 0; i<M2; i++){ d_A[i] = A[i]; }


#ifdef my_debug


	const int shrdMem = N2;//amount of shared memory is the size of B (we move it all in shared memory)

	float* cpu_result = (float*)malloc(sizeof(float) * M * N * M * N);
	KroneckerCPU(M, N, A, B, cpu_result);

	HANDLE_ERROR(cudaProfilerStart());
	KroneckerKernelSmall <<< 512, 32, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
	HANDLE_ERROR(cudaProfilerStop());
	HANDLE_ERROR(cudaDeviceSynchronize());
	exit(0);

	const int numIter = 100;
	const float mem = float((sizeof(float)*(N2 + M2 + M2*N2)) / (1024.0f * 1024.0f)); //in mega bytes

	int numBlocks=0;
	for (int bb = 2; bb <=2048; bb += 2){
		int mod = M2 % (bb*32);
		if(mod != 0 ){continue;}
		numBlocks = bb;
	}
  numBlocks = 1024;
	int numThreads = 32;

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));//timing
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	for (int i = 0; i<numIter; i++){
		KroneckerKernelSmall <<< numBlocks, numThreads, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /= 1000;//time in seconds
	float myBW = mem / time;
	float gflops = (float(M2)*float(N2))/(1024.0f*1024.0f*1024.0f);
	std::cout << "\n M= "<<M << " N= "<< N <<" numThreads= " << numThreads << " numBlocks= " << numBlocks << " time(sec)= " << time << " BW(MB/s)= " << myBW
	<< " BW_prec= " << 100*(myBW / maxBW)<<" GFLOPs/Sec= "<< gflops/time  << std::endl;

#else
	const int shrdMem = N2;//amount of shared memory is the size of B (we move it all in shared memory)

	int numBlocks = 1024;
	int numThreads = 32;

	KroneckerKernelSmall <<< numBlocks, numThreads, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
	HANDLE_ERROR(cudaDeviceSynchronize());

#endif

	//****move data to host***//
	for (int i = 0; i<N2*M2; i++){ C[i] = d_C[i]; }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	HANDLE_ERROR(cudaDeviceReset());
}



/*
int main(int argc, char* argv[]) {

	if (argc != 3){
		std::cout << "\n	Usage ./kron [M][N]" << std::endl;
		std::cout << "	M and N is the size of matrix A and B respectively." << std::endl;
		std::cout << "	M and N should be power of 2" << std::endl;
		exit(EXIT_FAILURE);
	}

	const int M = atoi(argv[1]);
	const int N = atoi(argv[2]);

	if (ceil(log2(float(M))) != floor(log2(float(M))) || ceil(log2(float(N))) != floor(log2(float(N)))){
		std::cout << "\n	M and N should be power of 2" << std::endl;
		exit(EXIT_FAILURE);
	}

	//const int M = 32; //A size
	//const int N = 32; //B size

	///////////////////////////////////////////////////////////////////////////////////////////////
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		printf("\n deviceCount is zero. I quit!!!");
		exit(EXIT_FAILURE);
	}
	const int dev = (deviceCount == 1) ? 0 : 3;
	cudaSetDevice(dev);
	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, dev));
	printf("\n  Total number of device: %d", deviceCount);
	printf("\n  Using device Number: %d", dev);
	printf("\n  Device name: %s", devProp.name);
	printf("\n  devProp.major: %d", devProp.major);
	printf("\n  devProp.minor: %d", devProp.minor);
	printf("\n  Total global memory (GB): %zu", devProp.totalGlobalMem>>30 );

	if (devProp.major == 1){//Fermi
		if (devProp.minor == 1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount * 48);
		}
		else{
			printf("\n  SM Count: %d", devProp.multiProcessorCount * 32);
		}
	}
	else if (devProp.major == 3){//Kepler
		printf("\n  SM Count: %d", devProp.multiProcessorCount * 192);
	}
	else if (devProp.major == 5){//Maxwell
		printf("\n  SM Count: %d", devProp.multiProcessorCount * 128);
	}
	else if (devProp.major == 6){//Pascal
		if (devProp.minor == 1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount * 128);
		}
		else if (devProp.minor == 0){
			printf("\n  SM Count: %d", devProp.multiProcessorCount * 64);
		}
	}
	printf("\n  Compute Capability: v%d.%d", (int)devProp.major, (int)devProp.minor);
	printf("\n  Memory Clock Rate: %d(kHz)", devProp.memoryClockRate);
	printf("\n  Memory Bus Width: %d(bits)", devProp.memoryBusWidth);
	maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth / 8.0) / 1.0E3;
	//printf("\n  Peak Memory Bandwidth: %f(MB/s)\n", maxBW);
	///////////////////////////////////////////////////////////////////////////////////////////////



	float* A = (float*)malloc(sizeof(float) * M * M);
	float* B = (float*)malloc(sizeof(float) * N * N);
	float* cpu_result = (float*)malloc(sizeof(float) * M * N * M * N);
	float* gpu_result = (float*)malloc(sizeof(float) * M * N * M * N);

	for (int i = 0; i < M * M; i++){
		A[i] = float(i + 1);
	}

	for (int i = 0; i < N * N; i++){
		B[i] = float(i + 1);
	}

	KroneckerGPU(M, N, A, B, gpu_result);
	KroneckerCPU(M, N, A, B, cpu_result);

	checkResults(N, M, gpu_result, cpu_result);

	printf("\nDone %f %f\n", gpu_result[M * N * M * N - 1], cpu_result[M * N * M * N - 1]);


	free(A);
	free(B);
	free(cpu_result);
	free(gpu_result);

	return 0;
}*/

//****************************************************************************//
void KroneckerGPU(int M, int N, float* A, float* B, float* C) {

	//The assumtion here is that B is at most 16kB
	//Prefer more L1-cache because we at max need 16kB shared memory
	//cudaFuncSetCacheConfig(KroneckerKernelSmall, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(KroneckerKernelSmall, cudaFuncCachePreferShared);

	float *d_A, *d_B, *d_C;
	const int M2 = M*M;
	const int N2 = N*N;

	//****allocate memory on device***//
	HANDLE_ERROR(cudaMallocManaged(&d_A, M2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_B, N2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_C, M2*N2*sizeof(float)));

//#define my_debug

	//****move data to device***//
	for (int i = 0; i<N2; i++){ d_B[i] = B[i]; }
	for (int i = 0; i<M2; i++){ d_A[i] = A[i]; }


#ifdef my_debug


	const int shrdMem = (N*N <= 64*64) ? N*N: 64*64;//amount of shared memory is the size of B (we move it all in shared memory)


	const int numIter = 1;
	const float mem = float((sizeof(float)*(N2 + M2 + M2*N2)) / (1024.0f * 1024.0f)); //in mega bytes


	int numBlocks = 1024;
	int numThreads = 32;

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));//timing
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	for (int i = 0; i<numIter; i++){
		KroneckerKernel <<< numBlocks, numThreads, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /= 1000;//time in seconds
	float myBW = mem / time;
	float gflops = (float(M2)*float(N2))/(1024.0f*1024.0f*1024.0f);
	std::cout << "\n M= "<<M << " N= "<< N <<" numThreads= " << numThreads << " numBlocks= " << numBlocks << " time(sec)= " << time << " BW(MB/s)= " << myBW
	<< " BW_prec= " << 100*(myBW / maxBW)<<" GFLOPs/Sec= "<< gflops/time  << std::endl;

#else
	const int shrdMem = (N*N <= 64*64) ? N*N: 64*64;//amount of shared memory is the size of B (we move it all in shared memory)

	int numBlocks = 2048;
	int numThreads = 32;

	KroneckerKernel <<< numBlocks, numThreads, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
	HANDLE_ERROR(cudaDeviceSynchronize());

#endif

	//****move data to host***//
	for (int i = 0; i<N2*M2; i++){ C[i] = d_C[i]; }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	HANDLE_ERROR(cudaDeviceReset());
}
