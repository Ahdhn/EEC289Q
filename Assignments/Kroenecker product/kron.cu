#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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



__global__ void KroneckerKernelSmall(const int M, const int N, float* A, float* B, float* C){

	//Each thread will read it multiple entry in A, do N multiply and then jump 
	// N*M (in C array) to do the same N mutliple but with different values in B. Keep jumpping
	//N times. 	
	//The above will be packed inside a loop that iterate number of times equal to N*M/numWarps

	//**** Move B to shared memory ***//
	extern __shared__ float shrd_ptr[];
	float*sh_B = shrd_ptr; 

	//uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x
	uint32_t tid = threadIdx.x;
	//uint32_t lane = threadIdx.x & 0x1F; //lane of the thread in the warp 

	for (uint32_t i = tid; i < N*N; i += blockDim.x){
		//move data to shared memory
		sh_B[i] = B[i];
	}
	__syncthreads(); //sync before moving on

	
	//the indcies 
	
	uint32_t id_C;
	
	uint32_t numJumps = max(1,((M*M) / (blockDim.x * gridDim.x)));

	//if (blockIdx.x == 0 && threadIdx.x == 0){
	//	printf("\n numJumps=%d", numJumps);
	//}
	
	for (uint32_t jump_id = 0; jump_id < numJumps; jump_id++){ //how many different groups of elements a block will touch
		                                                        //groups of elements are elemnts next to one another in A
		

		uint32_t A_id = blockIdx.x * blockDim.x + tid + jump_id*gridDim.x*blockDim.x; //the element A id
		uint32_t A_i = A_id / M; 
		uint32_t A_j = A_id % M;

		float A_ele = A[A_id];

#pragma unroll 
		//operate on A_ele (do all its products)
		for (uint32_t B_id = 0; B_id < N*N; B_id++){
			float B_ele = sh_B[B_id];
			
			uint32_t B_i = B_id / N; 
			uint32_t B_j = B_id % N;

			//get the position in C 
			uint32_t C_i = A_i*N + B_i;
			uint32_t C_j = A_j*N + B_j;
			uint32_t C_id = C_i*N*M + C_j;
			
			//store the product
			C[C_id] = A_ele*B_ele;

			//Debug 
			//if ( blockIdx.x == 0){
			//	printf("\n\n tid= %d, bkId= %d, j_id= %d, A_id= %d [%f] (%d, %d), B_id= %d [%f] (%d, %d), C_id= %d [%f] (%d, %d)",
			//		         tid,   blockIdx.x, jump_id,  A_id, A_ele, A_i, A_j,  B_id, B_ele, B_i, B_j, C_id, C[C_id], C_i, C_j);
			//}
		}
	}
}

__global__ void KroneckerKernel(const int M, const int N, float* A, float* B, float* C){


}

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
			system("pause");
			
		}
	}

	printf("\n Ok!!!");
}

void KroneckerGPUSmall(int M, int N, float* A, float* B, float*C){

	//The assumtion here is that B is at most 16kB 

	//Prefer more L1-cache because we at max need 16kB shared memory
	cudaFuncSetCacheConfig(KroneckerKernelSmall, cudaFuncCachePreferL1);

	float *d_A, *d_B, *d_C;
	const int M2 = M*M;
	const int N2 = N*N;

	//****allocate memory on device***//
	HANDLE_ERROR(cudaMallocManaged(&d_A, M2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_B, N2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_C, M2*N2*sizeof(float)));

#define my_debug	

	//****move data to device***//
	for (int i = 0; i<N2; i++){ d_B[i] = B[i]; }
	for (int i = 0; i<M2; i++){ d_A[i] = A[i]; }


	//****Calc gird and block size***//
	//Each thread will read it multiple entry in A, do N multiply and then jump 
	// N*M (in C array) to do the same N mutliple but with different values in B. Keep jumpping
	//N times. 	
	//The above will be packed inside a loop that iterate number of times equal to N*M/numWarps

	const int warpSize = 32; //Warp size (better to query it from device prop)
	const int numWarps = 16; //TODO play with this (it should be M/warpSize)	                            
	const int numBlocks = M2;//number of block is such that each block have a copy of B in shared
	                         // memory, thus the number of block is same as size of A
	const int numThreads = warpSize * numWarps;
	const int shrdMem = N2;//amount of shared memory is the size of B (we move it all in shared memory)

	float* cpu_result = (float*)malloc(sizeof(float) * M * N * M * N);
	KroneckerCPU(M, N, A, B, cpu_result);

	
	KroneckerKernelSmall << < 512, 64, shrdMem*sizeof(float) >> >(M, N, d_A, d_B, d_C);
	HANDLE_ERROR(cudaDeviceSynchronize());


	for (int t = 2; t < 1024; t *= 2){
		for (int b = 2; b < 1024; b *= 2){
			printf("\n t=%d, b=%d", t, b);
			KroneckerKernelSmall <<< b, t, shrdMem*sizeof(float) >>>(M, N, d_A, d_B, d_C);
			HANDLE_ERROR(cudaDeviceSynchronize());

			checkResults(N, M, d_C, cpu_result);
			for (int l = 0; l < M2*N2; l++){ d_C[l] = 0; }
		}
	}


//#ifdef my_debug
	//const int numIter = 1;
	//const float mem = float((sizeof(float)*(N2 + M2 + M2*N2)) / (1024.0f * 1024.0f)); //in mega bytes
	//cudaEvent_t start, stop;
	//HANDLE_ERROR(cudaEventCreate(&start));//timing
	//HANDLE_ERROR(cudaEventCreate(&stop));
	//HANDLE_ERROR(cudaEventRecord(start, 0));
	//for (int i = 0; i<numIter; i++){
//#endif	
		//KroneckerKernelSmall <<< numBlocks, numThreads, shrdMem*sizeof(float) >>>(M, N, A, B, C);
		//KroneckerKernelSmall << < 1024, M2/2, shrdMem*sizeof(float) >> >(M, N, d_A, d_B, d_C);
//#ifdef my_debug
	//}
//#endif 
	//HANDLE_ERROR(cudaDeviceSynchronize());
//#ifdef my_debug
	//HANDLE_ERROR(cudaEventRecord(stop, 0));
	//HANDLE_ERROR(cudaEventSynchronize(stop));
	//float time = 0.0f;
	//HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	//time /= numIter; //time in mSec
	//time /= 1000;//time in seconds
	//float myBW = mem / time;
	//std::cout << "\n  numThreads= " << numThreads << " numBlocks= " << numBlocks << " time(sec)= " << time << "  BW(MB)=" << myBW << "   BW_frac(MB/sec)= " << myBW / maxBW << std::endl;
//#endif

	//****move data to host***//	
	for (int i = 0; i<N2*M2; i++){ C[i] = d_C[i]; }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	HANDLE_ERROR(cudaDeviceReset());
}

void KroneckerGPU(int M, int N, float* A, float* B, float* C) {

	float *d_A, *d_B, *d_C;
	const int M2 = M*M;
	const int N2 = N*N;
	//allocate memory on device
	HANDLE_ERROR(cudaMallocManaged(&d_A, M2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_B, N2*sizeof(float)));
	HANDLE_ERROR(cudaMallocManaged(&d_C, M2*N2*sizeof(float)));

#define my_debug

	//move data to device
	for (int i = 0; i<N2; i++){ d_B[i] = B[i]; }
	for (int i = 0; i<M2; i++){ d_A[i] = A[i]; }


	//calc gird and block size 
	const int numBlocks = 4096;
	const int numThreads = 256;


#ifdef my_debug
	const int numIter = 1000;
	const float mem = float((sizeof(float)*(N2 + M2 + M2*N2)) / (1024.0f * 1024.0f)); //in mega bytes
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));//timing
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, NULL));
	for (int i = 0; i<numIter; i++){
#endif
		KroneckerKernelSmall << < numBlocks, numThreads >> >(M, N, A, B, C);
#ifdef my_debug
	}
#endif 

	cudaDeviceSynchronize();

#ifdef my_debug
	HANDLE_ERROR(cudaEventRecord(stop, NULL));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /= 1000;//time in seconds
	float myBW = mem / time;
	std::cout << "\n  numThreads= " << numThreads << " numBlocks= " << numBlocks << " time(sec)= " << time << "  BW(MB)=" << myBW << "   BW_frac(MB/sec)= " << myBW / maxBW << std::endl;
#endif

	//move data to host
	for (int i = 0; i<N2*M2; i++){ C[i] = d_C[i]; }

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}


int main(int argc, char* argv[]) {

	std::cout << (1<<30) << std::endl;
	/*if (argc != 3){
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
	}*/

	const int M = 16;
	const int N = 4;

	///////////////////////////////////////////////////////////////////////////////////////////////
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		printf("\n deviceCount is zero. I quit!!!");
		exit(EXIT_FAILURE);
	}
	const int dev = (deviceCount == 1) ? 0 : 0;
	cudaSetDevice(dev);
	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, dev));
	printf("\n  Total number of device: %d", deviceCount);
	printf("\n  Using device Number: %d", dev);
	printf("\n  Device name: %s", devProp.name);
	printf("\n  devProp.major: %d", devProp.major);
	printf("\n  devProp.minor: %d", devProp.minor);
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
	double maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth / 8.0) / 1.0E3;
	printf("\n  Peak Memory Bandwidth: %f(MB/s)\n", maxBW);
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

	KroneckerGPUSmall(M, N, A, B, gpu_result);
	KroneckerCPU(M, N, A, B, cpu_result);


	for (int i = 0; i < M * N * M * N; i++) {
		if (fabs(gpu_result[i] - cpu_result[i]) > 0.01) {
			printf("\n Mismatch at index %d: GPU: %f CPU %f\n", i, gpu_result[i], cpu_result[i]);

			free(A);
			free(B);
			free(cpu_result);
			free(gpu_result);
			return -1;
		}
	}

	printf("\nDone %f %f\n", gpu_result[M * N * M * N - 1], cpu_result[M * N * M * N - 1]);


	free(A);
	free(B);
	free(cpu_result);
	free(gpu_result);

	return 0;
}
