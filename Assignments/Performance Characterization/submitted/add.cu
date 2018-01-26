#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>

cudaDeviceProp devProp;

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <algorithm>

void basic(const double maxBW);
void part2_1(const double maxBW);
void part2_2(const double maxBW);
void part2_4(const double maxBW);

#define Nn 1048576
#define loop 600
#define sh_len 4

//error handling micro, wrap it around function whenever possible
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void addManyThreadsManyBlocks(int n, float *x, float *y){
	/* Basic Kernel*/
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
		    y[i] = x[i] + y[i];
	}
}

__global__ void addManyThreadsManyBlocks_float4(float4 *x, float4 *y){
	/*Part 2.1*/
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
#pragma unroll
  for (int i = index; i < Nn/4; i += stride){
		y[i].x = x[i].x + y[i].x;
		y[i].y = x[i].y + y[i].y;
		y[i].z = x[i].z + y[i].z;
		y[i].w = x[i].w + y[i].w;
	}
}

__global__ void addManyThreadsManyBlocks_float4_arth(float4 *x, float4 *y){
	/*Part 2.2*/
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

#pragma unroll
  for (int i = index; i < Nn/4; i += stride){
		float x1 = x[index].x;
		float y1 = x[index].y;
		float z1 = x[index].z;
		float w1 = x[index].w;
		float x2 = y[index].x;
		float y2 = y[index].y;
		float z2 = y[index].z;
		float w2 = y[index].w;
#pragma unroll
		for(int j=0; j < loop; j++){
				x1 = x1*x2 + y1;
				y1 = y1*y2 + z1;
				z1 = z1*z2 + w1;
				w1 = w1*w2 + x1;
			}
			y[index].x= x1;
			y[index].y= y1;
			y[index].z= z1;
			y[index].w= w1;
	}
}

__global__ void addManyThreadsManyBlocks_float4_balance(float4 *x, float4 *y, const int iter){
	/*Part 2.4*/
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

#pragma unroll
  for (int i = index; i < Nn/4; i += stride){
		float x1 = x[index].x;
		float y1 = x[index].y;
		float z1 = x[index].z;
		float w1 = x[index].w;
		float x2 = y[index].x;
		float y2 = y[index].y;
		float z2 = y[index].z;
		float w2 = y[index].w;
#pragma unroll
		for(int j=0; j < iter; j++){
				x1 = x1*x2 + y1;
				y1 = y1*y2 + z1;
				z1 = z1*z2 + w1;
				w1 = w1*w2 + x1;
			}
			y[index].x= x1;
			y[index].y= y1;
			y[index].z= z1;
			y[index].w= w1;
	}
}

/********************************************************************************/
    /************************************************************************/
		    /****************************************************************/
				    /********************************************************/

void part2_4(const double maxBW){
	/******* Part 2.4 ******/

	float4 *x, *y;
  cudaMallocManaged(&x, Nn/4*sizeof(float4));
  cudaMallocManaged(&y, Nn/4*sizeof(float4));
	// initialize x and y arrays on the host
  for (int i = 0; i < Nn/4; i++) {
  	x[i].x = 1.0f; x[i].y = 1.0f; x[i].z = 1.0f; x[i].w = 1.0f;
  	y[i].x = 2.0f; y[i].y = 2.0f; y[i].z = 2.0f; y[i].w = 2.0f;
  }

	const int numIter = 1000;
	const int N = Nn;
	const float mem = (3.0*sizeof(float)*N)/ (1024*1024); //in mega bytes

	cudaFuncSetCacheConfig(addManyThreadsManyBlocks_float4, cudaFuncCachePreferL1);

	const int numBlocks = 1024;
	const int numThreads = 256;

	for(int iter=1; iter<loop; iter++){
		cudaEvent_t start, stop;
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventRecord(start, NULL));
		for (int i=0; i< numIter; i++){
			addManyThreadsManyBlocks_float4_balance<<<numBlocks, numThreads>>>(x, y, iter);
		}
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaEventRecord(stop, NULL));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		float time = 0.0f;
		HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
		time /= numIter; //time in mSec
		time /=1000;//time in seconds
		//std::cout<<"\n  numThreads= "<<numThreads<< " numBlocks= "<<numBlocks<< " time(sec)= "<<time <<"  BW(MB)="<< mem/time << "   BW_frac(MB/sec)= "<<(mem/time)/maxBW<<std::endl;
		float gflops = (3.0*iter*Nn/time)/(1024*1024*1024);
		std::cout<<"\n  GFLOPS= "<<gflops<< " achieved(%)= "<< 100*gflops/4291.2;
		std::cout<<"    Arth_int (FLOPS/Byte)= "<< 1024*gflops/mem;
	}


	// Free memory
  cudaFree(x);
  cudaFree(y);
}

int main(int argc, char**argv){

	if(argc != 2){
		std::cout<<"  Usage ./add [ProblemID]"<<std::endl;
		std::cout<< "   ProblemID= 1 for part 1 "<<std::endl;
		std::cout<< "   ProblemID= 21 for part 2.1 "<<std::endl;
		std::cout<< "   ProblemID= 22 for part 2.2 "<<std::endl;
		std::cout<< "   ProblemID= 24 for part 2.4 "<<std::endl;
		exit(EXIT_FAILURE);
	}

	int ProblemID = atoi(argv[1]);
	if(ProblemID!=1 && ProblemID!=21 && ProblemID!=22 && ProblemID!=24){
		std::cout<< " Invalid input."<<std::endl;
		std::cout<<"  Usage ./add [ProblemID]"<<std::endl;
		std::cout<< "   ProblemID=1 for part 1 "<<std::endl;
		std::cout<< "   ProblemID=21 for part 2.1 "<<std::endl;
		std::cout<< "   ProblemID=22 for part 2.2 "<<std::endl;
		std::cout<< "   ProblemID=24 for part 2.4 "<<std::endl;
		exit(EXIT_FAILURE);
	}

  int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0){
		printf("\n deviceCount is zero. I quit!!!");
		exit(EXIT_FAILURE);
	}

	const int dev = (deviceCount == 1) ? 0 : 3;
	cudaSetDevice(dev);

	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, dev));
	printf("\n  Total number of device: %d", deviceCount);
	printf("\n  Using device Number: %d", dev);
	printf("\n  Device name: %s", devProp.name);
	//printf("\n  devProp.major: %d", devProp.major);
	//printf("\n  devProp.minor: %d", devProp.minor);
	if(devProp.major==1){//Fermi
		if(devProp.minor==1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*48);
		}else{
			printf("\n  SM Count: %d", devProp.multiProcessorCount*32);
		}
	}else if(devProp.major==3){//Kepler
		printf("\n  SM Count: %d", devProp.multiProcessorCount*192);
	}else if(devProp.major==5){//Maxwell
		printf("\n  SM Count: %d", devProp.multiProcessorCount*128);
	}else if(devProp.major==6){//Pascal
		if(devProp.minor==1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*128);
		}else if(devProp.minor==0){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*64);
		}
	}

	printf("\n  Compute Capability: v%d.%d", (int)devProp.major, (int)devProp.minor);
	printf("\n  Memory Clock Rate: %d(kHz)", devProp.memoryClockRate);
	printf("\n  Memory Bus Width: %d(bits)", devProp.memoryBusWidth);
	const double maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth/8.0)/1.0E3;
	printf("\n  Peak Memory Bandwidth: %f(MB/s)\n", maxBW);




	if(ProblemID == 1){
		basic(maxBW);
	}else if (ProblemID == 21){
			part2_1(maxBW);
	}else if(ProblemID == 22){
		part2_2(maxBW);
	}else if(ProblemID == 24){
		part2_4(maxBW);
	}


  return 0;
}

/******************************************/
/******************************************/
void part2_2(const double maxBW){
	/******* Part 2.2 ******/

	float4 *x, *y;
  cudaMallocManaged(&x, Nn/4*sizeof(float4));
  cudaMallocManaged(&y, Nn/4*sizeof(float4));
	// initialize x and y arrays on the host
  for (int i = 0; i < Nn/4; i++) {
  	x[i].x = 1.0f; x[i].y = 1.0f; x[i].z = 1.0f; x[i].w = 1.0f;
  	y[i].x = 2.0f; y[i].y = 2.0f; y[i].z = 2.0f; y[i].w = 2.0f;
  }

	const int numIter = 1000;
	const int N = Nn;
	const float mem = (3.0*sizeof(float)*N)/ (1024*1024); //in mega bytes

	cudaFuncSetCacheConfig(addManyThreadsManyBlocks_float4, cudaFuncCachePreferL1);

	const int numBlocks = 1024;
	const int numThreads = 256;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, NULL));
	for (int i=0; i< numIter; i++){
		addManyThreadsManyBlocks_float4_arth<<<numBlocks, numThreads>>>(x, y);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaEventRecord(stop, NULL));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /=1000;//time in seconds
	std::cout<<"\n  numThreads= "<<numThreads<< " numBlocks= "<<numBlocks<< " time(sec)= "<<time <<"  BW(MB)="<< mem/time << "   BW_frac(MB/sec)= "<<(mem/time)/maxBW<<std::endl;
	float gflops = (3.0*loop*Nn/time)/(1024*1024*1024);
	std::cout<<"\n  GFLOPS= "<<gflops<< " achieved(%)= "<< 100*gflops/4291.2  <<std::endl;
	std::cout<<"  Arth_int (FLOPS/Byte)= "<<1024*gflops/mem<<std::endl;


	// Free memory
  cudaFree(x);
  cudaFree(y);
}
/******************************************/
/******************************************/
void part2_1(const double maxBW){
	/******* Part 2.1 ******/

	float4 *x, *y;
  cudaMallocManaged(&x, Nn/4*sizeof(float4));
  cudaMallocManaged(&y, Nn/4*sizeof(float4));
	// initialize x and y arrays on the host
  for (int i = 0; i < Nn/4; i++) {
  	x[i].x = 1.0f; x[i].y = 1.0f; x[i].z = 1.0f; x[i].w = 1.0f;
  	y[i].x = 2.0f; y[i].y = 2.0f; y[i].z = 2.0f; y[i].w = 2.0f;
  }

	const int numIter = 1000;
	const int N = Nn;
	const float mem = (3.0*sizeof(float)*N)/ (1024*1024); //in mega bytes

	cudaFuncSetCacheConfig(addManyThreadsManyBlocks_float4, cudaFuncCachePreferL1);

	const int numBlocks = 4096;
	const int numThreads = 64;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, NULL));
	for (int i=0; i< numIter; i++){
		addManyThreadsManyBlocks_float4<<<numBlocks, numThreads>>>(x, y);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaEventRecord(stop, NULL));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /=1000;//time in seconds
	std::cout<<"\n  numThreads= "<<numThreads<< " numBlocks= "<<numBlocks<< " time(sec)= "<<time <<"  BW(MB)="<< mem/time << "   BW_frac(MB/sec)= "<<(mem/time)/maxBW<<std::endl;

	//std::cout<<"\n  numThreads= "<<numThreads<< " numBlocks= "<<numBlocks<< " time= "<<time <<"  BW="<< mem/time << "   BW_frac= "<<(mem/time)/maxBW<<std::endl;
	float gflops = (Nn/time)/(1024*1024*1024);
	std::cout<<"\n  GFLOPS= "<<gflops<< " achieved(%)= "<< 100*gflops/4291.2  <<std::endl;
	std::cout<<"  Arth_int (FLOPS/Byte)= "<<1024*gflops/mem<<std::endl;


	// Free memory
  cudaFree(x);
  cudaFree(y);

}

/******************************************/
/******************************************/
void basic(const double maxBW){
	/******* Part One ******/
	int N = 1<<20;
	float *x, *y;
	// Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
	// initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
  	x[i] = 1.0f;
  	y[i] = 2.0f;
  }
	const int numIter = 1000;
	const float mem = (3.0*sizeof(float)*N)/ (1024*1024); //in mega bytes
	const int numBlocks = 4096;
	const int numThreads = 256;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, NULL));
	for (int i=0;i<numIter;i++){
		addManyThreadsManyBlocks<<<numBlocks, numThreads>>>(N, x, y);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaEventRecord(stop, NULL));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	time /= numIter; //time in mSec
	time /=1000;//time in seconds
	float myBW = mem/time ;
	std::cout<<"\n  numThreads= "<<numThreads<< " numBlocks= "<<numBlocks<< " time(sec)= "<<time <<"  BW(MB)="<< myBW << "   BW_frac(MB/sec)= "<<myBW/maxBW<<std::endl;


  cudaFree(x);
  cudaFree(y);
}
