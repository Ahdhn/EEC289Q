//Applying Recursive Spoke Darts on GPU using CUDA 
//https://www.sciencedirect.com/science/article/pii/S1877705816333380

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>  
#include <stdio.h>

#include "tree.cpp"


typedef double real; //Change this between double or (float) single precision 


static void HandleError(cudaError_t err, const char *file, int line) {
	//Error handling micro, wrap it around function whenever possible
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		//system("pause");
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void PointsGen(std::string FileName, int Num){
	//generate Num points inside a unit box
	std::fstream file(FileName, std::ios::out);
	file.precision(30);
	file << Num << std::endl;

	for (int v = 0; v < Num; v++){
		file << double(rand()) / double(RAND_MAX) << " " <<
			    double(rand()) / double(RAND_MAX) << " " << 
				double(rand()) / double(RAND_MAX) << std::endl;
	}
	file.close();
}
void DeviceQuery(int dev = 0){

	//Display few releven information about the device 
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		printf("\n deviceCount is zero. I quit!!!");
		exit(EXIT_FAILURE);
	}
		
	cudaSetDevice(dev);

	cudaDeviceProp devProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, dev));	
	printf("\n  Total number of device: %d", deviceCount);
	printf("\n  Using device Number: %d", dev);
	printf("\n  Device name: %s", devProp.name);
	//printf("\n  devProp.major: %d", devProp.major);
	//printf("\n  devProp.minor: %d", devProp.minor);
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
	const double maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth / 8.0) / 1.0E3;
	printf("\n  Peak Memory Bandwidth: %f(MB/s)\n", maxBW);
}
void ReadPoints(std::string FileName, int&NumPoints, real**&Points){	
	//Read the input set of points
	//Points should be un-initialized 
	//the file should start with the number points
	//and then list (x,y,z) of the points 
	std::fstream file;
	file.open(FileName);

	NumPoints = 0;
	file >> NumPoints;
	Points = new real*[NumPoints];

	for (int i = 0; i < NumPoints; i++){
		Points[i] = new real[3];

		file >> Points[i][0] >> Points[i][1] >> Points[i][2];
	}
	file.close();
}

int main(int argc, char**argv){
	//0) Generate the input points
	//PointsGen("tiny.txt", 10);

	DeviceQuery();
	
	//1) Read input set of points
	int NumPoints;
	real ** Points=NULL;
	ReadPoints("tiny.txt",NumPoints, Points);


	//2) Build Data Structure 


	//3) Move Data to GPU


	//4) Launch kernels and record time 



	//5) Check correctness of the construction 


	//6) Release memory 

	
	
	return 0;
}