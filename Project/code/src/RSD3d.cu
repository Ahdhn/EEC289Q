// Final project EEC289Q - Winter 2018
// Applying Recursive Spoke Darts on GPU using CUDA
//https://www.sciencedirect.com/science/article/pii/S1877705816333380

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "tree.cpp"
#include "RSD_imp.cu"
#include <stdint.h>

//#include "spokes.cu"
#include "kdtree.h"


static void HandleError(cudaError_t err, const char *file, int line) {
	//Error handling micro, wrap it around function whenever possible
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		system("pause");
		//exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


void PointsGen(std::string FileName, int Num){
	//generate Num points inside a unit box
	std::fstream file(FileName.c_str(), std::ios::out);
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
	printf("\n  Peak Memory Bandwidth: %f(MB/s)\n\n", maxBW);
}
void ReadPoints(std::string FileName, int&NumPoints, real3*&Points){
	//Read the input set of points
	//Points should be un-initialized
	//the file should start with the number points
	//and then list (x,y,z) of the points
	std::fstream file;
	file.open(FileName.c_str());
	if (!file.is_open()){
		std::cout << " Error:: Can not open "<<FileName << std::endl;
		exit(EXIT_FAILURE);
	}
	
//	std::fstream test("test.txt", std::ios::out);
//	test << __FILE__;
//	test.close();

	NumPoints = 0;
	file >> NumPoints;
	Points = new real3[NumPoints];

	for (int i = 0; i < NumPoints; i++){
//		Points[i] = new real[3];
//		file >> Points[i][0] >> Points[i][1] >> Points[i][2];
		file >> Points[i].x >> Points[i].y >> Points[i].z;
	}
	file.close();
	std::cout<<" NumPoints= "<<NumPoints<<std::endl;
}
void TestTree(kdtree& tree, size_t NumPoints)
{
	uint32_t numInside0 = 0;
	uint32_t numInside1 = 0;
	uint32_t* inside0 = new uint32_t[1 << 14];
	uint32_t* inside1 = new uint32_t[1 << 14];
	for (size_t iPoint = 0; iPoint < 20; iPoint++)
	{
		real r = 0.1;

		numInside0 = 0;
		tree.treePointsInsideSphere(iPoint, r, inside0, numInside0);
		std::sort(inside0, inside0 + numInside0);


		numInside1 = 0;
		tree.treePointsInsideSphereBF(iPoint, r, inside1, numInside1);
		std::sort(inside1, inside1 + numInside1);

		if (numInside0 != numInside1)
			printf("mismatch.\n");
		for (size_t in = 0; in < numInside0; in++)
		{
			if (inside0[in] != inside1[in])
				printf("mismatch.\n");
		}
		if (0)
		{
			for (size_t in = 0; in < numInside0; in++)
				printf("%i, ", inside0[in]);

			printf("point %i neighbors are (%i) : ", iPoint, numInside0);
			for (size_t in = 0; in < numInside0; in++)
				printf("%i, ", inside0[in]);
			printf("\n");

			printf("point %i neighbors are (%i) : ", iPoint, numInside1);
			for (size_t in = 0; in < numInside1; in++)
				printf("%i, ", inside1[in]);
			printf("\n");
		}
	}

	printf("All good!\n");

}
void BuildNeighbors(kdtree&tree, size_t NumPoints, uint32_t*& h_neighbors, size_t offset)
{
	h_neighbors = new uint32_t[NumPoints* offset];
	memset(h_neighbors, 0, NumPoints* offset * sizeof(uint32_t));
	real r = 0.4;
	for (size_t iPoint = 0; iPoint < NumPoints; iPoint++)
	{
		size_t start = iPoint * offset;
		h_neighbors[start] = 0;
		tree.treePointsInsideSphere(iPoint, r, (&h_neighbors[start] + 1), h_neighbors[start]);
		if (h_neighbors[start] > offset - 1)
			printf("Error! in line %i.\n",__LINE__);
	}


}

int main(int argc, char**argv){
	//0) Generate the input points
	PointsGen("../../data/tiny.txt", 100);


	DeviceQuery();

	//1) Read input set of points
	int NumPoints;
	real3* Points=NULL;
	ReadPoints("../../data/tiny.txt",NumPoints, Points);


	//2) Build Data Structure
	kdtree tree; 
	uint32_t* h_neighbors;
	int MaxOffset = 32;
	tree.bulkBuild(Points, NumPoints);
	BuildNeighbors(tree, NumPoints, h_neighbors, MaxOffset);
	//TestTree(tree, NumPoints);
	
	//3) Move Data to GPU
	real3* d_points = NULL; uint32_t* d_neighbors = NULL; uint32_t* d_delaunay = NULL;
	cudaMalloc((void**)&d_delaunay, NumPoints * MaxOffset * sizeof(uint32_t));
	HANDLE_ERROR(cudaGetLastError());


	cudaMalloc((void**)&d_points, NumPoints * sizeof(real3));
	cudaMemcpy(d_points, Points, NumPoints * sizeof(real3), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	cudaMalloc((void**)&d_neighbors, NumPoints * MaxOffset * sizeof(uint32_t));
	cudaMemcpy(d_neighbors, h_neighbors, NumPoints * MaxOffset * sizeof(uint32_t), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	//4) Launch kernels and record time
	RSD_Imp << <1, 1 >> > (d_points, d_neighbors, NumPoints, d_delaunay, MaxOffset);
	HANDLE_ERROR(cudaGetLastError());
	cudaDeviceSynchronize();

	//5) Move results to CPU
	uint32_t* h_delaunay = new uint32_t[NumPoints * MaxOffset];
	cudaMemcpy(h_delaunay, d_delaunay, NumPoints * MaxOffset * sizeof(uint32_t), cudaMemcpyDeviceToHost);


	//6) Check correctness of the construction


	//7) Release memory


	int dummy = 0;
	std::cin >> dummy;


	cudaFree(d_points);
	cudaFree(d_neighbors);
	cudaFree(d_delaunay);

	delete[] Points;
	delete[] h_neighbors;
	delete[] h_delaunay;
	return 0;
}
