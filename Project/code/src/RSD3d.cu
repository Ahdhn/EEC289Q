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
#include <stdint.h>

#include "kdtree.h"
#include "utilities.h"
#include "RSD_imp.cu"
#include "validate.h"


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
	HANDLE_ERROR(cudaMalloc((void**)&d_delaunay, NumPoints * MaxOffset * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMalloc((void**)&d_points, NumPoints * sizeof(real3)));
	HANDLE_ERROR(cudaMemcpy(d_points, Points, NumPoints * sizeof(real3), cudaMemcpyHostToDevice));	
	HANDLE_ERROR(cudaMalloc((void**)&d_neighbors, NumPoints * MaxOffset * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_neighbors, h_neighbors, NumPoints * MaxOffset * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	//4) Launch kernels and record time
	RSD_Imp << <1, 1 >> > (d_points, d_neighbors, NumPoints, d_delaunay, MaxOffset);
	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());

	//5) Move results to CPU
	uint32_t* h_delaunay = new uint32_t[NumPoints * MaxOffset];
	HANDLE_ERROR(cudaMemcpy(h_delaunay, d_delaunay, NumPoints * MaxOffset * sizeof(uint32_t), cudaMemcpyDeviceToHost));


	//6) Check correctness of the construction
	validate(NumPoints, Points, h_neighbors, h_delaunay);

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
