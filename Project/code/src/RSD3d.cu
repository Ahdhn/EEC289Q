// Final project EEC289Q - Winter 2018
// Applying Recursive Spoke Darts on GPU using CUDA
//https://www.sciencedirect.com/science/article/pii/S1877705816333380

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <curand.h> 


#include "kdtree.h"
#include "utilities.h"
#include "RSD_imp.cu"
#include "validate.h"
#include "extractTets.h"
#include "circumSphere.h"
#include "defines.h"

#ifdef _WIN32
#define DEVICE_ID 0
#else
#define DEVICE_ID 3
#endif



__global__ void initialise_curand_on_kernels(curandState * state, unsigned long seed)
{
	//stolen from https://nidclip.wordpress.com/2014/04/02/cuda-random-number-generation/

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

int main(int argc, char**argv){
	//0) Generate the input points
	//PointsGen("../data/small.txt", 1000);

	DeviceQuery(DEVICE_ID);
	

	//1) Read input set of points
	int NumPoints;
	real3* Points=NULL;
	//ReadPoints("../data/0.1.dat",NumPoints, Points);
	ReadPoints(argv[1],NumPoints, Points);



	//2) Build Data Structure
	kdtree tree; 
	uint32_t* h_neighbors;
	//int MaxOffset = MaxOffsets;
	tree.bulkBuild(Points, NumPoints);
	BuildNeighbors(tree, NumPoints, h_neighbors, MaxOffsets, atof(argv[2]));
	//TestTree(tree, NumPoints);

	std::fstream file("tree.csv", std::ios::out);
	file.precision(30);
	file<<"x coord, y coord, z coord, radius"<<std::endl;
	for(int i=0;i<NumPoints;i++){
		file<<Points[i].x<<", "<<Points[i].y<<", "<<Points[i].z<< ", "<< 0.001f+real(i)/(2.0*NumPoints)<<std::endl;
	}
	file.close();

	
	bool * h_bMarkers = new bool[NumPoints];
	uint32_t * h_triangluate = new uint32_t[NumPoints];
	uint32_t NumTriangultePoints = 0;
	for (int i = 0; i<NumPoints; i++){
		h_bMarkers[i] = 1;
		// boundary
		if (Points[i].x < 0.0) continue;
		if (Points[i].y < 0.0) continue;
		if (Points[i].z < 0.0) continue;
		if (Points[i].x > 1.0) continue;
		if (Points[i].y > 1.0) continue;
		if (Points[i].z > 1.0) continue;
		h_bMarkers[i] = 0;
		h_triangluate[NumTriangultePoints++] = i;
	}


	//3) Move Data to GPU
	real3* d_points = NULL; uint32_t* d_neighbors = NULL; uint32_t* d_delaunay = NULL;
	uint32_t * d_triangluate; bool * d_bMarkers;

	HANDLE_ERROR(cudaMalloc((void**)&d_delaunay, NumPoints * MaxOffsets * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMalloc((void**)&d_points, NumPoints * sizeof(real3)));
	HANDLE_ERROR(cudaMemcpy(d_points, Points, NumPoints * sizeof(real3), cudaMemcpyHostToDevice));	
	HANDLE_ERROR(cudaMalloc((void**)&d_neighbors, NumPoints * MaxOffsets * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_neighbors, h_neighbors, NumPoints * MaxOffsets * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	//boundary + to triangulate
	HANDLE_ERROR(cudaMalloc((void**)&d_triangluate, NumTriangultePoints * sizeof(uint32_t)));
	HANDLE_ERROR(cudaMemcpy(d_triangluate, h_triangluate, NumTriangultePoints * sizeof(uint32_t), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_bMarkers, NumPoints * sizeof(bool)));
	HANDLE_ERROR(cudaMemcpy(d_bMarkers, h_bMarkers, NumPoints * sizeof(bool), cudaMemcpyHostToDevice));

	//3.5) initialize rand number generator 
	//srand(time(NULL));
	curandState* deviceStates = NULL;
	int num = 1;
	HANDLE_ERROR(cudaMalloc(&deviceStates, num * sizeof(curandState)));
	initialise_curand_on_kernels << <num / 1024 + 1, 1024 >> >(deviceStates, unsigned(time(NULL)));
	HANDLE_ERROR(cudaDeviceSynchronize());
	std::cout<<" NumTriangultePoints= "<<NumTriangultePoints<<std::endl;

	//4) Launch kernels and record time		
	float exec_time = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, NULL);

	int numBlocks = NumTriangultePoints / NumThreads + 1;
	RSD_Imp << <numBlocks, NumThreads >> > (d_points, d_neighbors, NumPoints, d_delaunay, deviceStates, d_triangluate, d_bMarkers, NumTriangultePoints);
	

	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&exec_time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	HANDLE_ERROR(cudaGetLastError());
	HANDLE_ERROR(cudaDeviceSynchronize());
	
	
	printf("Consturction Time: %f mSec.\n", exec_time);

	//5) Move results to CPU
	uint32_t* h_delaunay = new uint32_t[NumPoints * MaxOffsets];
	HANDLE_ERROR(cudaMemcpy(h_delaunay, d_delaunay, NumPoints * MaxOffsets * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	SaveNeighborsCSV("del.csv", NumPoints, h_delaunay);
	//6) Check correctness of the construction
	//std::vector<std::vector<uint32_t>> myTets = extractTets(NumPoints, h_delaunay, MaxOffsets);
	//validate(myTets, Points, h_neighbors,MaxOffsets);	

	//7) Release memory


	
	cudaFree(d_points);
	cudaFree(d_neighbors);
	cudaFree(d_delaunay);
	
	cudaFree(d_triangluate);
	cudaFree(d_bMarkers);
	//cudaFree(deviceStates);

	delete[] Points;
	delete[] h_neighbors;
	delete[] h_delaunay;

	delete[] h_triangluate;
	delete[] h_bMarkers;

	return 0;
}
