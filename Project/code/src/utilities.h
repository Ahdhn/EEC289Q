#pragma once
#include "defines.h"

static void HandleError(cudaError_t err, const char *file, int line) {
	//Error handling micro, wrap it around function whenever possible
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
#ifdef _WIN32
		system("pause");
#else
		exit(EXIT_FAILURE);
#endif
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void PointsGen(std::string FileName, int Num){
	//generate Num points inside a unit box
	std::fstream file(FileName.c_str(), std::ios::out);
	file.precision(30);
	file << Num << std::endl;

	for (int v = 0; v < Num; v++){

		double randX = double(rand()) / double(RAND_MAX) * 2.0 - 0.5;
		double randY = double(rand()) / double(RAND_MAX) * 2.0 - 0.5;
		double randZ = double(rand()) / double(RAND_MAX) * 2.0 - 0.5;

		file << randX << " " <<
			randY << " " <<
			randZ << std::endl;
	}
	file.close();
}

void SaveNodePoints(std::string FileName, int Num, real3* Points){

	std::fstream file(FileName.c_str(), std::ios::out);
	file.precision(30);
	file << Num << "	" << "3 0 0" << std::endl;

	for (int v = 0; v < Num; v++){
		file << v + 1 << "	" <<
			Points[v].x << "	" <<
			Points[v].y << "	" <<
			Points[v].z << std::endl;
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
		std::cout << " Error:: Can not open " << FileName << std::endl;
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
	std::cout << " NumPoints= " << NumPoints << std::endl;
}
void TestTree(kdtree& tree, size_t NumPoints)
{
	uint32_t numInside0 = 0;
	uint32_t numInside1 = 0;
	uint32_t* inside0 = new uint32_t[1 << 14];
	uint32_t* inside1 = new uint32_t[1 << 14];
	for (size_t iPoint = 0; iPoint < 20; iPoint++)
	{
		real r = real(0.1);

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

			printf("point %lu neighbors are (%i) : ", iPoint, numInside0);
			for (size_t in = 0; in < numInside0; in++)
				printf("%i, ", inside0[in]);
			printf("\n");

			printf("point %lu neighbors are (%i) : ", iPoint, numInside1);
			for (size_t in = 0; in < numInside1; in++)
				printf("%i, ", inside1[in]);
			printf("\n");
		}
	}

	printf("All good!\n");

}
void BuildNeighbors(kdtree&tree, size_t NumPoints, uint32_t*& h_neighbors, size_t offset, const real r)
{
	h_neighbors = new uint32_t[NumPoints* offset];
	memset(h_neighbors, 0, NumPoints* offset * sizeof(uint32_t));
	//real r = real(0.2);
	for (size_t iPoint = 0; iPoint < NumPoints; iPoint++)
	{
		size_t start = iPoint * offset;
		h_neighbors[start] = 0;
		tree.treePointsInsideSphere(iPoint, r, (&h_neighbors[start] + 1), h_neighbors[start]);
		if (h_neighbors[start] > offset - 1)
			printf("Error! in line %i (Building Neighbors). Increase offset.\n", __LINE__);
	}
}
template <typename T>
inline T Dist(T x1, T y1, T z1, T x2, T y2, T z2){
	//square distance between point (x1,y1,z1) and (x2,y2,z2) 
	T dx, dy, dz;
	dx = x1 - x2;
	dy = y1 - y2;
	dz = z1 - z2;
	dx *= dx;
	dy *= dy;
	dz *= dz;
	return dx + dy + dz;
}
void SaveNeighborsCSV(std::string FileName, int Num, uint32_t* con){
	
	std::fstream file(FileName.c_str(), std::ios::out);
	file.precision(30);

	for (int v = 0; v < Num; v++){
		file << v;
		int base = v * MaxOffsets;
		int numNeighbors = con[base];
		for (size_t i = 1; i <= numNeighbors; i++)
		{
			file << "," << con[base + i] + 1;

		}
		file << std::endl;
	}
	file.close();
}