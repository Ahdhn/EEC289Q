#pragma once 

//#include "spokes.cu"
#include <stdint.h>
#include "explore.cu"

template <class T>
__device__ __forceinline__ void swapTwoValues(T& lhs, T& rhs)
{
	T temp = lhs;
	lhs = rhs;
	rhs = temp;
}

__device__ bool isAdded(uint32_t& idx, uint32_t list[], uint32_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		if (list[i] == idx)
		{
			return true;
		}
	}

	return false;
}


__device__ bool isAdded(uint32_t& ix, uint32_t& iy, uint32_t& iz, uint3 list[], uint32_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		if (list[i].x == ix && list[i].y == iy && list[i].z == iz)
		{
			return true;
		}
	}
	//printf("Tets are:\n");
	//
	//for (size_t i = 0; i < size; i++)
	//	printf("%i, %i, %i\n", list[i].x, list[i].y, list[i].z);
	//printf("----\n");
	//printf("%i, %i, %i\n", ix, iy, iz);
	//printf("----\n");

	return false;
}
__device__  uint32_t getNewCircum(
	real3& curPoint, 
	real3& node0Loc, 
	real3& node1Loc,
	real3& thirdLoc,
	real3& spokeSt,
	uint32_t& node0,
	uint32_t& node1,
	uint32_t& node2,
	uint32_t& currentNode,
	real3& newCircum,
	uint32_t& neighbour_count,
	uint32_t& base, 
	uint32_t*& d_neighbors, 
	real3*& d_points)
{
	//float spokeEndX, spokeEndY, spokeEndZ;
	real3 bla;
	//TriCircumcenter3d(curPoint.x, curPoint.y, curPoint.z,
	//	node0Loc.x, node0Loc.y, node0Loc.z,
	//	node1Loc.x, node1Loc.y, node1Loc.z,
	//	newCircum.x, newCircum.y, newCircum.z);

	//printf("%f, %f, %f \n", newCircum.x, newCircum.y, newCircum.z);
	//printf("Tri corners:\n");
	//printf("%f, %f, %f \n", curPoint.x, curPoint.y, curPoint.z);
	//printf("%f, %f, %f \n", node0Loc.x, node0Loc.y, node0Loc.z);
	//printf("%f, %f, %f \n", node1Loc.x, node1Loc.y, node1Loc.z);

	//real3 triCenter;
	//triCenter.x = newCircum.x;
	//triCenter.y = newCircum.y;
	//triCenter.z = newCircum.z;


	//newCircum.x = 1000.0 * (newCircum.x - spokeSt.x) + spokeSt.x;
	//newCircum.y = 1000.0 * (newCircum.y - spokeSt.y) + spokeSt.y;
	//newCircum.z = 1000.0 * (newCircum.z - spokeSt.z) + spokeSt.z;
	
	
	real3 uT;
	real3 uuT, vvT;
	uuT.x = node0Loc.x - curPoint.x;
	uuT.y = node0Loc.y - curPoint.y;
	uuT.z = node0Loc.z - curPoint.z;

	vvT.x = node1Loc.x - curPoint.x;
	vvT.y = node1Loc.y - curPoint.y;
	vvT.z = node1Loc.z - curPoint.z;

	CrossProdcut(uuT.x, uuT.y, uuT.z, vvT.x, vvT.y, vvT.z, uT.x, uT.y, uT.z);
	
	NormalizeVector(uT.x, uT.y, uT.z);


	newCircum.x = 1000.0 * uT.x + spokeSt.x;
	newCircum.y = 1000.0 * uT.y + spokeSt.y;
	newCircum.z = 1000.0 * uT.z + spokeSt.z;

	//uT.x = triCenter.x - spokeSt.x;
	//uT.y = triCenter.y - spokeSt.y;
	//uT.z = triCenter.z - spokeSt.z;
	
	
	real dNT = thirdLoc.x * uT.x + thirdLoc.y * uT.y + thirdLoc.z * uT.z;
	dNT -= (node0Loc.x * uT.x + node0Loc.y * uT.y + node0Loc.z * uT.z);

	real dNC = newCircum.x * uT.x + newCircum.y * uT.y + newCircum.z * uT.z;
	dNC -= (node0Loc.x * uT.x + node0Loc.y * uT.y + node0Loc.z * uT.z);
	//if (node0 == 78 && node1 == 88 && node2 == 91)
	//{
	//	printf("Here we go\n");
	//	printf("%f , %f \n", dNT, dNC);
	//}
	if (((dNT > 0.0 && dNC > 0) || (dNT < 0.0 && dNC < 0)))
	{
		//printf("I reversed \n");

		uT.x *= -1.0;
		uT.y *= -1.0;
		uT.z *= -1.0;


		newCircum.x = 1000.0 * uT.x + spokeSt.x;
		newCircum.y = 1000.0 * uT.y + spokeSt.y;
		newCircum.z = 1000.0 * uT.z + spokeSt.z;

		//spokeSt.x = triCenter.x;
		//spokeSt.y = triCenter.y;
		//spokeSt.z = triCenter.z;
	}
	uint32_t newNeighbor;
	
	//printf("Ray:\n");
	//printf("%f, %f, %f \n", spokeSt.x, spokeSt.y, spokeSt.z);
	//printf("%f, %f, %f \n", newCircum.x, newCircum.y, newCircum.z);
	
	newNeighbor = NeighbourTriming(curPoint.x, curPoint.y, curPoint.z,
		spokeSt.x, spokeSt.y, spokeSt.z,
		newCircum.x, newCircum.y, newCircum.z,
		bla.x, bla.y,bla.z,
		node0, node1, node2, currentNode, 
		neighbour_count, base, d_neighbors, d_points);
	
	//printf("To Add %i \n", newNeighbor);
	//printf("Ray:\n");
	//printf("%f, %f, %f \n", spokeSt.x, spokeSt.y, spokeSt.z);
	//printf("%f, %f, %f \n", newCircum.x, newCircum.y, newCircum.z);

	return newNeighbor;
}
__device__ void Propagate(
	real3& curPoint, 
	int& tid, 
	uint3& neighbors, 
	real3& vertex, 
	real3* d_points, 
	uint32_t* d_delaunay,
	uint32_t* d_neighbors,
	uint32_t base,
	uint32_t neighbour_count,
	bool * d_bMarkers)
{
	
	//uint32_t numDelaunayNeighbors = 3;
	//d_delaunay[base + 1] = neighbors.x;
	//d_delaunay[base + 2] = neighbors.y;
	//d_delaunay[base + 3] = neighbors.z;

	//

	uint32_t currentNode = tid;
	bool seedB = d_bMarkers[currentNode];

	const int queueMaxSize = 40;
	const int delMaxSize = 40;

	int queueSize = 1;
	uint3 neighborsQueue[queueMaxSize];  // tets
	real3 vertexQueue[queueMaxSize];		// tets circum
	uint32_t delaunayList[delMaxSize];

	uint32_t numDel = 3;
	delaunayList[0] = neighbors.x;
	delaunayList[1] = neighbors.y;
	delaunayList[2] = neighbors.z;

	int face0 = neighbors.x;
	int face1 = neighbors.y;
	int face2 = neighbors.z;

	if (face0 > face1)
		swapTwoValues(face0, face1);
	if (face0 > face2)
		swapTwoValues(face0, face2);
	if (face1 > face2)
		swapTwoValues(face1, face2);

	//printf("%i , %i , %i\n", face0, face1, face2);

	vertexQueue[0] = vertex;
	neighborsQueue[0].x = face0;
	neighborsQueue[0].y = face1;
	neighborsQueue[0].z = face2;

	uint3 testList[queueMaxSize];  // tets
	uint32_t numTets = 1;
	testList[0].x = face0;
	testList[0].y = face1;
	testList[0].z = face2;

	//return;
	uint32_t node0, node1, node2;
	uint32_t node00, node10, node20;
	uint32_t newNeighbor;
	uint32_t newNeighbor0;
	bool isBoundary;

	//real spokeEndX, spokeEndY, spokeEndZ;
//	NeighbourTriming(vertexQueue[0].x, )
	while (queueSize)
	{
		queueSize--;

		// grap a tet and its circum
		real3 spokeSt;
		spokeSt.x = vertexQueue[queueSize].x;
		spokeSt.y = vertexQueue[queueSize].y;
		spokeSt.z = vertexQueue[queueSize].z;
		node0 = neighborsQueue[queueSize].x;
		node1 = neighborsQueue[queueSize].y;
		node2 = neighborsQueue[queueSize].z;

		node00 = node0;
		node10 = node1;
		node20 = node2;

		real3 node0Loc = d_points[node0];
		real3 node1Loc = d_points[node1];
		real3 node2Loc = d_points[node2];
		
		bool node0B = d_bMarkers[node0];
		bool node1B = d_bMarkers[node1];
		bool node2B = d_bMarkers[node2];

		//printf("%f, %f, %f \n", spokeSt.x, spokeSt.y, spokeSt.z);
		//
		//printf("%f, %f, %f \n", node0Loc.x, node0Loc.y, node0Loc.z);
		//printf("%f, %f, %f \n", node1Loc.x, node1Loc.y, node1Loc.z);
		//printf("%f, %f, %f \n", node2Loc.x, node2Loc.y, node2Loc.z);
		//
		//printf("%f, %f, %f \n", curPoint.x, curPoint.y, curPoint.z);

		//printf("---New tet: %i, %i, %i \n", node0, node1, node2);


		
		real3 newCircum;


		//if (!seedB && !node0B && !node1B)
		{
			// face cur-01
			newNeighbor = getNewCircum(
				curPoint,
				node0Loc, node1Loc,node2Loc,
				spokeSt,
				node0, node1, node2, currentNode,
				newCircum, neighbour_count, base, d_neighbors, d_points);

			//printf("%i, %i -> %i \n", node0, node1, newNeighbor);

			if (newNeighbor != UINT32_MAX)
			{
				if (queueSize == queueMaxSize)
					printf("Increase Queue size!\n");
				if (numDel == delMaxSize)
					printf("Increase Delaunay size!\n");
				if (numTets == queueMaxSize)
					printf("Increase Tets size!\n");

				//numDelaunayNeighbors++;
				//d_delaunay[base + numDelaunayNeighbors] = newNeighbor;
				newNeighbor0 = newNeighbor;

				if (node0 > node1)
					swapTwoValues(node0, node1);
				if (node0 > newNeighbor)
					swapTwoValues(node0, newNeighbor);
				if (node1 > newNeighbor)
					swapTwoValues(node1, newNeighbor);

				if (!isAdded(node0, node1, newNeighbor, testList, numTets))
				{
					if (!isAdded(newNeighbor0, delaunayList, numDel))
						delaunayList[numDel++] = newNeighbor0;


					testList[numTets].x = node0;
					testList[numTets].y = node1;
					testList[numTets].z = newNeighbor;
					numTets++;

					neighborsQueue[queueSize].x = node0;
					neighborsQueue[queueSize].y = node1;
					neighborsQueue[queueSize].z = newNeighbor;
			
					vertexQueue[queueSize].x = newCircum.x;
					vertexQueue[queueSize].y = newCircum.y;
					vertexQueue[queueSize].z = newCircum.z;
					queueSize++;
		
				//	printf("Added %i \n", newNeighbor);
				}
			}
		}

		node0 = node00;
		node1 = node10;
		node2 = node20;

		//if (!seedB && !node0B && !node2B)
		{
			// face cur-02
			newNeighbor = getNewCircum(
				curPoint,
				node0Loc, node2Loc,node1Loc,
				spokeSt,
				node0, node1, node2, currentNode,
				newCircum, neighbour_count, base, d_neighbors, d_points);
			
			//printf("%i, %i -> %i \n", node0, node2, newNeighbor);

			if (newNeighbor != UINT32_MAX )
			{
				//numDelaunayNeighbors++;
				//d_delaunay[base + numDelaunayNeighbors] = newNeighbor;
				if (queueSize == queueMaxSize)
					printf("Increase Queue size!\n");
				if (numDel == delMaxSize)
					printf("Increase Delaunay size!\n");
				if (numTets == queueMaxSize)
					printf("Increase Tets size!\n");
				newNeighbor0 = newNeighbor;

				if (node0 > node2)
					swapTwoValues(node0, node2);
				if (node0 > newNeighbor)
					swapTwoValues(node0, newNeighbor);
				if (node2 > newNeighbor)
					swapTwoValues(node2, newNeighbor);


				if (!isAdded(node0, node2, newNeighbor, testList, numTets))
				{
					if (!isAdded(newNeighbor0, delaunayList, numDel))
						delaunayList[numDel++] = newNeighbor0;

					testList[numTets].x = node0;
					testList[numTets].y = node2;
					testList[numTets].z = newNeighbor;
					numTets++;

					neighborsQueue[queueSize].x = node0;
					neighborsQueue[queueSize].y = node2;
					neighborsQueue[queueSize].z = newNeighbor;

					vertexQueue[queueSize].x = newCircum.x;
					vertexQueue[queueSize].y = newCircum.y;
					vertexQueue[queueSize].z = newCircum.z;
					queueSize++;
					//printf("Added %i \n", newNeighbor);
				}
			}
		}
		
		node0 = node00;
		node1 = node10;
		node2 = node20;
		//if (!seedB && !node1B && !node2B)
		{
			// face cur-12
			newNeighbor = getNewCircum(
				curPoint,
				node1Loc, node2Loc, node0Loc,
				spokeSt,
				node0, node1, node2, currentNode,
				newCircum, neighbour_count, base, d_neighbors, d_points);
			
			//printf("%i, %i -> %i \n", node1, node2, newNeighbor);
			
			if (newNeighbor != UINT32_MAX )
			{
				if (queueSize == queueMaxSize)
					printf("Increase Queue size!\n");

				if (queueSize == queueMaxSize)
					printf("Increase Queue size!\n");
				if (numDel == delMaxSize)
					printf("Increase Delaunay size!\n");
				newNeighbor0 = newNeighbor;

				if (node1 > node2)
					swapTwoValues(node1, node2);
				if (node1 > newNeighbor)
					swapTwoValues(node1, newNeighbor);
				if (node2 > newNeighbor)
					swapTwoValues(node2, newNeighbor);

				if (!isAdded(node1, node2, newNeighbor, testList, numTets))
				{
				//numDelaunayNeighbors++;
					//d_delaunay[base + numDelaunayNeighbors] = newNeighbor;
					if (!isAdded(newNeighbor0, delaunayList, numDel))
						delaunayList[numDel++] = newNeighbor0;

					testList[numTets].x = node1;
					testList[numTets].y = node2;
					testList[numTets].z = newNeighbor;
					numTets++;

					neighborsQueue[queueSize].x = node1;
					neighborsQueue[queueSize].y = node2;
					neighborsQueue[queueSize].z = newNeighbor;

					vertexQueue[queueSize].x = newCircum.x;
					vertexQueue[queueSize].y = newCircum.y;
					vertexQueue[queueSize].z = newCircum.z;
					queueSize++;
					//printf("Added %i \n", newNeighbor);
				}

			}

		}
	}

	d_delaunay[base] = numDel;
	for (int i = 0; i < numDel; i++)
		d_delaunay[base + i + 1] = delaunayList[i];

	//printf("Finished propagating! \n");
	//for (int i = 0; i < numDel; i++)
	//	printf("%i ", delaunayList[i]);
	//printf("Finished propagating! \n");

}