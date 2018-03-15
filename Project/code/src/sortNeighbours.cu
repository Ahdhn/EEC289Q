#pragma once 

#include "defines.h"

__device__ __forceinline__ void sortNeighbours(const int vertexID, uint32_t* d_neighbors, int sortedNeighbours[]){
	//Each thread sorts its neighbours in local registers 
	//using insert sort 
	
	uint32_t base = MaxOffsets*vertexID;

	

#pragma unroll 	
	//Non-coalesced read 
	for(uint32_t i=0; i< MaxOffsets; i++){
		sortedNeighbours[i] = d_neighbors[base + i];
	}
	
	/*if(vertexID == 0){
		printf("\n{ ");
		for(int i=1; i<=sortedNeighbours[0]; i++){
			printf("%d,",sortedNeighbours[i]);
		}
		printf("}\n");
	}*/


	for(int i=2; i<=sortedNeighbours[0]; i++){
		int j;
		uint32_t val = sortedNeighbours[i];

		for(j = i-1; j>=1 && sortedNeighbours[j] > val; j--){
			sortedNeighbours[j+1] = sortedNeighbours[j];
		}
		sortedNeighbours[j + 1] = val;
	}	

	/*if(vertexID == 0){
		printf("\n{");
		for(int i=1; i<=sortedNeighbours[0]; i++){
			printf("%d,",sortedNeighbours[i]);
		}
		printf("}\n");
	}*/

}