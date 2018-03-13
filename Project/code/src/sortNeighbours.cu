#pragma once 

#include "defines.h"

__device__ __forceinline__ void sortNeighbours(){
	//Each thread sorts its neighbours in local registers 
	uint32_t sortedNeighbours[MaxOffsets];

}