#pragma once 

__device__  __forceinline__ void moveDataToSharedMem(const int vertexID,
													 const int tid,
													 const uint32_t numNeighbour,	                								 
	                                                 real3 sh_points[],
	                                                 real3* d_points, uint32_t* d_neighbors){
	// threads cooperate to move data 
	uint32_t base = vertexID*MaxOffsets;
	uint32_t base_tid = tid*MaxOffsets;
	for(int i= 1; i<= numNeighbour; i++){
		uint32_t n = d_neighbors[base + i];
		sh_points[base_tid + i].x = d_points[n].x;
		sh_points[base_tid + i].y = d_points[n].y;
		sh_points[base_tid + i].z = d_points[n].z;
		/*if(vertexID == 10){
			printf("\n %i --> %i ", base_tid +  i , n);
		}*/
	}
}