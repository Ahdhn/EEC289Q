#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"
#include "explore_sh.cu"
#include "circumSphere.h"
#include "sortNeighbours.cu"
#include "moveDataToSharedMem.cu"
#include <stdint.h>
#include <curand_kernel.h>

//#define UseSharedMem

__global__ void RSD_Imp(real3* d_points, uint32_t* d_neighbors, int NPoints, uint32_t* d_delaunay,  
	                    curandState* globalState,
	                    uint32_t * d_triangluate,
	                    bool * d_bMarkers,
	                    uint32_t NumTriangultePoints){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > NumTriangultePoints){		
		return; 
	}

	uint32_t vertexID = d_triangluate[tid];
	uint3 exploredID;
	real3 sharedVertex;	

	real3 currentPoint = d_points[vertexID];
	uint32_t base = MaxOffsets* vertexID;//base for index the neighbour list 
	uint32_t neighbour_count = d_neighbors[base]; //number of neighbour around this vertex



#ifdef UseSharedMem
	__shared__ uint32_t sh_delaunay[MaxOffsets*NumThreads];
	__shared__ real3 sh_points[MaxOffsets*NumThreads];

	real x_vertex(d_points[vertexID].x), y_vertex(d_points[vertexID].y), z_vertex(d_points[vertexID].z);
	const uint32_t numNeighbour = d_neighbors[vertexID*MaxOffsets];
	moveDataToSharedMem(vertexID, tid, numNeighbour, sh_points, d_points, d_neighbors);

	
	explore_sh(vertexID,
			   tid,	
			   x_vertex, y_vertex, z_vertex,
	           sh_points,
	           numNeighbour, 
	           globalState, vertexID,
	           exploredID,
	           sharedVertex);
#else 
	explore(vertexID,
	        d_points,
	        d_neighbors,	        
	        globalState, vertexID,
	        exploredID,
	        sharedVertex);
#endif

#ifdef DEBUG
	real x_cirm, y_cirm, z_cirm;
	real r_cirm = circumSphere(d_points[vertexID].x, d_points[vertexID].y, d_points[vertexID].z,
			                   d_points[exploredID.x].x, d_points[exploredID.x].y, d_points[exploredID.x].z,
			                   d_points[exploredID.y].x, d_points[exploredID.y].y, d_points[exploredID.y].z,
			                   d_points[exploredID.z].x, d_points[exploredID.z].y, d_points[exploredID.z].z,
			                   x_cirm, y_cirm, z_cirm);		

		for(uint32_t i=0; i<NPoints; i++){

			if(i == vertexID || i == exploredID.x ||i == exploredID.y ||i == exploredID.z){continue;}

			real dist = cuDist(d_points[i].x,d_points[i].y,d_points[i].z, x_cirm, y_cirm, z_cirm);
			if(dist+0.000001 < r_cirm){
				uint32_t base = MaxOffsets* vertexID;//base for index the neighbour list 
				uint32_t neighbour_count = d_neighbors[base]; //number of neighbour around this vertex
				bool true_invalid = false;;
				for(uint32_t j=1; j<neighbour_count; j++){
					if(d_neighbors[base + j] == i){
						true_invalid = true;
						break;
					}
				}
				if(true_invalid){
					printf("\n TRUE Invalid vertexID (%d) circumSphere( %f,%f, %f, %f ) insidePoint( %f,%f, %f )\n",vertexID, x_cirm, y_cirm, z_cirm, sqrt(r_cirm), d_points[i].x,d_points[i].y,d_points[i].z);
				}else{
					printf("\n FALSE Invalid vertexID thread(%d)", int(tid));
				}

			}
		}
#endif



	// Now we have 3 neighbors and a vertex:
	Propagate(currentPoint, vertexID, exploredID, sharedVertex, d_points, d_delaunay, d_neighbors, base, neighbour_count, d_bMarkers);

}