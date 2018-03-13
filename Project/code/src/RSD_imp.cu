#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"
#include "circumSphere.h"

#include <stdint.h>
#include <curand_kernel.h>
#define DEBUG

__global__ void RSD_Imp(real3* d_points, uint32_t* d_neighbors, int NPoints, uint32_t* d_delaunay, const int MaxOffset, curandState* globalState,
	uint32_t * d_triangluate,
	bool * d_bMarkers,
	uint32_t NumTriangultePoints){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > NumTriangultePoints){ return; }

	int vertexID = d_triangluate[tid];

	uint3 exploredID;
	real3 sharedVertex;
	/*
	explore(vertexID,
	        d_points,
	        d_neighbors,
	        MaxOffset,
	        globalState, vertexID,
	        exploredID,
	        sharedVertex);
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
			//	printf("Invalud vertex thread(%i) circumSphere( %f,%f, %f, %f ) insidePoint( %f,%f, %f )\n",tid, x_cirm, y_cirm, z_cirm, sqrt(r_cirm), d_points[i].x,d_points[i].y,d_points[i].z);

			}
		}
#endif
	//printf("\n FINAL sharedVertex.x= %f sharedVertex.y= %f sharedVertex.z= %f\n", sharedVertex.x, sharedVertex.y, sharedVertex.z);
	//printf("\n FINAL tid =%u, exploredID.x= %d exploredID.y= %d exploredID.z= %d\n",tid, int(exploredID.x), int(exploredID.y), int(exploredID.z));
	

  */

	exploredID.x = 88;
	exploredID.y = 97;
	exploredID.z = 95;

	sharedVertex.x = 0.517865996449;
	sharedVertex.y = 0.406874834770;
	sharedVertex.z = -0.483357391886;

	// todo: read in the beginning
	real3 currentPoint = d_points[vertexID];
	uint32_t base = MaxOffset* vertexID;//base for index the neighbour list 
	uint32_t neighbour_count = d_neighbors[base]; //number of neighbour around this vertex

	// Now we have 3 neighbors and a vertex:
	Propagate(currentPoint, tid, exploredID, sharedVertex, d_points, d_delaunay, d_neighbors, base, neighbour_count, d_bMarkers);

}