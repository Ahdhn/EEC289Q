#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"
#include "circumSphere.h"

#include <stdint.h>
#include <curand_kernel.h>
#define DEBUG

__global__ void RSD_Imp(real3* d_points, uint32_t* d_neighbors, int NPoints, uint32_t* d_delaunay, const int MaxOffset, curandState* globalState){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > NPoints){ return; }
	
	uint3 exploredID;
	real3 sharedVertex;
	
	explore(tid,
	        d_points,
	        d_neighbors,
	        MaxOffset,
	        globalState, tid,
	        exploredID,
	        sharedVertex);
#ifdef DEBUG
	real x_cirm, y_cirm, z_cirm;
		real r_cirm = circumSphere(d_points[tid].x, d_points[tid].y, d_points[tid].z,
			                       d_points[exploredID.x].x, d_points[exploredID.x].y, d_points[exploredID.x].z,
			                       d_points[exploredID.y].x, d_points[exploredID.y].y, d_points[exploredID.y].z,
			                       d_points[exploredID.z].x, d_points[exploredID.z].y, d_points[exploredID.z].z,
			                       x_cirm, y_cirm, z_cirm);

		

		for(uint32_t i=0; i<NPoints; i++){

			if(i == tid || i == exploredID.x ||i == exploredID.y ||i == exploredID.z){continue;}

			real dist = cuDist(d_points[i].x,d_points[i].y,d_points[i].z, x_cirm, y_cirm, z_cirm);
			if(dist+0.000001 < r_cirm){
			//	printf("Invalud vertex thread(%i) circumSphere( %f,%f, %f, %f ) insidePoint( %f,%f, %f )\n",tid, x_cirm, y_cirm, z_cirm, sqrt(r_cirm), d_points[i].x,d_points[i].y,d_points[i].z);

			}
		}
#endif
	//printf("\n FINAL sharedVertex.x= %f sharedVertex.y= %f sharedVertex.z= %f\n", sharedVertex.x, sharedVertex.y, sharedVertex.z);
	//printf("\n FINAL tid =%u, exploredID.x= %d exploredID.y= %d exploredID.z= %d\n",tid, int(exploredID.x), int(exploredID.y), int(exploredID.z));
	

  
	// Now we have 3 neighbors and a vertex:
	//Propagate(tid,exploredID, sharedVertex, d_delaunay);

}