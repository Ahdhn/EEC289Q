#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"
#include <stdint.h>
#include <curand_kernel.h>


__global__ void RSD_Imp(real3* d_points, uint32_t* d_neighbors, int NPoints, uint32_t* d_delaunay, int MaxOffset, curandState* globalState){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//printf("I'm Thread No. %i\n", tid);
	
	//printf("My Point is (%f,%f,%f). \n", d_points[tid].x, d_points[tid].y, d_points[tid].z);
	
	//printf("My neighbors (%i):", d_neighbors[MaxOffset * tid]);
	for (int iN = 1; iN <= d_neighbors[MaxOffset * tid]; iN++){
		uint32_t myN = d_neighbors[MaxOffset * tid + iN];
		//printf(" %i", myN);
	}
	//printf("\n");

	if (tid > NPoints){ return; }
	
	real xx, yy, zz;
	RandSpoke3D(0, 0, 0, xx, yy, zz, globalState, tid);

	explore(tid); 

	// Now we have 3 neighbors and a vertex:
	
	//uint3 neighbors;
	//real3 vertex;
	//Propagate(tid,neighbors, vertex, d_delaunay);

}