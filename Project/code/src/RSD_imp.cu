#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"

__global__ void RSD_Imp(real* d_points, int* d_neighbors, int NPoints, int* d_delaunay){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	printf("I'm Thread No. %i\n", tid);

	if (tid > NPoints) return;

	Explore(tid); 

	// Now we have 3 neighbors and a vertex:
	uint3 neighbors;
	real3 vertex;

	Propagate(tid,neighbors, vertex, d_delaunay);

}