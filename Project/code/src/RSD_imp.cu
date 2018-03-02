#include "spokes.cu"
#include "propagate.cu"
#include "explore.cu"

__global__ void RSD_Imp(real3* d_points, int* d_neighbors, int NPoints, int* d_delaunay){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	printf("I'm Thread No. %i\n", tid);
	printf("My Point is (%f,%f,%f). \n", d_points[tid].x, d_points[tid].y, d_points[tid].z);
	if (tid > NPoints) return;

	Explore(tid); 

	// Now we have 3 neighbors and a vertex:
	uint3 neighbors;
	real3 vertex;

	Propagate(tid,neighbors, vertex, d_delaunay);

}