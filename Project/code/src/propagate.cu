#include "spokes.cu"
#include <stdint.h>

__device__ __forceinline__ void Propagate(int tid, uint3 neighbors, real3 vertex, uint32_t* d_delaunay){

	printf("I'm Thread No. %i and I'm Propagating.\n", tid);



}