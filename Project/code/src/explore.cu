#include "spokes.cu"
#include <stdint.h>

__device__ __forceinline__ void Explore(int tid){

	printf("I'm Thread No. %i and I'm Exploring.\n", tid);


}