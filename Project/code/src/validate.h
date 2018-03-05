#include <stdint.h>
#include <stdio.h>
#include "spokes.cu"

void validate(size_t NumPoints, real3* Points, uint32_t*& h_neighbors, uint32_t* d_delaunay){
	//validate the correctness of the delaunay by checking the empty sphere properity 
	for (uint32_t p = 0; p < NumPoints; p++){

	}
}