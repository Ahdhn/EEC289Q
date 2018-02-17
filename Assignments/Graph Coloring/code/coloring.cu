#include "assign_color.cu"
#include "indept_set.cu"

__global__ void graphColoring(uint32_t NumRow, //number of vertices (= number of rows in adjacency matrix)
	                          uint32_t numNNZ, //number of non zero entry of the adjacency matrix
	                          uint32_t *col_id, //the column id in the CSR format 
	                          uint32_t *offset, //the row offset in the CSR
	                          unsigned char* color //the color of the vertices (output)
	                          ){




}
