__device__ __forceinline__ 
void assign_color(uint32_t currentColor, uint32_t NumRow,  bool*set, unsigned char* color){

	//Assigne color k to vertices marked as true in set array
	int row = blockIdx.x * blockDim.x + threadIdx.x;	

	if(row < NumRow){		
		color[row] = color[row] + currentColor*set[row]*(color[row]==0);
		//to prevent an if statement  if set[row] is false (zero), color [row] won't be affected 
		//otherwise, it will get the correct color 
	}
}