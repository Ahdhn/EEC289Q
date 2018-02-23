__device__ __forceinline__ 
void assign_color(const uint32_t set_id, 
                  const  uint32_t currentColor, 
                  const uint32_t NumRow,  
                  bool*sh_set, 
                  unsigned char my_thd_colors[], 
                  uint32_t color_id){

	//Assigne color k to vertices marked as true in set array
	if(set_id < NumRow){		
		my_thd_colors[color_id] = my_thd_colors[color_id] + (my_thd_colors[color_id]==0)*
		                                                    currentColor*
		                                                    sh_set[set_id];
        
	}
}

/*__device__ __forceinline__ 
void assign_color(uint32_t currentColor, uint32_t NumRow,  bool*set, unsigned char* color){

	//Assigne color k to vertices marked as true in set array
	int row = blockIdx.x * blockDim.x + threadIdx.x;	

	if(row < NumRow){
		
		color[row] = color[row] + currentColor*set[row]*(color[row]==0);
		//to prevent an if statement  if set[row] is false (zero), color [row] won't be affected 
		//otherwise, it will get the correct color 
	}
}*/