//#define NUM_COLORS_PER_THREAD 10

//can be done such that all threads will coperate to update one color and them 
//move on to another one
//still shared memory will try to hold as much as possible of color 
//and the remaning part will be written to before writting to global mem 

__global__ void conflict_graph(uint32_t NumRow, //total number of rows  							   
	                           unsigned char* color, int*numColor,
	                           uint32_t *conflict_vertices,
	                           uint32_t *conflict_offset,
	                           uint32_t items_in_sh_mem){

	//copy subtset of the vertices colors 
	//each thread is responsible of one color
	//each thread will loop over the copied subset and figure out if it 
	//need to add it or not 

	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

	//grab some colors from global memory (multiplie of warpSize)
	__shared__ uint32_t sh_shift;
	sh_shift = 0;	
	extern __shared__ unsigned char shrd_ptr1[];
	unsigned char* sh_color= shrd_ptr1; 
	uint32_t myColor_Start/*, myColor_End*/;

	if(tid <(*numColor)) {//since each thread owns a color (and 0 means not colored)
		myColor_Start = conflict_offset[tid+1]; //since 0 is uncolored 
		//myColor_End = conflict_offset[tid+2];	
	}
	uint32_t numMyColors = 0;
	

	//loop starts here 
	for(uint32_t i=tid; i< items_in_sh_mem; i+=blockDim.x){
		sh_color[i] = color[i+sh_shift];		
	}
	__syncthreads(); 
	

	for(uint32_t v=0;v<items_in_sh_mem;v++){
		if(uint32_t(sh_color[v]) == tid + 1){//the color of this thread is tid+1 (since 0 is uncolored)	
			conflict_vertices[myColor_Start + numMyColors] = v + sh_shift; 
			//printf("\n tid=%d ---> conflict_vertices[%d]= %d, myColor_Start= %d, numMyColors= %d, v= %d, sh_shift= %d\n", tid, myColor_Start + numMyColors, conflict_vertices[myColor_Start + numMyColors],myColor_Start , numMyColors,  v, sh_shift);
			numMyColors++;
		}
	}
	

	if(tid == 0){
		sh_shift +=items_in_sh_mem; 
	}
	__syncthreads(); 

	//loop ends here 



	/*//create a csr for the conflicting graph 
	//working on one block only 

	//remember that after local coloring, for vertex i and j to be conflict, they have to be 
	//assigned colors using different blocks
	//i.e., floor(int(j)/int(blockingSize)) != floor(int(i)/int(blockingSize))

	//move color to shared memory (since it will be extensively reused)

	extern __shared__ unsigned char shrd_ptr1[]; 
	//since we have just one block so we won't care about the occupancy
	//so we would have one big shared memory assigned to the colors 
	extern __shared__ uint32_t conf[]; 

	unsigned char* sh_colors = shrd_ptr1; 


	for(uint32_t i=threadIdx.x; i<NumRow;i+=blockDim.x++){
		//move color to share mem
		sh_colors[i] = color[i];
	}
	__syncthreads();

	uint32_t myVertices[NUM_COLORS_PER_THREAD];

	//every thread works on one row in the graph 
	//MIACAP (make it as crappy as possible)

	uint32_t tid = blockDim*blockIdx.x + threadIdx.x;
	for(uint32_t i = tid; i < NumRow/NUM_COLORS_PER_THREAD +1; i+= blockDim.x ++){ //for each thread 

#pragma unroll 
		for(uint32_t j=0;j <NUM_COLORS_PER_THREAD; j++){

			uint32_t vertexID = i*NUM_COLORS_PER_THREAD + j;

			unsigned char myCol = sh_colors[vertexID];

			uint32_t my_offset_start = lowTr_offset[vertexID];
			uint32_t my_offset_end = lowTr_offset[vertexID + 1];

			for(uint32_t k=my_offset_start; k<my_offset_end;k++){
				uint32_t otherVertexID = lowTr_col[k];
				if(floor(int(otherVertexID)/int(blockingSize)) != floor(int(vertexID)/int(blockingSize))){ 
					//potentially conflicting 
					//just vertexID 
					//increment my pucket by one


				}

			}

		}
	}*/
}