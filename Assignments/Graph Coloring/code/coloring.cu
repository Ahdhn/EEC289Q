#include "assign_color.cu"
#include "indept_set.cu"
//#include "filter.cu"

#define NUM_COLOR_PER_THREAD 1 //this is changed to be more than one, then we need to move my_offset_start and my_offset_end to be in shared memory instead 

 __device__ int numColored = 0;  

__global__ void coloring(uint32_t NumRow, //number of vertices (= number of rows in adjacency matrix)
	                     uint32_t *col_id, //the column id in the CSR format 
	                     uint32_t *offset, //the row offset in the CSR
	                     unsigned char* color, //the color of the vertices (output)	                     
	                     int*numColors,//the total colors that has been assigned 	                     
	                     uint32_t*numberVerticesPerColor
	                     ){
	
	                     

	//Every block will own number of vertices equal to blockDim.x 
	//We need to move the data (col_id, offset) for the block's vertices 
	//The color of each vertex will be stored in a register (since one vertex is handeled by one thread)
	//and will write just once at the end (coalesced write)
	if(threadIdx.x == 0 && blockIdx.x == 0){
		(*numColors)=0;
	}
		
	unsigned char my_thd_colors [NUM_COLOR_PER_THREAD];//the colos that this thread will do
#pragma unroll 	
	for(uint32_t i=0; i < NUM_COLOR_PER_THREAD; i++){ my_thd_colors[i]=0; }//initialize with no color 

	extern __shared__ bool shrd_ptr[];
	bool* sh_set = shrd_ptr;// the independent set of this block (has to be in shared memory because thread/vertex i needs to know if thread/vertex j is in the set or not when getting the independant set)
	
	uint32_t * sh_col_id = (uint32_t*)&shrd_ptr[NUM_COLOR_PER_THREAD*blockDim.x]; //has length of number of nnz elements for this block's vertices 	

	uint32_t my_offset_start, my_offset_end;//the offset of of the row this thread is responsible of 
	                                        //we read the start and end because we need them to iterate over 
	                                        //the columns of this row

	const uint32_t start_row = blockIdx.x * blockDim.x; //starting point of my block (in the offset array) 
	                                                    //starting row in the adjacency matris
	                                                    //this block is resposible of vertices with id from 
	                                                    //start up to start + blockDim.x
	const uint32_t end_row = blockIdx.x * blockDim.x + blockDim.x;

	const uint32_t tid = start_row + threadIdx.x;//equal to this thread's vertex id

#pragma unroll 
	for(uint32_t i= threadIdx.x; i + start_row < end_row ; i+=blockDim.x) {//make sure that we 		                                                               //don't go beyound offste []
		my_offset_start = offset[i + start_row];
		my_offset_end = offset[i + 1 + start_row];//hopefully this is cached 
	}
#pragma unroll 
	for(uint32_t i= threadIdx.x; i< NUM_COLOR_PER_THREAD*blockDim.x; i+=blockDim.x){
		sh_set[i] = false;
	}


	//count the number of nnz element owned by this block
	//reduce within a block 
	__shared__ uint32_t myNNZ; 
	__shared__ uint32_t start_col_id;

	if(threadIdx.x == blockDim.x -1){
		myNNZ = my_offset_end;
	}
	__syncthreads(); //make sure myNNZ is updated 
	if(threadIdx.x == 0){
		myNNZ -= my_offset_start;
		start_col_id = my_offset_start;
	}
	__syncthreads(); //make sure myNNZ is updated 
	my_offset_start -= start_col_id; //decremented so that we can use them to index sh_col_id directly
	my_offset_end -= start_col_id;

	//move the col_id (coalesced read)
#pragma unroll 	
	for(uint32_t i = threadIdx.x; i < myNNZ; i+=blockDim.x){
		sh_col_id[i] = col_id[start_col_id + i]; 

	}
	__syncthreads(); //make sure all col_id are loaded

	//*****************************************************//

	unsigned char currentColor = 1;	

	while(numColored < blockDim.x * NUM_COLOR_PER_THREAD){ //loop untill all this blocks vertices are colored
		
		numColored = 0;
		indept_set(tid, my_offset_start, my_offset_end, start_row, end_row, sh_col_id, NumRow,
			       numColored, /*currentColor % 2 ==*/ 0, sh_set);
		__syncthreads();	
		//if(blockIdx.x == 0 && threadIdx.x == 0){
		//	printf("\n numColored= %d\n",numColored);
		//}
		assign_color(tid, currentColor, NumRow, sh_set, my_thd_colors, 0);		
		__syncthreads();				
		currentColor++;		
	}		
	//move color to global memory
	if(tid < NumRow){
		color[tid] = my_thd_colors[0];				
		atomicMax(numColors,int(color[tid]));
		atomicAdd(&numberVerticesPerColor[my_thd_colors[0]], uint32_t(1));
	}
}


/*__device__ int numColored = 0;

__global__ void coloring(uint32_t NumRow, //number of vertices (= number of rows in adjacency matrix)
	                     uint32_t numNNZ, //number of non zero entry of the adjacency matrix
	                     uint32_t *col_id, //the column id in the CSR format 
	                     uint32_t *offset, //the row offset in the CSR
	                     unsigned char* color, //the color of the vertices (output)
	                     bool*set //the indepent set (global memory)
	                     ){

	unsigned char currentColor = 1;
	

	while(numColored < NumRow){//loop untill all vertices are colored 

		indept_set(NumRow, numNNZ, col_id, offset, set, currentColor%2 == 1, color, numColored);
		__syncthreads();		
		assign_color(currentColor, NumRow, set,color);
		__syncthreads();
		//filter();
		currentColor++;	
	}		
}*/