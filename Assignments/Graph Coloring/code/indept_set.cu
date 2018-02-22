__device__ __forceinline__ 
//void indept_set(uint32_t NumRow, uint32_t numNNZ, uint32_t *col_id, uint32_t *offset, bool*set, bool useMax, unsigned char* color, int&numColored){
void indept_set(const uint32_t tid,
	            const uint32_t my_offset_start,
	            const uint32_t my_offset_end,	            
	            const uint32_t start_row,
	            const uint32_t end_row,
                uint32_t * const sh_col_id,    
                const uint32_t NumRow,
                int&numColored,
                bool useMax,
                bool*&sh_set
                ){		

	if(tid < NumRow){//only if my vertex was not colored before

		bool inSet = true;
		for(uint32_t i = my_offset_start; i < my_offset_end; i++){//read inside the shared col id
			//avoid thread diveregnce by arthematics 				
			uint32_t j = sh_col_id[i];
			if(useMax){
				//my vertex is in the independent set if it is the "maximum" of its neighbours
				inSet = inSet && (j < start_row || j >= end_row || //j not in this blocks processed vertices
					              tid < j ||//if j id (or rand num) is greater than i, ignore               
					              sh_set[j]    //if j is been already in the set, ignore
					              );

				              
			}else{				
				//my vertex is in the independent set if it is the "minimum" of its neighbours				
				inSet = inSet && (j < start_row || j >= end_row || //j not in this blocks processed vertices
					              tid > j ||//if j id (or rand num) is less than i, ignore               
					              sh_set[j]  //if j is been already in the set, ignore
					              );
			}
		}

		sh_set[tid] = inSet;

		if(sh_set[tid]){
			atomicAdd(&numColored, 1);//if it is in the independent set, then it will be colored
		}


		__syncthreads();
		if(threadIdx.x == 0 && blockIdx.x == 0) {
			printf("\n The ind set is: \n");
			for(int i=0;i<NumRow;i++){
				if(sh_set[i]){
					printf(" %d ", i );
				}
			}
			printf("\n");			
		}

	}
}



/*__device__ __forceinline__ 
void indept_set(uint32_t NumRow, uint32_t numNNZ, uint32_t *col_id, uint32_t *offset, bool*set, bool useMax, unsigned char* color, int&numColored){

	//Create independent set 
	//TODO optimize for memory  

	//Each thread will work on one element 
	//Operate one global memory all the way

	int row = blockIdx.x * blockDim.x + threadIdx.x;	

	if(row < NumRow && !set[row]){//only if my vertex was not colored before 

		uint32_t row_start = offset[row];
		uint32_t row_end = offset[row + 1]; //this one is cached already (next thread reads it)
		
		bool inSet = true;
		for(uint32_t i=row_start; i<row_end; i++){
			if(!useMax){
				//my vertex is in the independent set if it is the "minimum" of its neighbours				
				inSet = inSet && (set[col_id[i]] || row < col_id[i]); //avoid thread diveregnce by arthematics 				
			}else{				
				//my vertex is in the independent set if it is the "maximum" of its neighbours				
				inSet = inSet && (set[col_id[i]] || row > col_id[i]);				
			}
		}
		set[row] = inSet;


		if(set[row]){
			atomicAdd(&numColored, 1);//if it is in the independent set, then it will be colored
		}


		//__syncthreads();
		//if(threadIdx.x == 0 && blockIdx.x == 0) {
		//	printf("\n The ind set is: \n");
		//	for(int i=0;i<NumRow;i++){
		//		if(set[i]){
		//			printf(" %d ", i );
		//		}
		//	}
		//	printf("\n");			
		//}

	}
}*/