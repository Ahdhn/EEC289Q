#include <cub/cub.cuh>

__global__ void predicates(uint32_t *indset, uint32_t numElement, uint32_t *col_id, uint32_t size_col, uint32_t * value, uint32_t size_value, uint32_t *predicate1, uint32_t *predicate2, uint32_t*predicate3 ){

        //Create filter: input independent set. output new CSR 
	// predicate1 are all 0 by default
	// predicate1 is array of predicate if each element in col_id is in indset, if yes, 1, o.w 0
	// predicate2 is array of predicate if each element in col_id is in indset, if yes, 0, o.w 1
	// scan1 is from predicate1 with InclusiveSum
	// scan2 is from predicate2 with ExclusiveSum
	for(int i=threadIdx.x; i<size_col; i+=blockDim.x*gridDim.x)
	{
		predicate1[i] = 0;
		for(int j=0; j<numElement; j++)
		{
			if(indset[j]==col_id[i])
			{
				predicate1[i] = 1;
				break;
			}
		}
		predicate2[i] = predicate1[i]^1;
	}

	for(int i=threadIdx.x; i<size_value; i+=blockDim.x*gridDim.x)
	{
		predicate3[i] = 1;
		for(int j=0; j<numElement; j++)
		{
			if(indset[j]==value[i])
			{
				predicate3[i]=0;
				break;
			}
		}
	}
}	
__device__ __forceinline__ void filter(uint32_t *indset, uint32_t numElement, uint32_t *col_id, uint32_t *offset, uint32_t *value, uint32_t size_col, uint32_t size_offset, uint32_t *scan1, uint32_t *scan2, uint32_t *new_value, uint32_t *new_col, uint32_t *new_offset, uint32_t size_new_col, uint32_t size_new_offset){

	//filter col_id;
	for(int i=threadIdx.x; i<size_col; i+=blockDim.x*gridDim.x)
	{	
		if(scan2[i+1]!=scan2[i])
		{
			new_col[scan2[i+1]-1]=col_id[i];
		}
	}

	//filter value;
	
	}

}
	// scan predicate1 and predicate2
        //// Declare, allocate, and initialize device-accessible pointers for input and output
        //int  num_items;      // e.g., 7
        //int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
        //int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
        //...
        //// Determine temporary device storage requirements for inclusive prefix sum
        //void     *d_temp_storage = NULL;
        //size_t   temp_storage_bytes = 0;
        //cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        //// Allocate temporary storage for inclusive prefix sum
        //cudaMalloc(&d_temp_storage, temp_storage_bytes);
        //// Run inclusive prefix sum
        //cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
        //// d_out <-- [8, 14, 21, 26, 29, 29, 38]	



