#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

using namespace cub;

template<
	int BLOCK_THREADS,
	int ITEMS_PER_THREAD>
__global__ 
void conflict_resolve(uint32_t* conflit_color, // Array of conflict vetices grouped by color
		      uint32_t *conflict_color_offset, // offset of different color on conflit_color 
                      uint32_t *tr_col_id, // CSR of graph, but only lower triangle part
                      uint32_t *tr_offset, // CSR offset of graph, but only lower triangle part
                      uint32_t numVertices, // number of vertices
                      uint32_t size_tr_col,// size of tr_col_id
		      uint32_t numColor,// number of color has been used
		      unsigned char *color) // color array for all vertices 
{
	int currentColor = blockIdx.x;
	
	//Each block assign a color
	//Get the size of conflict node
	int start_cc = conflict_color_offset[blockIdx.x];
	int end_cc = conflict_color_offset[blockIdx.x+1];
	int sizeNode = end_cc - start_cc;
	//load conflit vertices into shared_memory, the sharedPool is 32KB, can have 8192 int
	extern __shared__ int sharedPool[];
	int total_shm = 8192;
	int * nodes = sharedPool;
	for(int i = threadIdx.x; i<sizeNode; i+=blockDim.x)
	{
		nodes[i]=conflit_color[start_cc+i];
	}
	__syncthreads();
	int counter = 0;
  while(true){
	if(threadIdx.x==0)
	{
		for(int i=0; i<sizeNode; i++)
			printf("nodes[%d]=%d    ", i, nodes[i]);
		printf("\n");
	}
	if(threadIdx.x==0)
	printf("counter = %d\n", counter);

	total_shm = total_shm - sizeNode;
	int *neighberLen = sharedPool+sizeNode;
//	int currentNode = 0;
	for(int i=threadIdx.x; i<sizeNode; i+=blockDim.x)
	{
		neighberLen[i] = tr_offset[nodes[i]+1]-tr_offset[nodes[i]];
	}
	__syncthreads();
	if(threadIdx.x==0)
	{
		for(int i=0; i<sizeNode; i++)
			printf("neighberLen[%d]=%d   ", i, neighberLen[i]);
		printf("\n");
	}

	//scan neighberLen
	typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
	typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
	typedef BlockScan<int, BLOCK_THREADS, BLOCK_SCAN_RAKING> BlockScanT;

	__shared__ union TempStorage
    	{
        	typename BlockLoadT::TempStorage    load;
        	typename BlockStoreT::TempStorage   store;
        	typename BlockScanT::TempStorage    scan;
    	} temp_storage;
	int data[ITEMS_PER_THREAD];

	BlockLoadT(temp_storage.load).Load(neighberLen, data);
	__syncthreads();
      
	int aggregate;
	BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);
	__syncthreads();

	BlockStoreT(temp_storage.store).Store(neighberLen, data);
	if(threadIdx.x ==0)
	{
		for(int i=0; i<=sizeNode; i++)
			printf("scan neighbor[%d]=%d  ", i, neighberLen[i]);
		printf("aggregate = %d\n", aggregate);
	}
	
	//Load conflit vertices neighbor list
	int *changeColor = neighberLen+sizeNode+1;
	for(int i=threadIdx.x; i<sizeNode; i+=blockDim.x)
		changeColor[i] = 0;

	int neighbor=-1;
	int neighborOwner=-1;
	for(int i=threadIdx.x; i<neighberLen[sizeNode]; i+=blockDim.x)
	{
		// calculate who to fetch
		for(int j=0; j<sizeNode; j++)
		{
			if(i>=neighberLen[j] && i<neighberLen[j+1])
			{
				//j's neighborlist
				neighborOwner = j;
				neighbor = tr_col_id[tr_offset[nodes[j]]+(i-neighberLen[j])];
				printf("I am thread%d, my neighberOwner=%d nodes[%d]=%d neighbor=%d\n", threadIdx.x, neighborOwner,j,nodes[j], neighbor);
			}
		}
	}


	if(threadIdx.x<neighberLen[sizeNode])
	{
		for(int i=0; i<sizeNode; i++)
		{
			printf("nodes[%d]=%d, neighbor=%d\n", i, nodes[i], neighbor);
			if(nodes[i]==neighbor)
				changeColor[neighborOwner]=1;
		}
	}

	if(threadIdx.x==0)
	{
		for(int i=0; i<sizeNode; i++)
			printf("changeColor[%d]=%d  ", i, changeColor[i]);
	}

	// check if conflict resolved for this color
	BlockLoadT(temp_storage.load).Load(changeColor, data);
	__syncthreads();

	BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);
	__syncthreads();

	BlockStoreT(temp_storage.store).Store(changeColor, data);
	if(threadIdx.x ==0)
	{
		for(int i=0; i<=sizeNode; i++)
			printf("scan changeColor[%d]=%d  ", i, changeColor[i]);
		printf("aggregate = %d\n", aggregate);
	}
	if(changeColor[sizeNode]==0)
		return;
	
	// assigne new color to nodes who need to change
	int newColor = numColor+blockIdx.x;
	for(int i=threadIdx.x; i<sizeNode; i+=blockDim.x)
	{
		if(changeColor[i]==changeColor[i+1])
		{
			color[nodes[i]]=(unsigned char)currentColor;
		}
			
	}
	currentColor = newColor;

	int *newNodes = changeColor+sizeNode+1;
	for(int i=threadIdx.x; i<sizeNode; i+=blockDim.x)
	{
		if(changeColor[i]+1==changeColor[i+1])
		{
			newNodes[changeColor[i]]=nodes[i];
		}
	}
	__syncthreads();
	if(threadIdx.x==0)
	{
		for(int i=0; i<changeColor[sizeNode]; i++)
			printf("newNodes[%d]=%d  ", i, newNodes[i]);
		printf("\n");
		
	}

	for(int i=threadIdx.x; i<changeColor[sizeNode]; i+=blockDim.x)
		nodes[i]=newNodes[i];
//	nodes = newNodes;
	sizeNode = changeColor[sizeNode];
	counter++;
  }	
			
			
	
}
