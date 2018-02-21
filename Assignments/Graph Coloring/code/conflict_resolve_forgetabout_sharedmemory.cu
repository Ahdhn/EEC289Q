//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>
//#include <cub/block/block_scan.cuh>
//
//using namespace cub;
//
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

using namespace cub;

__global__
void GetneighLen(uint32_t *nodes, int sizeNode,  uint32_t *tr_offset, uint32_t *neighLen, uint32_t *changeColor){
	
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNode; i+=gridDim.x*blockDim.x)
	{
		neighLen[i] = tr_offset[nodes[i]+1]-tr_offset[nodes[i]];
		changeColor[i] = 0;
	}
}



__global__
void FindChangeColor(uint32_t *changeColor, uint32_t sizeNode, uint32_t *nodes, uint32_t *neighLen, uint32_t sizeNeighLen, uint32_t *tr_col_id, uint32_t *tr_offset) {

        int neighbor=-1;
	int neighborOwner = -1;
	printf("sizeNeighLen: %d\n", sizeNeighLen);
        for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNeighLen; i+=blockDim.x*gridDim.x)
        {
		printf("I am thread%d\n", i);
                // calculate who to fetch
                for(int j=0; j<sizeNode; j++)
                {
                        if(i>=neighLen[j] && i<neighLen[j+1])
                        {
                                //j's neighborlist
                                neighborOwner = j;
                                printf("I am thread%d, my neighberOwner=%d nodes[%d]=%d neighbor=%d\n", threadIdx.x, neighborOwner,j,nodes[j], neighbor);
				break;
                        }
                }

		neighbor = tr_col_id[tr_offset[nodes[neighborOwner]]+(i-neighLen[neighborOwner])];
		for(int j=0; j<sizeNode; j++)
		{
			if(neighbor == nodes[j])
			{
				changeColor[neighborOwner]=1;
				break;
			}
		}
        }
}

__global__
void Conflict_assignColor(uint32_t *changeColor, int theColor, unsigned char *color, uint32_t *nodes, uint32_t sizeNode) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNode; i+=blockDim.x*gridDim.x)
	{
		if(changeColor[i]==0)
			color[nodes[i]] = (unsigned char)theColor;
	}
}


__global__
void GenNewNodes(uint32_t *nodes, uint32_t *newNodes, uint32_t *neighLen, uint32_t *newNeighLen, int &sizeNode, uint32_t *changeColor, int &check) {
	if(changeColor[sizeNode]==0)
	{
		check = 1;
		return;
	}
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNode; i+=blockDim.x*gridDim.x)
	{
		if(changeColor[i]+1 == changeColor[i+1])
		{
			newNodes[changeColor[i]] = nodes[i];
			newNeighLen[changeColor[i]] = neighLen[i];
		}
	}
	if(threadIdx.x==0 && blockIdx.x==0)
		sizeNode = changeColor[sizeNode];
}




void conflict_resolve_forgetabout_sharedmemory(uint32_t* conflict_color, // Array of conflict vetices grouped by color
                      uint32_t *conflict_color_offset, // offset of different color on conflit_color 
                      uint32_t *tr_col_id, // CSR of graph, but only lower triangle part
                      uint32_t *tr_offset, // CSR offset of graph, but only lower triangle part
                      uint32_t numVertices, // number of vertices
                      uint32_t size_tr_col,// size of tr_col_id
                      uint32_t numColor,// number of color has been used
                      unsigned char *color,// color array for all vertices
		      uint32_t colorID ) // working space and the size of this array shoudl be BLOCK_THREADS*ITEM_PER_THREAD, o,w it overwrittern information
{
	uint32_t *nodes(NULL), *changeColor(NULL), *nodes1(NULL), *nodes2(NULL), *neighLen1(NULL), *neighLen2(NULL), *scanArray(NULL), *neighLen(NULL);
	uint32_t *newNodes(NULL), *newNeighLen(NULL);
	int start = conflict_color_offset[colorID];
	int end = conflict_color_offset[colorID+1];
	int sizeNode = end-start;
	std::cout<<"start: "<<start<<" end: "<<end<<" sizeNode: "<<sizeNode <<std::endl;
	nodes = conflict_color+start;
	HANDLE_ERROR(cudaMallocManaged(&changeColor, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&nodes1, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&nodes2, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&neighLen1, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&neighLen2, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&scanArray, (sizeNode+1)*sizeof(uint32_t)));
	
	std::cout<<"allocate succeed"<<std::endl;
	for(int i=0; i<sizeNode; i++)
	{
		std::cout<<"nodes["<<i<<"]= "<<nodes[i]<<"  ";
	}
	std::cout<<std::endl;

	GetneighLen<<<1, 32>>>(nodes, sizeNode,  tr_offset, neighLen1, changeColor);
	cudaDeviceSynchronize();

	for(int i=0; i<sizeNode; i++)
	{
		std::cout<<"neighLen1["<<i<<"]= "<<neighLen1[i]<<"  ";
	}
	std::cout<<std::endl;

	for(int i=0; i<sizeNode; i++)
	{
		std::cout<<"changeColor["<<i<<"]= "<<changeColor[i]<<"  ";
	}
	std::cout<<std::endl;

	neighLen = neighLen1;
	int theColor = colorID;
	void  *d_temp_storage = NULL;
    	size_t    temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, neighLen, scanArray, sizeNode);
		cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	int choseL = 0;
	int counter = 1;
	
	while(true) {

		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, neighLen, scanArray, sizeNode);			
		cudaDeviceSynchronize();
		for(int i=0; i<=sizeNode; i++)
		{
			std::cout<<"scan_neighLen["<<i<<"]= "<<neighLen[i]<<"   ";
		}
		std::cout<<std::endl;
		
		FindChangeColor<<<1,32>>>(changeColor, sizeNode, nodes, scanArray, scanArray[sizeNode], tr_col_id, tr_offset);
		cudaDeviceSynchronize();

		Conflict_assignColor<<<1,32>>>(changeColor, theColor, color, nodes, sizeNode);
		cudaDeviceSynchronize();
		theColor = numColor+counter;

		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, changeColor, scanArray, sizeNode);	
		cudaDeviceSynchronize();
		
		choseL = choseL^1;
		if(choseL == 1)
		{
			newNeighLen = neighLen2;
			newNodes = nodes2;
		}
		else
		{
			newNeighLen = neighLen1;
			newNodes = nodes1;
		}
	
		int check = 0;
		GenNewNodes<<<1,32>>>(nodes, newNodes, neighLen, newNeighLen, sizeNode, scanArray, check);
		cudaDeviceSynchronize();
		if(check == 1) break;
		nodes = newNodes;
		neighLen = newNeighLen;
		std::cout<<"counter: "<<counter<<std::endl;
		counter++;
	}	
}
