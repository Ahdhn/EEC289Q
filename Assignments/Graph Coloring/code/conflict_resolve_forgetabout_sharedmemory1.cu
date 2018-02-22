#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

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
void FindChangeColor(uint32_t *changeColor, uint32_t sizeNode, uint32_t *nodes, int *wir, int *lbs, uint32_t sizeLbs, uint32_t *tr_col_id, uint32_t *tr_offset, int theColor, unsigned char *color) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		int neighborOwner = lbs[i];
		int neighbor = tr_col_id[tr_offset[nodes[lbs[i]]] + wir[i]];
		if(threadIdx.x==0)
		{
			for(int x=0; x<sizeNode; x++)
				printf("nodes[%d]=%d  ", x, nodes[i]);
			printf("\n");
			printf("theColor: %u\n", (unsigned char)theColor);
		}
			printf("\n");
			printf("thread %d, my neighbor is %d, neighborOwner is %d, \n", i, neighbor, neighborOwner);
			printf("thread %d, nodes[%d] is %d and color[%d] is %u\n", i, neighborOwner, nodes[neighborOwner], neighbor, color[neighbor]);
		
		if(color[neighbor] == (unsigned char)theColor)
		{
			changeColor[neighborOwner]=1;
		}
	}
	if(threadIdx.x==0)
                {
                        for(int i=0; i<sizeNode; i++)
                                printf("changeColor[%d]=%d   ", i,changeColor[i] );
                        printf("\n");
                }

}

__global__
void Conflict_assignColor(uint32_t *changeColor, int theColor, unsigned char *color, uint32_t *nodes, uint32_t sizeNode) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNode; i+=blockDim.x*gridDim.x)
	{
		if(changeColor[i]==1)
			color[nodes[i]] = (unsigned char)theColor;
	}
}


__global__
void GenNewNodes(uint32_t *nodes, uint32_t *newNodes, uint32_t *neighLen, uint32_t *newNeighLen, int sizeNode, int *changeColor) {

	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeNode; i+=blockDim.x*gridDim.x)
	{
		if(changeColor[i]+1 == changeColor[i+1])
		{
			newNodes[changeColor[i]] = nodes[i];
			newNeighLen[changeColor[i]] = neighLen[i];
		}
	}
}

__global__
void WorkItemRank(int *scan, int *lbs, int *wir, int sizeLbs) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		wir[i] = i - scan[lbs[i]];
	}
}



int conflict_resolve_forgetabout_sharedmemory1(uint32_t* conflict_color, // Array of conflict vetices grouped by color
                      uint32_t *conflict_color_offset, // offset of different color on conflit_color 
                      uint32_t *tr_col_id, // CSR of graph, but only lower triangle part
                      uint32_t *tr_offset, // CSR offset of graph, but only lower triangle part
                      uint32_t numVertices, // number of vertices
                      uint32_t size_tr_col,// size of tr_col_id
                      uint32_t numColor,// number of color has been used
                      unsigned char *color,// color array for all vertices
		      uint32_t colorID ) // working space and the size of this array shoudl be BLOCK_THREADS*ITEM_PER_THREAD, o,w it overwrittern information
{
        standard_context_t context;
	uint32_t *nodes(NULL), *changeColor(NULL), *nodes1(NULL), *nodes2(NULL), *neighLen1(NULL), *neighLen2(NULL),  *neighLen(NULL);
	uint32_t *newNodes(NULL), *newNeighLen(NULL);
	int start = conflict_color_offset[colorID];
	int end = conflict_color_offset[colorID+1];
	int sizeNode = end-start;
	if(sizeNode==0) return numColor;
	std::cout<<"start: "<<start<<" end: "<<end<<" sizeNode: "<<sizeNode <<std::endl;
	nodes = conflict_color+start;
	HANDLE_ERROR(cudaMallocManaged(&changeColor, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&nodes1, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&nodes2, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&neighLen1, sizeNode*sizeof(uint32_t)));
	HANDLE_ERROR(cudaMallocManaged(&neighLen2, sizeNode*sizeof(uint32_t)));
        int *lbs(NULL), *wir(NULL);
	int *scanArray(NULL);
	HANDLE_ERROR(cudaMallocManaged(&scanArray, (sizeNode+1)*sizeof(int)));
	
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
		std::cout<<"changeColor["<<i<<"]= "<<changeColor[i]<<"  ";
	}
	std::cout<<std::endl;
		for(int i=0; i<numVertices; i++)
		{
			printf("color[%d]=%u  ", i, color[i]);
		}
		std::cout<<std::endl;

	int choseL = 0;
	int counter = 1;
	neighLen = neighLen1;
	int theColor = colorID+1;
	void  *d_temp_storage = NULL;
    	size_t    temp_storage_bytes = 0;
	DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, neighLen, scanArray, sizeNode+1);
	HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	while(true) {
		for(int i=0; i<sizeNode; i++)
		{
			std::cout<<"neighLen["<<i<<"]= "<<neighLen[i]<<"  ";
		}
		std::cout<<std::endl;

		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, neighLen, scanArray, sizeNode+1);			
		cudaDeviceSynchronize();
		for(int i=0; i<sizeNode+1; i++)
		{
			std::cout<<"scan_neighLen["<<i<<"]= "<<scanArray[i]<<"   ";
		}
		std::cout<<std::endl;
		
		int sizeLbs = scanArray[sizeNode];

		if(counter == 1) {
        		HANDLE_ERROR(cudaMallocManaged(&lbs, sizeLbs*sizeof(int)));
        		HANDLE_ERROR(cudaMallocManaged(&wir, sizeLbs*sizeof(int)));
		}

		load_balance_search(sizeLbs, scanArray, sizeNode, lbs, context);
		cudaDeviceSynchronize();
		for(int i=0; i<sizeLbs; i++)
		{
			std::cout<<"lbs["<<i<<"]= "<<lbs[i]<<"   ";
		}
		std::cout<<std::endl;
		
		WorkItemRank<<<1,32>>>(scanArray, lbs, wir, sizeLbs);		
		cudaDeviceSynchronize();
		for(int i=0; i<sizeLbs; i++)
		{
			std::cout<<"WIR["<<i<<"]= "<<wir[i]<<"   ";
                }
		std::cout<<std::endl;
		
                FindChangeColor<<<1,32>>>(changeColor, sizeNode, nodes, wir, lbs, sizeLbs, tr_col_id, tr_offset, theColor, color);
		cudaDeviceSynchronize();

		theColor = numColor+counter;
//		if(counter!=0)
			Conflict_assignColor<<<1,32>>>(changeColor, theColor, color, nodes, sizeNode);
		cudaDeviceSynchronize();

		for(int i=0; i<numVertices; i++)
		{
			 printf("color[%d]=%u  ", i, color[i]);
		}
		std::cout<<std::endl;

		DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, changeColor, scanArray, sizeNode+1);	
		cudaDeviceSynchronize();

		for(int i=0; i<sizeNode; i++)
		{
			std::cout<<"changeColor["<<i<<"]= "<<changeColor[i]<<"  ";
		}
		std::cout<<std::endl;	
		
		for(int i=0; i<sizeNode+1; i++)
		{
			std::cout<<"scan_changeColor["<<i<<"]= "<<scanArray[i]<<"   ";
		}
		std::cout<<std::endl;

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
	
		std::cout<<"sizeNode: "<< sizeNode<<std::endl;
		GenNewNodes<<<1,32>>>(nodes, newNodes, neighLen, newNeighLen, sizeNode, scanArray);
		cudaDeviceSynchronize();
		sizeNode = scanArray[sizeNode];
		if(sizeNode == 0) break;

		printf("new sizeNode: %d\n", sizeNode);
		nodes = newNodes;
		for(int i=0; i<sizeNode; i++)
		{
			std::cout<<"newNode["<<i<<"]=  "<<nodes[i]<<"   ";
		}
		std::cout<<std::endl;
		neighLen = newNeighLen;
		for(int i=0; i<sizeNode; i++)
		{
			std::cout<<"newNeighLen["<<i<<"]=  "<<neighLen[i]<<"   ";
		}
		std::cout<<std::endl;
		std::cout<<"counter: "<<counter<<std::endl;
		counter++;
		std::cout<<std::endl;
		std::cout<<std::endl;
	}	
	std::cout<<counter-1<<" color is added, total number of color is "<<theColor-1<<std::endl;
	return theColor-1;
}
