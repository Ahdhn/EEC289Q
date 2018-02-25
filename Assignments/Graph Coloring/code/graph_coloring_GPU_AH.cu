#include <sstream>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverSp.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>

//Error handling micro, wrap it around function whenever possible
static void HandleError(cudaError_t err, const char *file, int line) {
   if (err != cudaSuccess) {
      printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      //system("pause");
      exit(EXIT_FAILURE);
   }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#include "validate.h"
#include "serial.h"
#include "utility.h"
#include "coloring.cu"
#include "cuda_query.cu"
#include "conflict_resolve.cu"
//#include "conflict_resolve_forgetabout_sharedmemory.cu"
#include "conflict_resolve_forgetabout_sharedmemory1.cu"
#include "conflict_graph.cu"

void GraphColoringGPU(const char filename[], int**&color_int)
{
   bool* graph;
   int V;     
   uint32_t blockingSize = 5;//TODO
   uint32_t numNNZ=0;
   uint32_t NumRow=0; 
   uint32_t numNNZ_blocked = 0;


   //1) Read graph
   if (std::string(filename).find(".col") != std::string::npos){
     ReadColFile(filename, &graph, &V, &numNNZ,&NumRow, blockingSize, &numNNZ_blocked);
   } else if (std::string(filename).find(".mm") != std::string::npos){
     ReadMMFile(filename, &graph, &V, &numNNZ,&NumRow,blockingSize, &numNNZ_blocked);
   } else{
      std::cout<<" Invalid file formate!!"<<std::endl;
      exit(EXIT_FAILURE);
   }      
   /***********************************************************************/  

   //2) Allocate memory (on both sides)
   uint32_t *col_id(NULL),*offset(NULL);   

   HANDLE_ERROR(cudaMallocManaged(&col_id, numNNZ_blocked*sizeof(uint32_t)));   
   HANDLE_ERROR(cudaMallocManaged(&offset, (NumRow +1)*sizeof(uint32_t)));   


   unsigned char* color;
   HANDLE_ERROR(cudaMallocManaged(&color, NumRow*sizeof(unsigned char)));   
   memset(color, 0, NumRow);   
   int*numColor;
   HANDLE_ERROR(cudaMallocManaged(&numColor, sizeof(int)));
   memset(numColor, 0, 1);   
   uint32_t*numberVerticesPerColor;//allocate as if each vertex will take different color 
   HANDLE_ERROR(cudaMallocManaged(&numberVerticesPerColor, NumRow*sizeof(uint32_t)));
   memset(numberVerticesPerColor, 0, NumRow);   
   /***********************************************************************/
   

   //3) Get graph in CSR format 
   //getCSR(numNNZ, NumRow, graph, col_id, offset);   
   uint32_t maxLeftout=0; //maxLeftout the maximum number of vertices j connected to i that are left out when constructing blocked CSR (used to allocate conflicting graph)
   getBlockedCSR(NumRow, graph, col_id, offset, blockingSize, maxLeftout);   
   
   

   //CUB parameters 
   void *d_temp_storage = NULL;
   size_t temp_storage_bytes = 0;
   cub::CachingDeviceAllocator  g_allocator(true);

   int numBlocks(1), numThreads(1);
   numBlocks = (NumRow/blockingSize) + 1;
   numThreads = blockingSize;

   //A) Do local colring 
   uint32_t max_NNZ_per_block= maxNNZ_per_segment(offset, NumRow, blockingSize);        
   uint32_t shrd_mem = numThreads*sizeof(bool) /*+ max_NNZ_per_block*sizeof(uint32_t)*/;  
   std::cout<<"NumRow= "<<NumRow <<"   max_NNZ_per_block= "<<max_NNZ_per_block << " shrd_mem= "<<shrd_mem  << std::endl;
   std::cout<<" numThreads= "<<numThreads<< " numBlocks= "<< numBlocks<<std::endl;
   coloring <<<numBlocks, numThreads, shrd_mem>>> (NumRow, col_id, offset, color, numColor,numberVerticesPerColor, max_NNZ_per_block);
   cudaDeviceSynchronize();     

   HANDLE_ERROR(cudaFree(offset));//free what you dont need 
   HANDLE_ERROR(cudaFree(col_id));

   //B) Get conflicting graph
   uint32_t *conflict_vertices(NULL), *conflict_offset(NULL);   
   HANDLE_ERROR(cudaMallocManaged(&conflict_offset, ((*numColor) +2)*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&conflict_vertices, NumRow*sizeof(uint32_t)));   
   HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes, numberVerticesPerColor,conflict_offset, (*numColor)+2));
   HANDLE_ERROR(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));   
   HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes, numberVerticesPerColor,conflict_offset, (*numColor)+2));
   cudaDeviceSynchronize();
   HANDLE_ERROR(cudaFree(numberVerticesPerColor));
   uint32_t items_in_sh_mem = min(40*1024, NumRow);//same as bytes of sh mem since colors are unisgned char 
   conflict_graph<<< 1, (*numColor), items_in_sh_mem>>> (NumRow, color, numColor,conflict_vertices, conflict_offset, items_in_sh_mem);   
   cudaDeviceSynchronize();     





   //C) Resolving conflict 
   uint32_t *lowTr_col(NULL), *lowTr_offset(NULL);
   HANDLE_ERROR(cudaMallocManaged(&lowTr_col, numNNZ/2*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&lowTr_offset, (NumRow+1)*sizeof(uint32_t)));
   getLowTrCSR(numNNZ, NumRow, graph, lowTr_col, lowTr_offset);

   int newNumColor = (*numColor);
   for(int i=1; i <= (*numColor); i++){
      newNumColor = conflict_resolve_forgetabout_sharedmemory1(conflict_vertices, conflict_offset, lowTr_col, lowTr_offset, NumRow, numNNZ/2, newNumColor, color, i, numBlocks, numThreads);
   }

   cudaDeviceSynchronize();
   /***********************************************************************/

    //5.5) Validate parallel LOCAL solution    
   //printf("Parallel LOCAL solution has %d colors\n", CountColors(V, color));
   //printf("Valid LOCAL coloring: %d\n\n", IsValidColoring_Blocked(graph, V, color, blockingSize));  
   //PrintSolution(color,V);  
   //exit(0);   

   //6) Validate parallel solution 
   //printf("Parallel solution has %d colors\n", CountColors(V, color));
   //printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);

   color_int = new int*[NumRow];
   for(uint32_t i=0;i<NumRow;i++){
      color_int[i]=new int[1];
      color_int[i][0] = int(color[i]);
   }
}


int main(int argc, char* argv[]){

   cuda_query(3); //Set the deivde number here 

   if(argc != 2){
      std::cout<<"  Usage ./graphGPU INPUTFILE"<<std::endl;
      std::cout<<"input files can be found under input/ "<<std::endl;
      exit(EXIT_FAILURE);
   }

   int** color = NULL;

   GraphColoringGPU(argv[1],color);

   /*for(int i=0;i<12;i++){
      std::cout<< " color["<< i<<"][0]= "<< color[i][0]<<std::endl;
   }*/


   /*bool* graph;
   int V;     
   const uint32_t blockingSize = 5;//TODO
   uint32_t numNNZ=0;
   uint32_t NumRow=0; 
   uint32_t numNNZ_blocked = 0;


   //1) Read graph
   if (std::string(argv[1]).find(".col") != std::string::npos){
     ReadColFile(argv[1], &graph, &V, &numNNZ,&NumRow, blockingSize, &numNNZ_blocked);
   } else if (std::string(argv[1]).find(".mm") != std::string::npos){
     ReadMMFile(argv[1], &graph, &V, &numNNZ,&NumRow,blockingSize, &numNNZ_blocked);
   } else{
      std::cout<<" Invalid file formate!!"<<std::endl;
      exit(EXIT_FAILURE);
   }      
  

   //2) Allocate memory (on both sides)
   uint32_t *col_id(NULL),*offset(NULL);   

   HANDLE_ERROR(cudaMallocManaged(&col_id, numNNZ_blocked*sizeof(uint32_t)));   
   HANDLE_ERROR(cudaMallocManaged(&offset, (NumRow +1)*sizeof(uint32_t)));   


   unsigned char* color;
   HANDLE_ERROR(cudaMallocManaged(&color, NumRow*sizeof(unsigned char)));   
   memset(color, 0, NumRow);   
   int*numColor;
   HANDLE_ERROR(cudaMallocManaged(&numColor, sizeof(int)));
   memset(numColor, 0, 1);   
   uint32_t*numberVerticesPerColor;//allocate as if each vertex will take different color 
   HANDLE_ERROR(cudaMallocManaged(&numberVerticesPerColor, NumRow*sizeof(uint32_t)));
   memset(numberVerticesPerColor, 0, NumRow);   
  
   

   //3) Get graph in CSR format 
   //getCSR(numNNZ, NumRow, graph, col_id, offset);   
   uint32_t maxLeftout=0; //maxLeftout the maximum number of vertices j connected to i that are left out when constructing blocked CSR (used to allocate conflicting graph)
   getBlockedCSR(NumRow, graph, col_id, offset, blockingSize, maxLeftout);   
   //printCSR(numNNZ_blocked,NumRow,col_id, offset);   
   //exit(0);
   //getLowTrCSR(numNNZ, NumRow, graph, lowTr_col, lowTr_offset);
   //printCSR(numNNZ/2, NumRow, lowTr_col, lowTr_offset);
   

   //CUB parameters 
   void *d_temp_storage = NULL;
   size_t temp_storage_bytes = 0;
   cub::CachingDeviceAllocator  g_allocator(true);

   int numBlocks(1), numThreads(1);
   numBlocks = (NumRow/blockingSize) + 1;
   numThreads = blockingSize;
   //if(blockingSize < 1024){//if it is less than NumRow vertex, then launch one block 
   //   numBlocks = 1;
   //   numThreads = blockingSize;
   //}else{//otherwise, launch as many as 1024-blocks as you need      
   //   numBlocks = (blockingSize/1024) + 1;
   //   numThreads = 1024;
   //}


   //A) Do local colring 
   uint32_t max_NNZ_per_block= maxNNZ_per_segment(offset, NumRow, blockingSize);        
   uint32_t shrd_mem = numThreads*sizeof(bool);
   //+ max_NNZ_per_block*sizeof(uint32_t);  
   //std::cout<<"NumRow= "<<NumRow <<"   max_NNZ_per_block= "<<max_NNZ_per_block << " shrd_mem= "<<shrd_mem  << std::endl;
   //std::cout<<" numThreads= "<<numThreads<< " numBlocks= "<< numBlocks<<std::endl;
   coloring <<<numBlocks, numThreads, shrd_mem>>> (NumRow, col_id, offset, color, numColor,numberVerticesPerColor, max_NNZ_per_block);
   cudaDeviceSynchronize();     

   HANDLE_ERROR(cudaFree(offset));//free what you dont need 
   HANDLE_ERROR(cudaFree(col_id));

   //B) Get conflicting graph
   uint32_t *conflict_vertices(NULL), *conflict_offset(NULL);   
   HANDLE_ERROR(cudaMallocManaged(&conflict_offset, ((*numColor) +2)*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&conflict_vertices, NumRow*sizeof(uint32_t)));   
   HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes, numberVerticesPerColor,conflict_offset, (*numColor)+2));
   HANDLE_ERROR(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));   
   HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage,temp_storage_bytes, numberVerticesPerColor,conflict_offset, (*numColor)+2));
   cudaDeviceSynchronize();
   HANDLE_ERROR(cudaFree(numberVerticesPerColor));
   uint32_t items_in_sh_mem = min(40*1024, NumRow);//same as bytes of sh mem since colors are unisgned char 
   conflict_graph<<< 1, (*numColor), items_in_sh_mem>>> (NumRow, color, numColor,conflict_vertices, conflict_offset, items_in_sh_mem);   
   cudaDeviceSynchronize();     

   //C) Resolving conflict 
   uint32_t *lowTr_col(NULL), *lowTr_offset(NULL);
   HANDLE_ERROR(cudaMallocManaged(&lowTr_col, numNNZ/2*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&lowTr_offset, (NumRow+1)*sizeof(uint32_t)));
   getLowTrCSR(numNNZ, NumRow, graph, lowTr_col, lowTr_offset);

//   for(int index=0; index<NumRow; index++)
//   {
//	if(graph[100*NumRow+index]!=0)
//   	std::cout<<"graph[100*NumRow+"<<index<<"]:"<<graph[100*NumRow+index]<<" ";
//   }
//   for(int index=0; index<10; index++)
//   {
//   	int nodenow = lowTr_offset[index+1]-lowTr_offset[index];
//	std::cout<<"index: "<<index<<std::endl;
//   	for(int i=0; i<nodenow; i++)
//   	{
//		std::cout<<i<<":"<<lowTr_col[lowTr_offset[index]+i]<<"  ";
//   	}
//   	std::cout<<std::endl;
//   }


//   int node149Len = lowTr_offset[149+1]-lowTr_offset[149];
//   for(int i=0; i<node149Len; i++)
//   {
//	std::cout<<i<<":"<<lowTr_col[lowTr_offset[149]+i]<<"  ";
//   }
//   std::cout<<std::endl;

   int newNumColor = (*numColor);
   for(int i=1; i <= (*numColor); i++){
      newNumColor = conflict_resolve_forgetabout_sharedmemory1(conflict_vertices, conflict_offset, lowTr_col, lowTr_offset, NumRow, numNNZ/2, newNumColor, color, i, numBlocks, numThreads);
   }

   cudaDeviceSynchronize();


    //5.5) Validate parallel LOCAL solution    
   //printf("Parallel LOCAL solution has %d colors\n", CountColors(V, color));
   //printf("Valid LOCAL coloring: %d\n\n", IsValidColoring_Blocked(graph, V, color, blockingSize));  
   //PrintSolution(color,V);  
   //exit(0);   

   printf("\n*********************************\n");
   //int* colorInt;
   //HANDLE_ERROR(cudaMallocManaged(&colorInt, NumRow*sizeof(int)));   
   //memset(colorInt, 0, NumRow);   

   //6) Validate parallel solution 
   printf("Parallel solution has %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);  

    //GreedyColoring(graph, V, &color);
   //printf("\n*********************************\n");
   //printf("Greedy solution has %d colors\n", CountColors(V, colorInt));
   //printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, colorInt));
   //PrintSolution(colorInt,V);


   //printf("\n*********************************\n");
   //7) Color Vertices on CPU
   //GraphColoring(graph, V, &colorInt);
   //printf("Brute-foce solution has %d colors\n", CountColors(V, colorInt));   
   //printf("Valid coloring: %d\n", IsValidColoring(graph, V, colorInt));  
   */


   return 0;
}
