#include <sstream>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
#include "hello.cu"


int main(int argc, char* argv[])
{
	cuda_query(3); //Set the deivde number here 

	if(argc != 2){
		std::cout<<"  Usage ./graphGPU INPUTFILE"<<std::endl;
		std::cout<<"input files can be found under input/ "<<std::endl;
		exit(EXIT_FAILURE);
	}

   bool* graph;
   int V;  
   uint32_t numNNZ=0;
   uint32_t NumRow=0; //same as V


   //1) Read graph
   if (std::string(argv[1]).find(".col") != std::string::npos){
     ReadColFile(argv[1], &graph, &V, &numNNZ,&NumRow);
   } else if (std::string(argv[1]).find(".mm") != std::string::npos){
     ReadMMFile(argv[1], &graph, &V, &numNNZ,&NumRow);
   } else{
   	std::cout<<" Invalid file formate!!"<<std::endl;
   	exit(EXIT_FAILURE);
   }
   /***********************************************************************/

   //2) Allocate memory (on both sides)
   uint32_t *col_id(NULL),*offset(NULL);   
   HANDLE_ERROR(cudaMallocManaged(&col_id, numNNZ*sizeof(uint32_t)));
   
   //last entry will be = numNonZero (so that we have always a pointer
   //to the first and last id for each row with no need for if statments)   
   HANDLE_ERROR(cudaMallocManaged(&offset, (NumRow +1)*sizeof(uint32_t)));
   /***********************************************************************/

   //3) Get graph in CSR format 
   getCSR(numNNZ, NumRow, graph, col_id, offset);
   printCSR(numNNZ,NumRow,col_id, offset);
   /***********************************************************************/

   uint32_t *lowTr_col(NULL), *lowTr_offset(NULL);
   HANDLE_ERROR(cudaMallocManaged(&lowTr_col, numNNZ/2*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&lowTr_offset, (NumRow+1)*sizeof(uint32_t)));
   getLowTrCSR(numNNZ, NumRow, graph, lowTr_col, lowTr_offset);
   printCSR(numNNZ/2, NumRow, lowTr_col, lowTr_offset);


   //5) Color Vertices in paralllel
   unsigned char* color;
   HANDLE_ERROR(cudaMallocManaged(&color, NumRow*sizeof(unsigned char)));
   memset(color, 0, NumRow );   

   // Two blocks, given conflict vertices array and its offset, resolve the conflict
   uint32_t *conflict_vertices(NULL), *conflict_offset(NULL);
   HANDLE_ERROR(cudaMallocManaged(&conflict_vertices, 4*sizeof(uint32_t)));
   HANDLE_ERROR(cudaMallocManaged(&conflict_offset, 6*sizeof(uint32_t)));

   conflict_vertices[0]=1;
   conflict_vertices[1]=10;
   conflict_vertices[2]=0;
   conflict_vertices[3]=7;

   conflict_offset[0]=0;
   conflict_offset[1]=2;
   conflict_offset[2]=2;
   conflict_offset[3]=4;
   conflict_offset[4]=4;
   conflict_offset[5]=4;
   int numColor = 5;

   conflict_resolve<32, 1><<<1, 32, 100*sizeof(uint32_t)>>>(conflict_vertices, conflict_offset, lowTr_col, lowTr_offset, NumRow, numNNZ/2, numColor, color);
   cudaDeviceSynchronize();

//   bool*set;
//   HANDLE_ERROR(cudaMallocManaged(&set, NumRow*sizeof(bool)));
//   memset(set, 0, NumRow); 
//   
//   int numBlocks(1), numThreads(1);
//   if(NumRow < 1024){//if it is less than 1024 vertex, then launch one block 
//   	numBlocks = 1;
//   	numThreads = 1024;
//   }else{//otherwise, launch as many as 1024-blocks as you need    	
//   	numBlocks = (NumRow/1024) + 1;
//   	numThreads = 1024;
//   }
//
//
//   coloring <<<numBlocks, numThreads>>> (NumRow, numNNZ, col_id, offset, color, set);   
//   cudaDeviceSynchronize();
//   /***********************************************************************/
//
//
//   //6) Validate parallel solution 
//   printf("Parallel solution has %d colors\n", CountColors(V, color));
//   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
//   //PrintSolution(color,V);
//   /***********************************************************************/
//
//
//   //7) Color Vertices on CPU
//   //GraphColoring(graph, V, &color);
//   //printf("Brute-foce solution has %d colors\n", CountColors(V, color));   
//   //printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));
//
   GreedyColoring(graph, V, &color);
   printf("\n***************\n");
   printf("Greedy solution has %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   //8)Compare solution 
   /***********************************************************************/

   return 0;
}
