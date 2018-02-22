#include <sstream>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverSp.h>

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

uint32_t maxNNZ_per_segment(uint32_t*offset, uint32_t NumRow, uint32_t segment_length){
   //count the max number of nonzero elements within a segment_length of offset array
   //i.e., how many nonzero elements between row i and row j such that j-i = segment_length
   uint32_t max_seg = 0;
   

   for(uint32_t seg_start=0; seg_start < NumRow; seg_start+= segment_length) {      
      uint32_t seg_end = (seg_start + segment_length < NumRow ) ? seg_start+segment_length : NumRow;
      uint32_t my_len = offset[seg_end] - offset[seg_start];
      //std::cout<<" seg_start= "<<seg_start<< " seg_end= "<<seg_end<<" my_len= "<<my_len<< " offset[seg_end]= "<< offset[seg_end]<< " offset[seg_start]= "<< offset[seg_start]<<std::endl;
      if (my_len > max_seg){max_seg = my_len;}
   }
   return max_seg;
}

int main(int argc, char* argv[]){
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


   int numBlocks(1), numThreads(1);
   if(NumRow < 1024){//if it is less than 1024 vertex, then launch one block 
      numBlocks = 1;
      numThreads = NumRow;
   }else{//otherwise, launch as many as 1024-blocks as you need      
      numBlocks = (NumRow/1024) + 1;
      numThreads = 1024;
   }

   const uint32_t blockingSize = 2;

   //3) Get graph in CSR format 
   //getCSR(numNNZ, NumRow, graph, col_id, offset);
   getBlockedCSR(numNNZ, NumRow, graph, col_id, offset, blockingSize);
   //printCSR(numNNZ,NumRow,col_id, offset);   
   /***********************************************************************/


   //5) Color Vertices in paralllel
   unsigned char* color;
   HANDLE_ERROR(cudaMallocManaged(&color, NumRow*sizeof(unsigned char)));
   memset(color, 0, NumRow );
   bool*set;
   HANDLE_ERROR(cudaMallocManaged(&set, NumRow*sizeof(bool)));
   memset(set, 0, NumRow); 

   uint32_t max_NNZ_per_block= maxNNZ_per_segment(offset, NumRow, blockingSize);
   //std::cout<< "  max_NNZ_per_block= "<<max_NNZ_per_block <<std::endl;

   
   const uint32_t shrd_mem = numThreads*sizeof(bool) + max_NNZ_per_block*numThreads*sizeof(uint32_t);
                              //^^ sh_set                     ^^ sh_col_id
   std::cout<<" shrd_mem="<<shrd_mem<<" = numThreads*sizeof(bool){"<<numThreads*sizeof(bool)<<"} + max_NNZ_per_block*numThreads*sizeof(uint32_t){"<<max_NNZ_per_block*numThreads*sizeof(uint32_t)<<"}"<<std::endl;
   coloring <<<numBlocks, numThreads, shrd_mem>>> (NumRow, col_id, offset, color);     
   cudaDeviceSynchronize(); 
   /***********************************************************************/

    //5.5) Validate parallel LOCAL solution 
   printf("Parallel LOCAL solution has %d colors\n", CountColors(V, color));
   printf("Valid LOCAL coloring: %d\n\n", IsValidColoring_Blocked(graph, V, color, blockingSize));
   PrintSolution(color,V);
   /***********************************************************************/


   //6) Validate parallel solution 
   //printf("Parallel solution has %d colors\n", CountColors(V, color));
   //printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   //7) Color Vertices on CPU
   //GraphColoring(graph, V, &color);
   //printf("Brute-foce solution has %d colors\n", CountColors(V, color));   
   //printf("Valid coloring: %d\n", IsValidColoring(graph, V, color));

   GreedyColoring(graph, V, &color);
   printf("\n*********************************\n");
   printf("Greedy solution has %d colors\n", CountColors(V, color));
   printf("Valid coloring: %d\n\n", IsValidColoring(graph, V, color));
   //PrintSolution(color,V);
   /***********************************************************************/


   //8)Compare solution 
   /***********************************************************************/

   return 0;
}