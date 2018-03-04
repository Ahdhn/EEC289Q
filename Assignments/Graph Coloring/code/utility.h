// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
void ReadMMFile(const char filename[], bool** graph, int* V, uint32_t*numNNZ, uint32_t*NumRow,
                uint32_t&blockingSize,uint32_t*numNNZ_blocked)
{
   using namespace std;
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
     printf("Failed to open %s\n", filename);
     exit(EXIT_FAILURE);
   }

   (*numNNZ)=0;
   (*numNNZ_blocked)=0;

   // Reading comments
   while (getline(infile, line)) {
      istringstream iss(line);
      if (line.find("%") == string::npos)
         break;
   }

   // Reading metadata
   istringstream iss(line);

   int num_cols, num_edges;
   iss >> (*NumRow) >> num_cols >> num_edges;

   *graph = new bool[(*NumRow) * (*NumRow)];

   for(int i = 0; i<(*NumRow) * (*NumRow);i++){
     (*graph)[i] = false;
   }

   memset(*graph, 0, (*NumRow) * (*NumRow) * sizeof(bool));
   *V = (*NumRow);
   if((*V) < 32){ //if it is small graph 
    blockingSize = min(5,(*V));
   }else {      
      blockingSize = max((*V) / 1024,1); 
      blockingSize*=32;
      if(blockingSize > 1024){
        blockingSize = 1024;
      }
   }

  

   // Reading nodes
   while (getline(infile, line)) {
      istringstream iss(line);
      int node1, node2, weight;
      iss >> node1 >> node2 >> weight;
      node1--;
      node2--;

      // Assume node numbering starts at 1
      //Only count numNNZ once (there might be an edge that is there more than
      //once)

      if(!(*graph)[(node1) * (*NumRow) + (node2)]
         //actually we just need to one of these
         //&& !(*graph)[(node2) * (*NumRow) + (node1)]
        ){
           (*numNNZ)++;
         if(floor(int(node1)/int(blockingSize)) == floor(int(node2)/int(blockingSize))){
          (*numNNZ_blocked)++;
         }
      }

      (*graph)[(node1) * (*NumRow) + (node2)] = true;
      (*graph)[(node2) * (*NumRow) + (node1)] = true;
   }
   infile.close();

   (*numNNZ)*=2;
   (*numNNZ_blocked)*=2;
}


// Read DIMACS graphs
// Assumes input node s are numbered starting from 1
void ReadColFile(const char filename[], bool** graph, int* V, uint32_t*numNNZ, uint32_t*NumRow,
                 uint32_t&blockingSize,uint32_t*numNNZ_blocked)
{
   using namespace std;
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      exit(EXIT_FAILURE);
   }

   (*numNNZ)=0; // initilize with zero
   int num_edges;

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
      int node1, node2;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> (*NumRow);
         iss >> num_edges;
         *V = (*NumRow);
         if((*V) < 32){ //if it is small graph 
         blockingSize = min(5,(*V));
         }else {
          blockingSize = max((*V) / 1024,1); 
          blockingSize*=32;
          if(blockingSize > 1024){
              blockingSize = 1024;
          }
         }
         *graph = new bool[(*NumRow) * (*NumRow)];
         memset(*graph, 0, (*NumRow) * (*NumRow) * sizeof(bool));
         for(int i = 0; i<(*NumRow) * (*NumRow);i++){
           (*graph)[i] = false;
         }
         continue;
      } else if (s != "e"){ continue;}

      iss >> node1 >> node2;
      node1--;
      node2--;

      //Only count numNNZ once (there might be an edge that is there more than
      //once)
      if(!(*graph)[(node1) * (*NumRow) + (node2)]
         //actually we just need to one of these
         //&& !(*graph)[(node2) * (*NumRow) + (node1)]
         ){
           (*numNNZ)++;
           if(floor(int(node1)/int(blockingSize)) == floor(int(node2)/int(blockingSize))){
            (*numNNZ_blocked)++;
           }
      }

      // Assume node numbering starts at 1
      (*graph)[(node1) * (*NumRow) + (node2)] = true;
      (*graph)[(node2) * (*NumRow) + (node1)] = true;
   }
   infile.close();

  
  (*numNNZ)*=2;
  (*numNNZ_blocked)*=2;
}

//Extract CSR format from the dense adjacency matrix
//Here we we only get blocks of the CSR 
//for each vertex i connected to j, store j in the col_id iff j is in the block of i

void getBlockedCSR(uint32_t&NumRow, bool* graph, uint32_t *&col_id, uint32_t*&offset, uint32_t blockSize, uint32_t & maxLeftout){
  //numNNZ is the total number of the non-zero entries in the matrix
  //graph is the input graph  (all memory should be allocated)  
 
  int num = 0;  
  maxLeftout = 0;
  for(int i=0; i<NumRow; i++){       
    uint32_t myLeftout = 0;
    for(int j=0; j<NumRow; j++){//ideally it is NumCol but our matrix is symmetric
      
      if(graph[i*NumRow + j]){

        if(floor(int(j)/int(blockSize)) == floor(int(i)/int(blockSize))){
          col_id[num]=j;          
          num++;          
        }else{
          myLeftout++;
        }
      }
    }
    offset[i+1] = num;    
    maxLeftout = max(maxLeftout, myLeftout);
  }
}

//Extract CSR format from the dense adjacency matrix
void getCSR(uint32_t numNNZ, uint32_t&NumRow, bool* graph, uint32_t *&col_id, uint32_t*&offset){
  //numNNZ is the total number of the non-zero entries in the matrix
  //graph is the input graph  (all memory should be allocated)
 
  int num = 0;
  for(int i=0;i<NumRow;i++){ 

    bool done = false;
    for(int j=0; j<NumRow; j++){//ideally it is NumCol but our matrix is symmetric
      if(graph[i*NumRow + j]){
        col_id[num]=j;
        //std::cout<<"col_id["<<num<<"]= "<<col_id[num]<<"  ";
        if(!done){
          offset[i]=num;
          done = true;
        }
        num++;
      }
    }
  }
  offset[NumRow] = numNNZ;
}

//Extract CSR format from the dense adjacency matrix
void getLowTrCSR(uint32_t&numNNZ, uint32_t&NumRow, bool* graph, uint32_t *&col_id, uint32_t*&offset){
  //numNNZ is the total number of the non-zero entries in the matrix
  //graph is the input graph  (all memory should be allocated)
 
  int num = 0;
  offset[0]=num;
  for(int i=1;i<NumRow;i++){ 
//    std::cout<<"row"<<i<<std::endl;

    bool done = false;
    for(int j=0; j<i; j++){//ideally it is NumCol but our matrix is symmetric
//  std::cout<<"colum"<<j<<std::endl;
      if(graph[i*NumRow + j]){
        col_id[num]=j;
//        std::cout<<"col_id["<<num<<"]= "<<col_id[num]<<std::endl;
        if(!done){
          offset[i]=num;
//    std::cout<<"offset["<<i<<"]= "<<offset[i]<<std::endl;
          done = true;
        }
        num++;
      }
    }
    if(!done){
  offset[i]=num;
//  std::cout<<"offset["<<i<<"]= "<<offset[i]<<std::endl;
  done = true;
    }
  }
  offset[NumRow] = numNNZ/2;
}

void printCSR(uint32_t numNNZ, uint32_t NumRow, uint32_t *col_id, uint32_t*offset)
{
  //print the CSR arries 
  std::cout<<" CSR::numNNZ-> "<<numNNZ <<"   CSR::NumRow->"<<NumRow<<std::endl;
  std::cout<< " CSR::col_id->"<<std::endl;
  for(int i=0; i<numNNZ; i++){
    std::cout<<"  "<< col_id[i];
  }
  std::cout<<""<<std::endl;

  std::cout<< " CSR::offset->"<<std::endl;
  for(int i=0;i<NumRow + 1;i++){
    std::cout<<"  "<<offset[i];   
  }
  std::cout<<""<<std::endl;
}

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