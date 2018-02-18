// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
void ReadMMFile(const char filename[], bool** graph, int* V, uint32_t*numNNZ, uint32_t*NumRow)
{
   using namespace std;
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
	   printf("Failed to open %s\n", filename);
     exit(EXIT_FAILURE);
   }

   (*numNNZ)=0;

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
      }

      (*graph)[(node1) * (*NumRow) + (node2)] = true;
      (*graph)[(node2) * (*NumRow) + (node1)] = true;
   }
   infile.close();

   (*numNNZ)*=2;
}


// Read DIMACS graphs
// Assumes input nodes are numbered starting from 1
void ReadColFile(const char filename[], bool** graph, int* V, uint32_t*numNNZ, uint32_t*NumRow)
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
      }

      // Assume node numbering starts at 1
      (*graph)[(node1) * (*NumRow) + (node2)] = true;
      (*graph)[(node2) * (*NumRow) + (node1)] = true;
   }
   infile.close();

  
  (*numNNZ)*=2;
  

}


//Extract CSR format from the dense adjacency matrix
void getCSR(uint32_t&numNNZ, uint32_t&NumRow, bool* graph, uint32_t *&col_id, uint32_t*&offset){
  //numNNZ is the total number of the non-zero entries in the matrix
  //graph is the input graph  (all memory should be allocated)
 
  int num = 0;
  for(int i=0;i<NumRow;i++){ 

    bool done = false;
    for(int j=0; j<NumRow; j++){//ideally it is NumCol but our matrix is symmetric
      if(graph[i*NumRow + j]){
        col_id[num]=j;
        //std::cout<<"col_id["<<num<<"]= "<<col_id[num]<<"	";
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

void printCSR(uint32_t numNNZ, uint32_t NumRow, uint32_t *col_id, uint32_t*offset)
{
	//print the CSR arries 
	std::cout<<" CSR::numNNZ-> "<<numNNZ <<"   CSR::NumRow->"<<NumRow<<std::endl;
	std::cout<< " CSR::col_id->"<<std::endl;
	for(int i=0; i<numNNZ; i++){
		std::cout<<"	"<< col_id[i];
	}
	std::cout<<""<<std::endl;

	std::cout<< " CSR::offset->"<<std::endl;
	for(int i=0;i<NumRow + 1;i++){
		std::cout<<"	"<<offset[i];		
	}
	std::cout<<""<<std::endl;
}