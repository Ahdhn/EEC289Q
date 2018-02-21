__global__
void conflict_resolve_forgetabout_sharedmemory(uint32_t* conflit_color, // Array of conflict vetices grouped by color
                      uint32_t *conflict_color_offset, // offset of different color on conflit_color 
                      uint32_t *tr_col_id, // CSR of graph, but only lower triangle part
                      uint32_t *tr_offset, // CSR offset of graph, but only lower triangle part
                      uint32_t numVertices, // number of vertices
                      uint32_t size_tr_col,// size of tr_col_id
                      uint32_t numColor,// number of color has been used
                      unsigned char *color) // color array for all vertices 
{
	



