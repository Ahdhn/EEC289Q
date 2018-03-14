#pragma once

#include "spokes.cu"
#include "defines.h"
#include <stdint.h>

//Everything in global memory 

//****************************************************************************************************
__device__ __forceinline__ uint32_t NeighbourTriming_sh(const uint32_t VertexID, //Input: vertex id (start of the spoke)
                                                        const uint32_t tid, //Input: thread id
														const real x_vertex, const real y_vertex, const real z_vertex, //Input: seed point
	                                                    const real spoke3d_x_st, const real spoke3d_y_st, const real spoke3d_z_st, //Input: spoke end point 
	                                                    real&spoke3d_x_end, real&spoke3d_y_end, real&spoke3d_z_end, //Input/Output: spoke end point 
	                                                    real&trimmingN_x, real&trimmingN_y, real&trimmingN_z, //Output: last neighbour to trim the spoke 
	                                                    const uint32_t skip1, const uint32_t skip2, const uint32_t skip3, const uint32_t skip4,//Input: neighbour to skip
											            const uint32_t neighbour_count, //Input: number of my vertex neighbours 											            											            
											            real3 sh_points[]){   //Input: all points 
	
	//loop over the neighbour of my vertex (neighbour_count) and trim the input spoke 
	//using Voronoi planes, 
	//return the trimmed spoke, the neighbour that trimmed it the shortest 
	uint32_t trimming_neighbour = UINT32_MAX;
	uint32_t base_tid = tid*MaxOffsets;
	for(uint32_t i=1; i<= neighbour_count; i++){
		uint32_t myNeighbour = base_tid + i;		
		
		if(myNeighbour == skip1 || myNeighbour == skip2 || myNeighbour == skip3 || myNeighbour == skip4){ continue; }

		real x_neighbour(sh_points[myNeighbour].x),
		     y_neighbour(sh_points[myNeighbour].y),
		     z_neighbour(sh_points[myNeighbour].z);

		real mid_x = (x_neighbour + x_vertex)/2;//point on the trimming plane 
		real mid_y = (y_neighbour + y_vertex)/2;
		real mid_z = (z_neighbour + z_vertex)/2;

		real norm_p_x = x_neighbour - x_vertex;//normal to the plane 
		real norm_p_y = y_neighbour - y_vertex;
		real norm_p_z = z_neighbour - z_vertex;


		if(SpokePlaneTrimming(mid_x, mid_y, mid_z, norm_p_x, norm_p_y, norm_p_z, //trimming plane 
			                  spoke3d_x_st, spoke3d_y_st, spoke3d_z_st, spoke3d_x_end, spoke3d_y_end, spoke3d_z_end) //spoke
			                  ){			
			trimming_neighbour = myNeighbour;
			trimmingN_x = x_neighbour;
			trimmingN_y = y_neighbour;
			trimmingN_z = z_neighbour;
		}		
	}
	return trimming_neighbour;
}


//****************************************************************************************************
__device__ __forceinline__ uint32_t ThreeDSpoking_sh(const uint32_t VertexID, //Input: vertex id (start of the spoke)
	                                                 const uint32_t tid, //Input: thread id
												     const real x_vertex, const real y_vertex, const real z_vertex, //Input: seed point
												     const real spoke3d_x_st, const real spoke3d_y_st, const real spoke3d_z_st, //Input: spoke starting point
												  	 real&spoke3d_x_end, real&spoke3d_y_end, real&spoke3d_z_end, //Output: spoke end point after trimming 										             
										             const uint32_t neighbour_count,//Input: neighbour of the vertex neighbours 
										             real&grandparent_x, real&grandparent_y, real&grandparent_z, //Output: the grandparent neighbour coordinates 										             
										             real3 sh_points[],       //Input: all points 
										             curandState* globalState, int randID){//Input: global state for rand generate 

	//Shot and trim 3D spoke 
	//Return the last neighbour vertex that trimmed the spoke and the end point of the spoke 
	//grandparent is the neighnour that will trim the spoke the shortest 

	//We use the spoke end as a proxy for direction here and then set the end point correctly after that
	uint32_t grandparent = UINT32_MAX;

	while(grandparent == UINT32_MAX){

	RandSpoke3D(spoke3d_x_st, spoke3d_y_st, spoke3d_z_st, spoke3d_x_end, spoke3d_y_end, spoke3d_z_end, globalState, randID);
	
		spoke3d_x_end = spoke3d_x_st + 1000*spoke3d_x_end;
		spoke3d_y_end = spoke3d_y_st + 1000*spoke3d_y_end;
		spoke3d_z_end = spoke3d_z_st + 1000*spoke3d_z_end;

		//printf("\n 1) spoke3d_end( %f,%f,%f )", spoke3d_x_end, spoke3d_y_end, spoke3d_z_end);
	
	
		grandparent = NeighbourTriming_sh(VertexID,
			                              tid,
			                              x_vertex, y_vertex, z_vertex,
		    	                          spoke3d_x_st, spoke3d_y_st, spoke3d_z_st, 
			                              spoke3d_x_end, spoke3d_y_end, spoke3d_z_end,
		        	                      grandparent_x, grandparent_y, grandparent_z,
		            	                  0,0,0,0,
		                	              neighbour_count, sh_points);		

		//if(grandparent == UINT32_MAX){printf(" Invalid grand\n");}		
	}
	/*printf("\n 2) spoke3d_end( %f,%f,%f )", spoke3d_x_end, spoke3d_y_end, spoke3d_z_end);
	printf("\n grandparent( %f,%f,%f )", grandparent_x, grandparent_y, grandparent_z);
	printf("\n 3dspoke_plan( %f,%f,%f, %f,%f,%f )\n", grandparent_x - x_vertex, grandparent_y- y_vertex , grandparent_z- z_vertex, 
			(grandparent_x + x_vertex)/2.0, (grandparent_y + y_vertex)/2.0 , (grandparent_z + z_vertex)/2.0);*/

	return grandparent;
}

//****************************************************************************************************
__device__ __forceinline__ uint32_t TwoDSpoking_sh(const uint32_t VertexID, //Input: vertex id (start of the spoke)
												   const uint32_t tid, //Input: thread id
												   const real x_vertex, const real y_vertex, const real z_vertex, //Input: seed point
												   const real spoke2d_x_st, const real spoke2d_y_st, const real spoke2d_z_st, //Input: spoke starting point
												   real&spoke2d_x_end, real&spoke2d_y_end, real&spoke2d_z_end, //Output: spoke end point after trimming 										           
										           const uint32_t neighbour_count,//Input: neighbour of the vertex neighbours 
										           real&parent_x, real&parent_y, real&parent_z, //Output: the parent neighbour coordinates 
										           const uint32_t grandparent, //Input: the neighbour with whom the spoke lives on its voronoi facet 
										           const real grandparent_x, const real grandparent_y,  const real grandparent_z,//Input: grandparent neighbour coordinates 										           
										           real3 sh_points[],       //Input: all points 
										           curandState* globalState, int randID){//Input: global state for rand generate

	//Shot and trim 2D spoke 
	//Return the last neighbour vertex that trimmed the spoke and the end point of the spoke 

	real norm_p_x = grandparent_x - x_vertex;//normal to the plane 
	real norm_p_y = grandparent_y - y_vertex;
	real norm_p_z = grandparent_z - z_vertex;
	NormalizeVector(norm_p_x, norm_p_y, norm_p_z);

	uint32_t parent = UINT32_MAX; 

	while (parent == UINT32_MAX){
		//We use the spoke end as a proxy for direction here and then set the end point correctly after that
		RandSpoke2D(spoke2d_x_st, spoke2d_y_st, spoke2d_z_st, //2D spoke starting point 
			        norm_p_x, norm_p_y, norm_p_z, //normal to the plane 
		    	    spoke2d_x_end, spoke2d_y_end, spoke2d_z_end, //2d spoke direction 
		        	globalState, randID);

		spoke2d_x_end = spoke2d_x_st + 1000*spoke2d_x_end;
		spoke2d_y_end = spoke2d_y_st + 1000*spoke2d_y_end;
		spoke2d_z_end = spoke2d_z_st + 1000*spoke2d_z_end;
	

		//printf("\n 1) spoke2d_end( %f,%f,%f )", spoke2d_x_end, spoke2d_y_end, spoke2d_z_end);

	
		parent = NeighbourTriming_sh(VertexID,
			                         tid,
			                         x_vertex, y_vertex, z_vertex, 
								     spoke2d_x_st, spoke2d_y_st, spoke2d_z_st, 
		    	                     spoke2d_x_end, spoke2d_y_end, spoke2d_z_end,
		        	                 parent_x, parent_y, parent_z,
		            	             0, grandparent, 0, 0, 
		                	         neighbour_count, sh_points);
		//if(parent == UINT32_MAX){printf(" Invalid parent\n");}		
	}
	/*printf("\n 2) spoke2d_end( %f,%f,%f )", spoke2d_x_end, spoke2d_y_end, spoke2d_z_end);
	printf("\n parent( %f,%f,%f )", parent_x, parent_y, parent_z);
	printf("\n 2dspoke_plan( %f,%f,%f, %f,%f,%f )\n", parent_x - x_vertex, parent_y- y_vertex , parent_z- z_vertex, 
			(parent_x + x_vertex)/2.0, (parent_y + y_vertex)/2.0 , (parent_z + z_vertex)/2.0);*/

	return parent;
}

//****************************************************************************************************
__device__ __forceinline__ uint32_t OneDSpoking_sh(const uint32_t VertexID, //Input: vertex id (start of the spoke)
												   const uint32_t tid, //Input: thread id
												   const real x_vertex, const real y_vertex, const real z_vertex, //Input: seed point
									               const real spoke1d_x_st, const real spoke1d_y_st, const real spoke1d_z_st, //Input: spoke starting point
									               real&spoke1d_x_end, real&spoke1d_y_end, real&spoke1d_z_end, //Output: spoke end point after trimming 									               
									               const uint32_t neighbour_count,//Input: neighbour of the vertex neighbours 									            
										           const uint32_t grandparent, //Input: the neighbour with whom the spoke lives on its voronoi facet 
										           const real grandparent_x, const real grandparent_y, const real grandparent_z,//Input: grandparent neighbour coordinates 
										           const uint32_t parent, //Input: the other neighbout with whom the spoke lives on another voronoi facet 
										           const real parent_x, const real parent_y, const real parent_z,//Input: parent neighbour coordinates 										           
										           real3 sh_points[],       //Input: all points 
										           curandState* globalState, int randID){//Input: global state for rand generate

	//Shot and trim 1D spoke 
	//Return the last neighbour vertex that trimmed the spoke and the end point of the spoke

	real norm_p1_x = grandparent_x - x_vertex;//normal to the plane 1
	real norm_p1_y = grandparent_y - y_vertex;
	real norm_p1_z = grandparent_z - z_vertex;

	NormalizeVector(norm_p1_x,norm_p1_y,norm_p1_z);

	real norm_p2_x = parent_x - x_vertex;//normal to the plane 2
	real norm_p2_y = parent_y - y_vertex;
	real norm_p2_z = parent_z - z_vertex;

	NormalizeVector(norm_p2_x,norm_p2_y,norm_p2_z);

	uint32_t child = UINT32_MAX; 
	real child_x, child_y, child_z;
	while(child == UINT32_MAX){
		RandSpoke1D(spoke1d_x_st, spoke1d_y_st, spoke1d_z_st,
	    	        norm_p1_x, norm_p1_y, norm_p1_z,
	            	norm_p2_x, norm_p2_y, norm_p2_z,
	            	spoke1d_x_end, spoke1d_y_end, spoke1d_z_end,
	            	globalState, randID);

		spoke1d_x_end = spoke1d_x_st + 1000*spoke1d_x_end;
		spoke1d_y_end = spoke1d_y_st + 1000*spoke1d_y_end;
		spoke1d_z_end = spoke1d_z_st + 1000*spoke1d_z_end;	
	
		
		child = NeighbourTriming_sh(VertexID,
			                        tid,
			                        x_vertex, y_vertex, z_vertex, 
								    spoke1d_x_st, spoke1d_y_st, spoke1d_z_st, 
		    	                    spoke1d_x_end, spoke1d_y_end, spoke1d_z_end,
		        	                child_x, child_y, child_z,
		            	            0, grandparent, parent, 0, 
		                	        neighbour_count, sh_points);
		//if(child == UINT32_MAX){printf(" Invalid child\n");}		
	}
	/*printf("\n 2) spoke1d_end( %f,%f,%f )", spoke1d_x_end, spoke1d_y_end, spoke1d_z_end);
	printf("\n child( %f,%f,%f )", child_x, child_y, child_z);
	printf("\n 1dspoke_plan( %f,%f,%f, %f,%f,%f )\n", child_x - x_vertex, child_y- y_vertex , child_z- z_vertex, 
			(child_x + x_vertex)/2.0, (child_y + y_vertex)/2.0 , (child_z + z_vertex)/2.0);*/
	return child;
}

//****************************************************************************************************
__device__ __forceinline__ void explore_sh(const uint32_t vertexID, //Input: vertex to explore 
										   const uint32_t tid, //Input: thread id
										   const real x_vertex,const real y_vertex,const real z_vertex,//Input: vertex coords
                                           real3 sh_points[],   //Input: all points 
                                           const uint32_t neighbour_count, //Input: num neighbours                                         
                                           curandState* globalState, int randID,  //Input: global state for rand generate 
                                           uint3&exploredID, //Output: the id of three samples connected to vertexID
                                           real3&sharedVertex){  //Output: shared voronoi vertex between exploredID                                        
		
	//printf("\n ID=%i, myVertex( %f,%f,%f )\n",vertexID, x_vertex, y_vertex, z_vertex);
	real grandparent_x, grandparent_y, grandparent_z, parent_x, parent_y, parent_z;	

	
	//Shot and trim a 3D spoke with all neighbours and keep a record for the last trimming 
	//neighbour -> grandparent neighbour 
	real spoke3d_x_end, spoke3d_y_end, spoke3d_z_end;
	uint32_t grandparent = ThreeDSpoking_sh(vertexID,
											tid,		
		                                    x_vertex, y_vertex, z_vertex,
										    x_vertex, y_vertex, z_vertex,
										    spoke3d_x_end, spoke3d_y_end, spoke3d_z_end, 										 
										    neighbour_count,
										    grandparent_x, grandparent_y, grandparent_z,										 
										    sh_points,
										    globalState, randID);

	
	//Shot 2D spoke from the intersection point 
	//Trim the 2D spoke with all the neighbours (excpect the grandparent neighbour)
	//and keep a record for the last trimming neighbour -> parent neighbour
	real spoke2d_x_end, spoke2d_y_end, spoke2d_z_end;
	uint32_t parent = TwoDSpoking_sh(vertexID, 
									 tid,
								     x_vertex, y_vertex, z_vertex,
								     spoke3d_x_end, spoke3d_y_end, spoke3d_z_end,
	                                 spoke2d_x_end, spoke2d_y_end, spoke2d_z_end,         
	                                 neighbour_count,
	                                 parent_x, parent_y, parent_z,
	                                 grandparent,
	                                 grandparent_x, grandparent_y, grandparent_z,	                                 
								     sh_points,
								     globalState, randID);
	

	                
	//Shot 1D spoke 
	//Trim the 1D spoke with all the neighbours  (excpect the grandparent and parent neighbours)
	//and keep a record for the last trimming neighbour -> child neighbour
    real spoke1d_x_end, spoke1d_y_end, spoke1d_z_end;
    uint32_t child = OneDSpoking_sh(vertexID, 
    								tid,
    							    x_vertex, y_vertex, z_vertex,
								    spoke2d_x_end, spoke2d_y_end, spoke2d_z_end,
								    spoke1d_x_end, spoke1d_y_end, spoke1d_z_end,								    
								    neighbour_count,								 
								    grandparent,
								    grandparent_x, grandparent_y, grandparent_z,
								    parent,
								    parent_x, parent_y, parent_z, 								    
								    sh_points,
								    globalState, randID);

    
	//Return the grandparent, parent and child as exploredID 
	//Return the end point of the 1D spoke as sharedVertex
	//printf("\n grandparent= %i,parent= %i, child= %i,\n",grandparent, parent, child);
	exploredID.x = grandparent; 
    exploredID.y = parent; 
    exploredID.z = child; 
    sharedVertex.x = spoke1d_x_end;
    sharedVertex.y = spoke1d_y_end;
    sharedVertex.z = spoke1d_z_end;
}