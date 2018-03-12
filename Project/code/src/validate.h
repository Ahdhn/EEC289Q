#pragma once
//Validate the correctness of the constructed delaunay 
//TODO make sure everything is going as meant to be (after testing extractTets)

#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "spokes.cu"
#include "circumSphere.h"
#include "utilities.h"

bool insideCircumSphere(const uint32_t p, const real r_circm, const real x_cirm, const real y_cirm, const real z_cirm,
	                    uint32_t*const h_neighbors, real3*const Points, const int MaxOffset,
						uint32_t skip1, uint32_t skip2, uint32_t skip3, uint32_t skip4){
	//use the neighbours of p to check if any is inside of the circumsphere
	//if so return true, otherwise return false;
	
	uint32_t base = p*MaxOffset;
	uint32_t count = h_neighbors[base];
	for (uint32_t i = 1; i < count; i++){
		uint32_t id = h_neighbors[base + i];
		if (id == skip1 || id == skip2 || id == skip3 || id == skip4){ continue; }
		real dist = Dist(Points[id].x, Points[id].y, Points[id].z, x_cirm, y_cirm, z_cirm);
		if (dist < r_circm - _tol*_tol){
			return true;
		}
	}
	return false;
}

void validate(const  std::vector<std::vector<uint32_t>>myTets, real3*const Points, uint32_t*const h_neighbors, const int MaxOffset){
	//validate the correctness of the delaunay by checking the empty sphere of each tet 
	//usig the neughbour of the tet vertices 
	for (uint32_t t = 0; t < myTets.size(); t++){
		uint32_t p0 = myTets[t][0];
		uint32_t p1 = myTets[t][1];
		uint32_t p2 = myTets[t][2];
		uint32_t p3 = myTets[t][3];

		real x_cirm, y_cirm, z_cirm;
		real r_cirm = circumSphere(Points[p0].x, Points[p0].y, Points[p0].z,
			                       Points[p1].x, Points[p1].y, Points[p1].z,
			                       Points[p2].x, Points[p2].y, Points[p2].z,
			                       Points[p3].x, Points[p3].y, Points[p3].z,
			                       x_cirm, y_cirm, z_cirm);
		if (r_cirm > _tol){
			if (insideCircumSphere(p0, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p0, p1, p2, p3) ||
				insideCircumSphere(p1, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p0, p1, p2, p3) ||
				insideCircumSphere(p2, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p0, p1, p2, p3) ||
				insideCircumSphere(p3, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p0, p1, p2, p3) ){

				printf("\n The tet connecting [%d] [%d] [%d] [%d] is invalid", p0, p1, p2, p3);			
			}
		}
	}
}

/*void validate(const size_t NumPoints, real3*const Points, uint32_t*const h_neighbors, uint32_t* const h_delaunay, const int MaxOffset){
	//validate the correctness of the delaunay by checking the empty sphere properity 
	for (uint32_t p = 0; p < NumPoints; p++){//first corner in the tet
		uint32_t base = p*MaxOffset;
		uint32_t numConnect = h_delaunay[base];
		for (uint32_t i = 1; i < numConnect - 2; i++){//second corner in the tet
			uint32_t p_i = h_delaunay[base + i];
			//TODO tgus us wrong, find the common points
			for (uint32_t j = i + 1; j < numConnect - 1; j++){//third corner in the tet
				uint32_t p_j = h_delaunay[base + j];

				//TODO tgus us wrong, find the common points
				for (uint32_t k = j + 1; k < numConnect - 1; k++){//forth corner in the tet
					uint32_t p_k = h_delaunay[base + k];	
					
					if (p < p_i &&p < p_j &&p  < p_k){//to make sure that we test a tet just one time 
						//get circumspheres 
						real x_cirm, y_cirm, z_cirm;
						real r_cirm = circumSphere(Points[p].x, Points[p].y, Points[p].z,
							                       Points[p_i].x, Points[p_i].y, Points[p_i].z,
							                       Points[p_j].x, Points[p_j].y, Points[p_j].z,
							                       Points[p_k].x, Points[p_k].y, Points[p_k].z,
							                       x_cirm, y_cirm, z_cirm);
						if (r_cirm > _tol){
							if (insideCircumSphere(p, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p, p_i, p_j, p_k) ||
								insideCircumSphere(p_i, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p, p_i, p_j, p_k) ||
								insideCircumSphere(p_j, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p, p_i, p_j, p_k) ||
								insideCircumSphere(p_k, r_cirm, x_cirm, y_cirm, z_cirm, h_neighbors, Points, MaxOffset, p, p_i, p_j, p_k)){
								printf("\n The tet connecting [%d] [%d] [%d] [%d] is invalid", p, p_i, p_j, p_k){

								}
							}
						}
					}
				}
			}
		}
	}
}*/