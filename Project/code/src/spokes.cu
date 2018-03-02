#pragma once

#include <curand.h>
#include <math.h>

typedef double real; //Change this between double or (float) single precision

__device__ __forceinline__ void CrossProdcut(const real xv1, const real yv1, const real zv1, //Input:Vector 1
	                                         const real xv2, const real yv2, const real zv2, //Input:Vector 2
	                                         real xx, real yy, real zz                       //Output:Vector 3
	                                        ){
	//Find the cross product between vector 1 and vectro 2
	xx = yv1*zv2 - zv1*yv2;
	yy = zv1*xv2 - xv1*zv2;
	zz = xv1*yv2 - yv1*xv2;

}
__device__ __forceinline__ real DotProdcut(const real xv1, const real yv1, const real zv1, //Input:Vector 1
	                                       const real xv2, const real yv2, const real zv2  //Input:Vector 2
									 	   ){
	//Dot product of two vectors 
	return xv1*xv2 + yv1*yv2 + zv1*zv2;

}

__device__ __forceinline__ void RandSpoke1D(const real x,const real y, const real z,        //Input: starting point of the spoke
	                                        const real xn1, const real yn1, const real zn1, //Input: normal to the plane 1 
											const real xn2, const real yn2, const real zn2, //Input: normal to the plane 2 
	                                        real&xv, real&yv, real&zv                       //Output: direction of the spoke 
											){

	//Random spoke sampling along a 1D line defined by the intersection
	//of two planes (plane 1 and plane 2)
	//spoke starting point should be on the 1D line (not checked)
	//the two planes are defined by their normal vectors

	real xv_per, yv_per, zv_per;//the perpenduclar vector of the two input vectors (the direction over which we gonna sample)
	CrossProdcut(xn1, yn1, zn1, xn2, yn2, zn2, xv, yv, zv);

	//TODO maybe it is beneficial to randomly alternative the direction to point 
	//in the opposite direction 
}
__device__ __forceinline__ void RandSpoke2D(const real x,const real y, const real z,     //Input: starting point of the spoke
	                                        const real xn, const real yn, const real zn, //Input: normal to the plane  
	                                        real&xv, real&yv, real&zv                    //Output: direction of the spoke 
											){
	//Random spoke sampling in a 2D plane embedded in the 3D domain
	//spoke starting point should be on the 2D plane (not checked)
	//The 2d plane is defined by its normal vector 
	
}
__device__ __forceinline__ void RandSpoke3D(const real x, const  real y, const real z, //Input: starting point of the spoke
	                                        real&xv, real&yv, real&zv                  //Output: direction of the spoke 
											){
	//Random spoke sampling in the 3d domain; there is no constraints at all

	//TODO use curand random generator 	

}

__device__ __forceinline__ bool SpokePlaneIntersect(const real pp_x, const real pp_y, const real pp_z, const real pv_x,  const real pv_y,  const real pv_z,  //Input: plane (point, normal vector)
	                                                const real pt_x, const real pt_y, const real pt_z, const real sp_v_x, const real sp_v_y, const real sp_v_z, //Input: spoke (point and vector)
	                                                real&point_x, real&point_y, real&point_z //Output: point
									 			   ){
	//Plane line intersection. Plane define by normal vector (pv_x,pv_y,pv_z) and point on it(pp_x,pp_y,pp_z)
	// and line between point ip1 and ip2
	
	real dot = DotProdcut(sp_v_x, sp_v_y, sp_v_z, pv_x, pv_y, pv_z);

	if (abs(dot) <= 0.0){
		return false;
	}
	
	real s = (DotProdcut(pv_x, pv_y, pv_z, pp_x - pt_x, pp_y - pt_y, pp_z - pt_z)) / (dot);


	if (s<-1.0*10E-8 || s >1.0 + 10E-8){
		return false;
	}
	
	point_x = pt_x + s*sp_v_x;
	point_y = pt_y + s*sp_v_y;
	point_z = pt_z + s*sp_v_z;

	return true;

}