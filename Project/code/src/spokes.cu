#pragma once


#include <curand_kernel.h>
#include <stdio.h>

#define _tol 10E-6

typedef float real; //Change this between double or (float) single precision
//typedef float3 real3; //Change this between double or (float) single precision
struct real3
{
	real x, y, z;

	real& operator [] (size_t index)
	{
		return *(&x + index);
	}
};

template <typename T>
__device__ __host__  __forceinline__ T cuDist(T x1, T y1, T z1, T x2, T y2, T z2){
	//square distance between point (x1,y1,z1) and (x2,y2,z2) 
	T dx, dy, dz;
	dx = x1 - x2;
	dy = y1 - y2;
	dz = z1 - z2;
	dx *= dx;
	dy *= dy;
	dz *= dz;
	return dx + dy + dz;
}

__device__  __forceinline__ real generateRAND(curandState* globalState, int ind)
{
	//generate random number (callable from the device)

	//stolen from https://nidclip.wordpress.com/2014/04/02/cuda-random-number-generation/
	//copy state to local mem
	ind = ind%1024;
	curandState localState = globalState[ind];
	//apply uniform distribution with calculated random
	real rndval = curand_uniform(&localState);
	//update state
	globalState[ind] = localState;
	//return value
	return rndval;
}
__device__ __forceinline__ void NormalizeVector(real&vector_x, real&vector_y, real&vector_z){
	//Normalize an input vector 
	
	//real nn = rnorm3df(vector_x, vector_y, vector_z);//1/sqrt(vector_x^2 + vector_y^2 + vector_z^2)	
	real nn = real(1)/sqrtf(vector_x*vector_x + vector_y*vector_y + vector_z*vector_z);
	vector_x *= nn; vector_y *= nn; vector_z *= nn;
}
__device__ __forceinline__ void CrossProdcut(const real xv1, const real yv1, const real zv1, //Input:Vector 1
	                                         const real xv2, const real yv2, const real zv2, //Input:Vector 2
	                                         real&xx, real&yy, real&zz){                     //Output:Vector 3
	                                        
	//Find the cross product between vector 1 and vectro 2
	xx = yv1*zv2 - zv1*yv2;
	yy = zv1*xv2 - xv1*zv2;
	zz = xv1*yv2 - yv1*xv2;

}
__device__ __forceinline__ real DotProdcut(const real xv1, const real yv1, const real zv1,  //Input:Vector 1
	                                       const real xv2, const real yv2, const real zv2){ //Input:Vector 2
									 	   
	//Dot product of two vectors 
	return xv1*xv2 + yv1*yv2 + zv1*zv2;

}

__device__ __forceinline__ void ProjectPointOntoPlane(const real point_x, const real point_y, const real point_z, //Input: Point to project
	                                                  const real normal_dx, const real normal_dy, const real normal_dz, //Input: normal to the plan
	                                                  const real orig_x, const real orig_y, const real orig_z,  //Input: point on the plane 
	                                                  real&projected_x, real&projected_y, real&projected_z){//Output: projected point 

	//return the ortho distance to the plane 
	//http://stackoverflow.com/questions/9605556/how-to-project-a-3d-point-to-a-3d-plane

	real point_orig_x(point_x - orig_x), point_orig_y(point_y - orig_y), point_orig_z(point_z - orig_z);
	//NormalizeVector(point_orig_x, point_orig_y, point_orig_z);
	real dot1 = DotProdcut(point_orig_x, point_orig_y, point_orig_z, normal_dx, normal_dy, normal_dz);
	projected_x = point_x - dot1*normal_dx;
	projected_y = point_y - dot1*normal_dy;
	projected_z = point_z - dot1*normal_dz; 

	//return sqrtf(cuDist(projected_x, projected_y, projected_z, point_x, point_y, point_z));
}

__device__ __forceinline__ void RandSpoke1D(const real x,const real y, const real z,        //Input: starting point of the spoke
	                                        const real xn1, const real yn1, const real zn1, //Input: normal to the plane 1 
											const real xn2, const real yn2, const real zn2, //Input: normal to the plane 2 
	                                        real&xv, real&yv, real&zv,                      //Output: direction of the spoke (normalized)
	                                        curandState* globalState, int randID){          //Input: global state for rand generate 
											

	//Random spoke sampling along a 1D line defined by the intersection
	//of two planes (plane 1 and plane 2)
	//spoke starting point should be on the 1D line (not checked)
	//the two planes are defined by their normal vectors

	
	CrossProdcut(xn1, yn1, zn1, xn2, yn2, zn2, xv, yv, zv);

	
	//randomly alternative the direction to point in the opposite direction 
	real randNum = generateRAND(globalState, randID);	
	if(randNum < 0.5){
		xv *=-1;
		yv *=-1;
		zv *=-1;
	}

	NormalizeVector(xv, yv, zv);
	//testing 
	/*real dot1 = DotProdcut(xv,yv,zv, xn1,yn1,zn1);
	real dot2 = DotProdcut(xv,yv,zv, xn2,yn2,zn2);
	printf("\n dot1= %f, dot2= %f", dot1, dot2);*/

}
__device__ __forceinline__ void RandSpoke2D(const real x,const real y, const real z,     //Input: starting point of the spoke
	                                        const real xn, const real yn, const real zn, //Input: normal to the plane  
	                                        real&xv, real&yv, real&zv,                   //Output: direction of the spoke (normalized)
	                                        curandState* globalState, int randID){       //Input: global state for rand generate 
											
	//Random spoke sampling in a 2D plane embedded in the 3D domain
	//spoke starting point should be on the 2D plane 
	//The 2d plane is defined by its normal vector

	//Algorithm: throw random point in the space, then project it
	//to the plane, return the direction as the ray starting from (x,y,z)
	//and pointing to the projected point 

	real Vx_rand = generateRAND(globalState, randID);	
	real Vy_rand = generateRAND(globalState, randID);
	real Vz_rand = generateRAND(globalState, randID);
	

	ProjectPointOntoPlane(Vx_rand, Vy_rand, Vz_rand,
	                      xn, yn,zn,	                      
	                      x,y,z,	                      
	                      xv,yv,zv);
	xv -=x;
	yv -=y;
	zv -=z;		
	NormalizeVector(xv, yv, zv);	

	//testing 
	//real dot = DotProdcut(xv,yv,zv, xn,yn,zn);
	//printf("\n RandSpoke2D() DOT= %f", dot);
	/*printf("\n Vx_rand= %f, Vy_rand= %f, Vz_rand= %f",Vx_rand, Vy_rand, Vz_rand);
	printf("\n \n xn= %f, yn= %f, zn= %f \n xv= %f, yv= %f, zv= %f",
		          xn,     yn,     zn,       xv,     yv,     zv);
	printf("\n dot =%f\n",dot );
	printf("\n px =%f, py =%f, pz =%f\n", x+0.2*xn, y+0.2*yn, z+0.2*zn);
	printf("\n qx =%f, qy =%f, qz =%f\n", x+0.2*xv, y+0.2*yv, z+0.2*zv);*/
	
}
__device__ __forceinline__ void RandSpoke3D(const real x, const  real y, const real z, //Input: starting point of the spoke
	                                        real&xv, real&yv, real&zv,                 //Output: direction of the spoke (normalized)
											curandState* globalState, int randID){     //Input: global state for rand generate 
											
	//Random spoke sampling in the 3d domain; there is no constraints at all
		
	xv = generateRAND(globalState, randID);
	yv = generateRAND(globalState, randID);
	zv = generateRAND(globalState, randID);

	NormalizeVector(xv, yv, zv);

	//printf("\n xv= %f, yv= %f, zv= %f\n", xv, yv, zv);
}

__device__ __forceinline__ bool SpokePlaneIntersect(const real pp_x, const real pp_y, const real pp_z, const real pv_x,  const real pv_y,  const real pv_z,  //Input: plane (point, normal vector)
	                                                const real pt_x, const real pt_y, const real pt_z, const real sp_v_x, const real sp_v_y, const real sp_v_z, //Input: spoke (point and vector)
	                                                real&point_x, real&point_y, real&point_z){ //Output: point
									 			   
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


__device__ __forceinline__ bool SpokePlaneTrimming(const real pp_x, const real pp_y, const real pp_z, const real pv_x,  const real pv_y,  const real pv_z,  //Input: plane (point, normal vector)
	                                               const real pt_x_st, const real pt_y_st, const real pt_z_st, real&pt_x_end, real&pt_y_end, real&pt_z_end){ //Input: spoke (starting and end point)
	                                               

	//Trim the spoke by the plane, 
	//Return the trimmed spoke; only the end point of the spoke is allowed to be change 
	 const real sp_v_x = pt_x_end - pt_x_st;
	 const real sp_v_y = pt_y_end - pt_y_st; 
	 const real sp_v_z = pt_z_end - pt_z_st; 

	 real point_x(0), point_y(0), point_z(0);
	 if(SpokePlaneIntersect(pp_x,pp_y, pp_z, pv_x, pv_y, pv_z, pt_x_st, pt_y_st, pt_z_st, sp_v_x, sp_v_y, sp_v_z, point_x, point_y, point_z)){
	 	pt_x_end = point_x;
	 	pt_y_end = point_y;
	 	pt_z_end = point_z;
	 	return true;
	 }else{
	 	return false;
	 }
}


__device__ __forceinline__ real TriCircumcenter3d(real xa, real ya, real za,
	                                              real xb, real yb, real zb,
	                                              real xc, real yc, real zc,
	                                              real&x_cir, real&y_cir, real&z_cir){
	
	//http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html 
	//http://gamedev.stackexchange.com/questions/60630/how-do-i-find-the-circumcenter-of-a-triangle-in-3d
	real xba, yba, zba, xca, yca, zca;
	real balength, calength;
	real xcrossbc, ycrossbc, zcrossbc;
	real denominator;
	real xcirca, ycirca, zcirca;

	xba = xb - xa;
	yba = yb - ya;
	zba = zb - za;
	xca = xc - xa;
	yca = yc - ya;
	zca = zc - za;
	
	balength = xba * xba + yba * yba + zba * zba;
	calength = xca * xca + yca * yca + zca * zca;

	xcrossbc = yba * zca - yca * zba;
	ycrossbc = zba * xca - zca * xba;
	zcrossbc = xba * yca - xca * yba;

	denominator = real(0.5) / (xcrossbc * xcrossbc + ycrossbc * ycrossbc + zcrossbc * zcrossbc);

	xcirca = ((balength * yca - calength * yba) * zcrossbc - (balength * zca - calength * zba) * ycrossbc) * denominator;
	ycirca = ((balength * zca - calength * zba) * xcrossbc - (balength * xca - calength * xba) * zcrossbc) * denominator;
	zcirca = ((balength * xca - calength * xba) * ycrossbc - (balength * yca - calength * yba) * xcrossbc) * denominator;
	x_cir = xcirca + xa;
	y_cir = ycirca + ya;
	z_cir = zcirca + za;
	
	real len1, dx, dy, dz;
	dx = xa - x_cir;
	dy = ya - y_cir;
	dz = za - z_cir;
	len1 = dx*dx + dy*dy + dz*dz;
	
#ifdef  debug
	dx = xb - x_cir;
	dy = yb - y_cir;
	dz = zb - z_cir;
	real len2 = dx*dx + dy*dy + dz*dz;

	dx = xc - x_cir;
	dy = yc - y_cir;
	dz = zc - z_cir;
	real len3 = dx*dx + dy*dy + dz*dz;

	if (fabs(len1 - len2)>_tol || fabs(len3 - len2)>_tol || fabs(len1 - len3)>_tol){
		printf("\nError at TriCircumcenter3d()..!!\n");	
	}
#endif

	return len1;


}