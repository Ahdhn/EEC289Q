#pragma once
//construct the circumsphere of four points in 3d 
#include "spokes.cu"
#include "utilities.h"
#include "defines.h"
#include <stdio.h>

__device__ __host__ __device__ real  Determinant3(real A[4][4])
{
	return A[1][1] * (A[2][2] * A[3][3] - A[3][2] * A[2][3]) -
		   A[1][2] * (A[2][1] * A[3][3] - A[2][3] * A[3][1]) +
		   A[1][3] * (A[2][1] * A[3][2] - A[3][1] * A[2][2]);

}
__device__ __host__ real Determinant4(real A[5][5])
{
	real A1[4][4];

	real det1, det2, det3, det4;

	A1[1][1] = A[2][2]; A1[1][2] = A[2][3]; A1[1][3] = A[2][4];
	A1[2][1] = A[3][2]; A1[2][2] = A[3][3]; A1[2][3] = A[3][4];
	A1[3][1] = A[4][2]; A1[3][2] = A[4][3]; A1[3][3] = A[4][4];
	det1 = Determinant3(A1);

	A1[1][1] = A[2][1]; A1[1][2] = A[2][3]; A1[1][3] = A[2][4];
	A1[2][1] = A[3][1]; A1[2][2] = A[3][3]; A1[2][3] = A[3][4];
	A1[3][1] = A[4][1]; A1[3][2] = A[4][3]; A1[3][3] = A[4][4];
	det2 = Determinant3(A1);

	A1[1][1] = A[2][1]; A1[1][2] = A[2][2]; A1[1][3] = A[2][4];
	A1[2][1] = A[3][1]; A1[2][2] = A[3][2]; A1[2][3] = A[3][4];
	A1[3][1] = A[4][1]; A1[3][2] = A[4][2]; A1[3][3] = A[4][4];
	det3 = Determinant3(A1);

	A1[1][1] = A[2][1]; A1[1][2] = A[2][2]; A1[1][3] = A[2][3];
	A1[2][1] = A[3][1]; A1[2][2] = A[3][2]; A1[2][3] = A[3][3];
	A1[3][1] = A[4][1]; A1[3][2] = A[4][2]; A1[3][3] = A[4][3];
	det4 = Determinant3(A1);

	real val = A[1][1] * det1 - A[1][2] * det2 + A[1][3] * det3 - A[1][4] * det4;

	return val;

}
//inline double circumSphere(size_t ip1, size_t ip2, size_t ip3, size_t ip4, double**samples, double&xo, double&yo, double&zo)
//{
//	double x1(samples[ip1][0]), y1(samples[ip1][1]), z1(samples[ip1][2]),
//		x2(samples[ip2][0]), y2(samples[ip2][1]), z2(samples[ip2][2]),
//		x3(samples[ip3][0]), y3(samples[ip3][1]), z3(samples[ip3][2]),
//		x4(samples[ip4][0]), y4(samples[ip4][1]), z4(samples[ip4][2]);

__device__ __host__ real circumSphere(const real x1, const real y1, const real z1, 
	                                  const real x2, const real y2, const real z2,
						              const real x3, const real y3, const real z3,
						              const real x4, const real y4, const real z4,
						              real&xo, real&yo, real&zo)
{
	
	real  M11, M12, M13, M14, M15;

	real a11(x1*x1 + y1*y1 + z1*z1), a21(x2*x2 + y2*y2 + z2*z2), a31(x3*x3 + y3*y3 + z3*z3), a41(x4*x4 + y4*y4 + z4*z4);

	real A[5][5];	

	A[1][1] = x1; A[1][2] = y1; A[1][3] = z1; A[1][4] = 1.0;
	A[2][1] = x2; A[2][2] = y2; A[2][3] = z2; A[2][4] = 1.0;
	A[3][1] = x3; A[3][2] = y3; A[3][3] = z3; A[3][4] = 1.0;
	A[4][1] = x4; A[4][2] = y4; A[4][3] = z4; A[4][4] = 1.0;
	M11 = Determinant4(A);

	if (abs(M11)<real(_tol)) {
		//printf("\n Warning 0 at Circumsphere()...\n");
		return -1;	
	}

	A[1][1] = a11; A[1][2] = y1; A[1][3] = z1; A[1][4] = 1.0;
	A[2][1] = a21; A[2][2] = y2; A[2][3] = z2; A[2][4] = 1.0;
	A[3][1] = a31; A[3][2] = y3; A[3][3] = z3; A[3][4] = 1.0;
	A[4][1] = a41; A[4][2] = y4; A[4][3] = z4; A[4][4] = 1.0;
	M12 = Determinant4(A);

	A[1][1] = a11; A[1][2] = x1; A[1][3] = z1; A[1][4] = 1.0;
	A[2][1] = a21; A[2][2] = x2; A[2][3] = z2; A[2][4] = 1.0;
	A[3][1] = a31; A[3][2] = x3; A[3][3] = z3; A[3][4] = 1.0;
	A[4][1] = a41; A[4][2] = x4; A[4][3] = z4; A[4][4] = 1.0;
	M13 = Determinant4(A);


	A[1][1] = a11; A[1][2] = x1; A[1][3] = y1; A[1][4] = 1.0;
	A[2][1] = a21; A[2][2] = x2; A[2][3] = y2; A[2][4] = 1.0;
	A[3][1] = a31; A[3][2] = x3; A[3][3] = y3; A[3][4] = 1.0;
	A[4][1] = a41; A[4][2] = x4; A[4][3] = y4; A[4][4] = 1.0;
	M14 = Determinant4(A);


	A[1][1] = a11; A[1][2] = x1; A[1][3] = y1; A[1][4] = z1;
	A[2][1] = a21; A[2][2] = x2; A[2][3] = y2; A[2][4] = z2;
	A[3][1] = a31; A[3][2] = x3; A[3][3] = y3; A[3][4] = z3;
	A[4][1] = a41; A[4][2] = x4; A[4][3] = y4; A[4][4] = z4;
	M15 = Determinant4(A);

	xo = real(0.5)*M12 / M11;
	yo = real(-0.5)*M13 / M11;
	zo = real(0.5)*M14 / M11;

	real rrr = xo*xo + yo*yo + zo*zo - M15 / M11;
#ifdef DEBUG
	real dist1, dist2, dist3, dist4;
	dist1 = cuDist(xo, yo, zo, x1, y1, z1);
	dist2 = cuDist(xo, yo, zo, x2, y2, z2);
	dist3 = cuDist(xo, yo, zo, x3, y3, z3);
	dist4 = cuDist(xo, yo, zo, x4, y4, z4);

	if (abs(dist1 - rrr)>real(_tol) || abs(dist2 - rrr)>real(_tol) || abs(dist3 - rrr)>real(_tol) || abs(dist4 - rrr)>real(_tol)){
		printf("\n Error 0 at Circumsphere(). Difference are (%f, %f, %f, %f) \n", abs(dist1 - rrr), abs(dist2 - rrr), abs(dist3 - rrr), abs(dist4 - rrr));
	}	
#endif

	return rrr;
}
