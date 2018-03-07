#pragma once

#include "spokes.cu"
#include <string>
#include <algorithm>
#include <time.h>
#include <stdint.h>

class kdtree
{
public:
	kdtree();
	~kdtree();
//	void bulkBuild(real**&Points);
	double bulkBuild(real3*&pointsExt, size_t numPointsExt);
	void   bulkBuild(size_t start, size_t end, size_t iDim);

	void treePointsInsideSphereBF(size_t iPoint, real r, uint32_t*& inside, uint32_t& numInside);
	void treePointsInsideSphere(size_t iPoint, real r, uint32_t* inside, uint32_t& numInside, size_t idx = 0);
	void addToKdtree(size_t idx, real3& pt);

	
	// helper
	void print_error(size_t line){
		std::string fname = __FILE__;
		fname = fname.substr(fname.find_last_of("/\\") + 1);
		printf("Error in File: %s, Line %lu. \n", fname.c_str(), line);
		//system("pause");
	}

private:
	struct node
	{
		size_t pve;
		size_t nve;
	};

	node* pointsTree;
	real3* points;
	size_t numPoints;
};

