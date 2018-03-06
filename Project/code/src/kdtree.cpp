#include "kdtree.h"

kdtree::kdtree(){

}


kdtree::~kdtree(){
	// destroy tree
	delete[] pointsTree;
}

double kdtree::bulkBuild(real3*&pointsExt, size_t numPointsExt){

	clock_t start = clock();
	points = pointsExt;
	numPoints = numPointsExt;

	pointsTree = new node[numPoints];
	for (size_t iPoint = 0; iPoint < numPoints; ++iPoint)
	{
		pointsTree[iPoint].pve = 0;
		pointsTree[iPoint].nve = 0;
	}

	bulkBuild(0, numPoints - 1, 0);
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
	return time;
}
void kdtree::bulkBuild(size_t start, size_t end, size_t iDim){

	iDim = (iDim == 3) ? 0 : iDim;
	if (start == end)
	{
		addToKdtree(start, points[start]);
		return;
	}

	std::sort(points + start, points + end,
		[iDim](real3& a, real3& b) -> bool
	{
		return a[iDim] < b[iDim];
	});

	size_t mid = size_t(0.5*double(start + end));
	std::swap(points[start], points[mid]);

	if (start != 0)
		addToKdtree(start, points[start]);
	start++;

	if (start <= mid)
		bulkBuild(start, mid, iDim + 1);
	if (mid + 1 <= end)
		bulkBuild(mid + 1, end, iDim + 1);
}
void kdtree::treePointsInsideSphere(size_t iPoint, real r, uint32_t* inside, uint32_t& numInside, size_t idx){
	
	size_t idim = idx % 3;

	real delta = points[iPoint][idim] - points[idx][idim];
	bool check_both = r > fabs(delta);

	if (check_both)
	{
		real dx = points[iPoint][0] - points[idx][0];
		real dy = points[iPoint][1] - points[idx][1];
		real dz = points[iPoint][2] - points[idx][2];

		real dist = dx * dx + dy * dy + dz * dz;

		if (dist < r*r && iPoint != idx)
			inside[numInside++] = uint32_t(idx);

		if (pointsTree[idx].pve) treePointsInsideSphere(iPoint, r, inside, numInside, pointsTree[idx].pve);
		if (pointsTree[idx].nve) treePointsInsideSphere(iPoint, r, inside, numInside, pointsTree[idx].nve);
		return;
	}
	else if (delta > 0 && pointsTree[idx].pve)
	{
		treePointsInsideSphere(iPoint, r, inside, numInside, pointsTree[idx].pve);
	}
	else if (delta < 0 && pointsTree[idx].nve)
	{
		treePointsInsideSphere(iPoint, r, inside, numInside, pointsTree[idx].nve);

	}

}
void kdtree::treePointsInsideSphereBF(size_t iPoint, real r, uint32_t*& inside, uint32_t& numInside){

	for (size_t in = 0; in < numPoints; in++)
	{
		real dx = points[iPoint][0] - points[in][0];
		real dy = points[iPoint][1] - points[in][1];
		real dz = points[iPoint][2] - points[in][2];

		real dist = dx * dx + dy * dy + dz * dz;

		if (dist < r*r && iPoint != in)
			inside[numInside++] = uint32_t(in);
	}

}

void kdtree::addToKdtree(size_t idx, real3& pt){

	size_t inode = 0;
	while (true)
	{
		size_t idim = inode % 3;
		double diff = pt[idim] - points[inode][idim];
		if (diff == 0.0)
		{
			pt[idim] = pt[idim] + real(1e-5);
			continue;

			printf("%f %f %f\n", pt[0], pt[1], pt[2]);
			printf("%f %f %f\n", points[inode][0], points[inode][1], points[inode][2]);
			
			print_error(__LINE__);
		}
		size_t leaf;
		if (diff > 0)
		{
			leaf = pointsTree[inode].pve;
			if (leaf == 0)
			{
				pointsTree[inode].pve = idx;
				return;
			}
		}
		else
		{
			leaf = pointsTree[inode].nve;
			if (leaf == 0)
			{
				pointsTree[inode].nve = idx;
				return;
			}
		}
		inode = leaf;
	}


}