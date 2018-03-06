//Extracting tets for points delaunay connectivity 
//TODO test extractTet and make sure it runs correctly

#include <vector>

std::vector<uint32_t> commonElements2(const uint32_t v1, const uint32_t v2, uint32_t* const h_delaunay, const int MaxOffset){

	//find the common vertices between v1 and v2 
	std::vector<uint32_t> common;
	const uint32_t base_v1 = v1*MaxOffset;
	const uint32_t base_v2 = v2*MaxOffset;
	const uint32_t count_v1 = h_delaunay[base_v1];
	const uint32_t count_v2 = h_delaunay[base_v2];

	for (uint32_t i = 1; i <= count_v1; i++){
		const uint32_t pv1 = h_delaunay[i];
		if (pv1 == v2){ continue; }
	
		for (uint32_t j = 1; j <= count_v2; j++){
			const uint32_t pv2 = h_delaunay[j];
			if (pv1 == pv2){
				common.push_back(pv1);
			}
		}
	}	
	return common;
}

std::vector<uint32_t> commonElements3(const uint32_t v1, const uint32_t v2, const uint32_t v3, uint32_t* const h_delaunay, const int MaxOffset){

	//find the common vertices between v1 and v2 and v3 
	std::vector<uint32_t> common;
	const uint32_t base_v1 = v1*MaxOffset;
	const uint32_t base_v2 = v2*MaxOffset;
	const uint32_t base_v3 = v3*MaxOffset;
	const uint32_t count_v1 = h_delaunay[base_v1];
	const uint32_t count_v2 = h_delaunay[base_v2];
	const uint32_t count_v3 = h_delaunay[base_v3];

	for (uint32_t i = 1; i <= count_v1; i++){
		const uint32_t pv1 = h_delaunay[base_v1 + i];
		if (pv1 == v2 || pv1 == v3){ continue; }
		for (uint32_t j = 1; j <= count_v2; j++){
			const uint32_t pv2 = h_delaunay[base_v2 + j];			
			if (pv2 == v1 || pv2 == v3){ continue; }			
			if (pv1 != pv2){ continue; }
			for (uint32_t k = 1; k <= count_v3; k++){				
				const uint32_t pv3 = h_delaunay[base_v3 + k];
				if (pv3 == v1 || pv3 == v2){ continue; }
				if (pv3 == pv1){//also pv3=pv2 if true since pv1=pv2
					common.push_back(pv3);
				}
			}
		}
	}
	return common;
}


std::vector<std::vector<uint32_t>> extractTets(const int NumPoints, uint32_t* const h_delaunay, const int MaxOffset){
	
	std::vector<std::vector<uint32_t>> myTets;

	for (uint32_t p = 0; p < NumPoints; p++){
		uint32_t base = p*MaxOffset; //the base inside h_delaunay 
		uint32_t numConnect = h_delaunay[base]; //number of points p is connect to 
		
		if (numConnect <3){ continue; }

		for (uint32_t i = 1; i <= numConnect - 2; i++){
			uint32_t p1 = h_delaunay[base + i];

			//get the common vertices that p and p1 are connected to
			std::vector<uint32_t> common2 = commonElements2(p, p1, h_delaunay, MaxOffset);
			if (common2.size() == 0){ continue; }

			for (uint32_t j = 0; j < common2.size(); j++){
				uint32_t p2 = common2[j];

				//get the common vertices that p, p1, p2 are connected to
				std::vector<uint32_t> common3 = commonElements3(p, p1, p2, h_delaunay, MaxOffset);
				
				if (common3.size() == 0){ continue; }

				if (common3.size() != 1){
					printf("\n Error at extractTets()\n!!!!");
				}

				//creat a tet 
				std::vector<uint32_t> tet(4);
				tet[0] = p;
				tet[1] = p1;
				tet[2] = p2;
				tet[3] = common3[0];

				myTets.push_back(tet);

			}

		}
	}

	return myTets;
}