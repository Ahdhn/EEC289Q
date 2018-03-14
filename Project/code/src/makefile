INCLUDES := -I src/ -I /usr/include/ -I /usr/local/cuda/include/ -I ./
NVCC_OPTIONS := -O3 -gencode arch=compute_35,code=sm_35 -DCUDA_ARCH=35 -lcudart -std=c++11
objects := RSD3d.o kdtree.o

all: $(objects)
	nvcc $(NVCC_OPTIONS) $(INCLUDES) -o rsd.out	RSD3d.o kdtree.o
RSD3d.o: RSD3d.cu *.cu circumSphere.h explore.cu explore_sh.cu extractTets.h propagate.cu RSD_imp.cu spokes.cu utilities.h validate.h sortNeighbours.cu defines.h
	nvcc $(NVCC_OPTIONS) $(INCLUDES) -c RSD3d.cu -o RSD3d.o
kdtree.o: kdtree.cpp
	nvcc $(NVCC_OPTIONS) $(INCLUDES) -c kdtree.cpp -o kdtree.o
clean:
	rm -f *.o *.out