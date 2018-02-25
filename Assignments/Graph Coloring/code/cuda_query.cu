
void cuda_query(const int dev){

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0){
		printf("\n deviceCount is zero. I quit!!!");
		exit(EXIT_FAILURE);
	}

	//const int dev = (deviceCount == 1) ? 0 : 3;

	cudaSetDevice(dev);
	/*cudaDeviceProp devProp;

	HANDLE_ERROR(cudaGetDeviceProperties(&devProp, dev));
	printf("\n  Total number of device: %d", deviceCount);
	printf("\n  Using device Number: %d", dev);
	printf("\n  Device name: %s", devProp.name);
	//printf("\n  devProp.major: %d", devProp.major);
	//printf("\n  devProp.minor: %d", devProp.minor);
	if(devProp.major==1){//Fermi
		if(devProp.minor==1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*48);
		}else{
			printf("\n  SM Count: %d", devProp.multiProcessorCount*32);
		}
	}else if(devProp.major==3){//Kepler
		printf("\n  SM Count: %d", devProp.multiProcessorCount*192);
	}else if(devProp.major==5){//Maxwell
		printf("\n  SM Count: %d", devProp.multiProcessorCount*128);
	}else if(devProp.major==6){//Pascal
		if(devProp.minor==1){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*128);
		}else if(devProp.minor==0){
			printf("\n  SM Count: %d", devProp.multiProcessorCount*64);
		}
	}

	printf("\n  Compute Capability: v%d.%d", (int)devProp.major, (int)devProp.minor);
	printf("\n  Memory Clock Rate: %d(kHz)", devProp.memoryClockRate);
	printf("\n  Memory Bus Width: %d(bits)", devProp.memoryBusWidth);
	const double maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth/8.0)/1.0E3;
	printf("\n  Peak Memory Bandwidth: %f(MB/s)\n\n", maxBW);*/
}