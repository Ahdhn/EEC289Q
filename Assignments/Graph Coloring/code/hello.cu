__global__
void hello()
{
	printf("hello I am %d\n", threadIdx.x);
}
