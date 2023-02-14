#include <malloc.h>
#include <stdio.h>
#include <math.h>
#define M_PI 3.14159265358979323846

#define N 10000000
//// d
//float FuncArrayFloat(float* my_array, int len)
//{
//	float sum = 0;
//	float w = (2 * M_PI) / N;
//	#pragma acc kernels
//	{
//		for (int i = 0; i < len; i++) {
//			my_array[i] = (float)(sin(i * w));
//			sum += my_array[i];
//		}
//	}
//	
//	return sum;
//}



int main()
{
    double* arrayD = (double*)malloc(sizeof(double*)*N);
	double* my_arrayD = (double*)malloc(sizeof(double*) * N);
	float* arrayF = (float*)malloc(sizeof(float*) * N);
    int len = N;
	double sumD = 0;
	float sumF = 0;
	#pragma acc data create(my_arrayD[:N]) //copy(sum) copyin(w)
	{
		double w = (2 * M_PI) / N;
#pragma acc kernels
		{
			for (int i = 0; i < len; i++) {

				my_arrayD[i] = (double)(sin(i * w));
			}
		}

#pragma acc kernels //parallel loop reduction(+:sum)
		{
			for (int i = 0; i < len; i++) {

				sumD += my_arrayD[i];
			}
		}
	}
	printf("%le\n", sumD);
	free(arrayD);
	//free(arrayF);
}
