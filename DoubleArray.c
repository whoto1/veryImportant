#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define M_PI 3.14159265358979323846
#define N 10000000
#define BILLION 1000000000.0




int main()
{
	double* my_arrayD = (double*)malloc(sizeof(double*) * N);
	double sumD = 0;
	double w = (2 * M_PI) / N;
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);
#pragma acc data create(my_arrayD[:N]) copy(sumD) copyin(w)
	{
		
#pragma acc kernels
		{
			for (int i = 0; i < N; i++) {

				my_arrayD[i] = (sin(i * w));
			}
		}

#pragma acc kernels
		{
			for (int i = 0; i < N; i++) {

				sumD += my_arrayD[i];
			}
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);
	double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/BILLION;
	printf("time=%f\n", time);
	printf("sum = %32.25f\n", sumD);
	return 0;
}
