#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define M_PI 3.14159265358979323846
#define N 10000000
#define BILLION 1000000000.0


int main()
{
    float* my_arrayD = (float*)malloc(sizeof(float*) * N);
    float sum = 0;
    float w = (2 * M_PI) / N;
    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME, &start);
#pragma acc data create(my_arrayD[:N]) copy(sum) copyin(w)
    {
#pragma acc kernels 
        {
            for (int i = 0; i < N; i++)
                my_arrayD[i] = (sinf(i * w));
        }
    
#pragma acc kernels 
    {
        for (int i = 0; i < N; i++)
            sum += my_arrayD[i];
    }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/BILLION;
    printf("time %f\n", time_spent);
    printf("sum = %-32.25f\n", sum);
    return 0;
}
