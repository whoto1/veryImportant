//#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define BILLION 1000000000
#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))
#define N 1024
int main(int argc, char** argv) {
	struct timespec start, stop;
	clock_gettime(CLOCK_REALTIME, &start);
	double delta;
	int m, iter_max;
	double tol;
	sscanf(argv[1], "%d", &m);
	sscanf(argv[2], "%d", &iter_max);
	sscanf(argv[3], "%lf", &tol);

	/*int m = 128; int iter_max = 1000000;
	double tol = 1e-6;*/
	double** arr = (double**)malloc(N * sizeof(double*)); // creatin array
	#pragma acc loop
	for(int i = 0; i < N; i++)//initializing array
	{
		arr[i] = (double*)malloc(2* N * sizeof(double));
		for(int j = 0; j < m; j++)
		{
			arr[i][j] = 0;
		}
	}
	//initializing boundaries
	arr[0][0] = arr[0][m] = 10;
	arr[0][m-1] = arr[0][2*m-1]= 20;
	arr[m-1][0] = arr[m-1][m] = 30;
	arr[m-1][m-1] = arr[m-1][2*m-1] = 20;
	//coefficients for the interpolation
	double top = (arr[0][m - 1] - arr[0][0]) / (m - 1);
	double btm = (arr[m - 1][m - 1] - arr[m - 1][0]) / (m - 1);
	double lft = (arr[m - 1][0] - arr[0][0]) / (m - 1);
	double rght = (arr[m - 1][m - 1] - arr[0][m - 1]) / (m - 1);
	#pragma acc parallel loop
	for(int j = 1; j < m-1; j++){ // interpolation
		arr[0][j] = arr[0][j+m] = j * top + arr[0][0];   	//top
		arr[m-1][j] = arr[m-1][j+m]= j * btm + arr[m-1][0]; 	//bottom
		arr[j][0] = arr[j][m]= j * lft + arr[0][0]; 		//left
		arr[j][m-1] = arr[j][2*m-1]= rght * j + arr[0][m-1]; 	//right
	}
	int iter = 0;

	double err = tol + 1;
	#pragma acc data copy(arr[:m][:(2*m)]) 

	{
		int right=0, left=m;
	while(err > tol && iter < iter_max){
		err = 0;
		if (!(iter % 2)) //swapping arrays
		{//left to right
			left = 0;
			right = m;
		}
		else //right to left
		{
			left = m;
			right = 0;
		}
		#pragma acc parallel loop reduction(max:err) 
		for(int j = 1; j < m - 1; j++)	{
			#pragma acc loop reduction(max:err)
			for(int i = 1; i < m - 1; i++){
				arr[i][j+left] = 0.25 * (arr[i+1][j+right] + arr[i-1][j+right]
							+ arr[i][j-1 +right] + arr[i][j+1+right]);
				err = fmax(err, fabs(arr[i][j+m] - arr[i][j]));
			}
		}

		if (iter % 100 == 0) {
			printf("%d, %0.6lf\n", iter, err);
		}
		iter++;
	}
	}
	clock_gettime(CLOCK_REALTIME, &stop);
	delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;
	printf("time %lf\n", delta);
	printf("Final result: %d, %0.6lf\n", iter, err);
	#pragma acc parallel loop
	for (int i = 0; i < N; i++) {
		free(arr[i]);
	}
	free(arr);
	//printf("%d%d%lf", m, iter_max, tol);
	return 0;
}
