// compilation:  nvc -mp=gpu nuss_pluto_openmp.c -O3

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>


#define N 7500

double S[N][N];

int sigma(int a, int b) {
    return a + b;  // Placeholder logic. Adjust as needed.
}

int max1(int a, int b) {
    return a > b ? a : b;  // Placeholder logic.
}

int main() 
{
    int n = N, c1, c3, c5;
    struct timeval tv1, tv2;
    struct timezone tz;
	double elapsed; 

    gettimeofday(&tv1, &tz);
    

 omp_set_num_teams(4096);
  omp_set_num_threads(512);


	#pragma omp target map(tofrom:S) 
	for( c1 = 1; c1 < 2 * n - 2; c1 += 1)
       #pragma omp parallel for private(c3,c5) shared(c1) 
        for( c3 = max1(0, -n + c1 + 1); c3 < (c1 + 1) / 2; c3 += 1) {      
           for( c5 = 0; c5 <= c3; c5 += 1)
             S[(n-c1+c3-1)][(n-c1+2*c3)] = max1(S[(n-c1+c3-1)][(n-c1+c3+c5-1)] + S[(n-c1+c3+c5-1)+1][(n-c1+2*c3)], S[(n-c1+c3-1)][(n-c1+2*c3)]);
          S[(n-c1+c3-1)][(n-c1+2*c3)] = max1(S[(n-c1+c3-1)][(n-c1+2*c3)], S[(n-c1+c3-1)+1][(n-c1+2*c3)-1] + sigma((n-c1+c3-1), (n-c1+2*c3)));
    }
	
	
    gettimeofday(&tv2, &tz);
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("elapsed time = %f seconds. threads\n%i\n", elapsed, omp_get_max_threads);
}
 
