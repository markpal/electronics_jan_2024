// compilation:  nvc -mp=gpu nuss_pluto_openmp.c -O3

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

// 5000 46.41
// 7500 109.12
// 10000  198.98
// 15000  453.88
// 20000

#define N 1000

double S[N][N];

int sigma(int a, int b) {
    return a + b;  // Placeholder logic. Adjust as needed.
}

int max1(int a, int b) {
    return a > b ? a : b;  // Placeholder logic.
}

int main() 
{
    int t2,t4,t6,lbp,ubp;
    struct timeval tv1, tv2;
    struct timezone tz;
	double elapsed; 

    gettimeofday(&tv1, &tz);
    

 omp_set_num_teams(4096);
  omp_set_num_threads(512);




    #pragma omp target map(tofrom:S) 
    if (N >= 2) {
  for (t2=1;t2<=N-1;t2++) {
    lbp=t2;
    ubp=N-1;
#pragma omp parallel for private(t4,t6) shared(t2,lbp,ubp)
    for (t4=lbp;t4<=ubp;t4++) {
      for (t6=0;t6<=t2-1;t6++) {
        S[(-t2+t4)][t4] = max1(S[(-t2+t4)][t6+(-t2+t4)] + S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
      }
      S[(-t2+t4)][t4] = max1(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1] + sigma((-t2+t4), t4));
    }
  }
}
	
	
    gettimeofday(&tv2, &tz);
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("elapsed time = %f seconds. threads\n%i\n", elapsed, omp_get_max_threads);
}
 
// #pragma omp target  map(tofrom:S) teams  distribute parallel for   private(t4,t6) shared(t2,lbp,ubp) 
