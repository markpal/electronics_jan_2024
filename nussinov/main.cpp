//nussinov

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <string>

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define ceild(n,d) ceil(((double)(n))/((double)(d)))



long double **S;
char *RNA;
int N;

#include "library.h"
#include "oryg.h"
#include "tstile.h"
#include "tilecorr.h"
#include "pluto.h"
#include "dapt.h"


int main(int argc, char *argv[]){

    int i,j,k;
    char *filename, *method;
    int num_proc=-1;

    if(argc > 1)
        num_proc = atoi(argv[1]);

    int kind=1;

    N = 8;
    
    if(argc > 2)
        N = atoi(argv[2]);
    
    if(argc > 3)
        kind = atoi(argv[3]);

    omp_set_num_threads(num_proc);  // else default max number
    
    S = mem();
    RNA = new char[N+5];

    //printf("\nmethod %s\n", method);
    //printf("N %i\n", N);

    double start = omp_get_wtime();


    if(kind==1)
       oryg();
    if(kind==4)
       tstile();

    if(kind==3)
       tilecorr();

    if(kind==2)
       pluto();

    if(kind==5)
       dapt();


    double stop = omp_get_wtime();
   // saveTable(method, num_proc, filename);
    //saveTable();



    printf("Time: %.2f\n", stop - start);

   // printf("Traceback:\n");
   //  char *wout = new char[256];
  //   strcpy(wout, filename);

   //  FILE *plik = fopen(strcat(wout, ".traceback.txt") ,"w");
   //  traceback(0, N-1, plik);
   //  fclose(plik);

    return 0;
}

