#include <string>

long double **mem()
{
int i;
long double **S = new long double*[N+5]();
for(i=0; i<N+5; i++)
  S[i] = new long double[N+5]();

return S;

}

void saveTable(char *method, int threads, char *sequence)
{
  int i,j;
  char *wout = new char[256];
  strcpy(wout, sequence);

  FILE *plik = fopen(strcat(wout, ".out.txt") ,"w");
  fprintf(plik, "method: %s %i %s \n", method, threads, sequence);
  for(i=0; i<N-1; i++){
   fprintf(plik, "0 ");
    for(j=0; j<N-1; j++)
      fprintf(plik, "%li ", S[i][j]);
    fprintf(plik, "\n");
  }
  fclose(plik);

  printf("Table saved: %s \n", wout);

}

void saveTable()
{
  int i,j;
  for(i=0; i<N-1; i++){
   printf("0 ");
    for(j=0; j<N-1; j++)
      printf("%li ", S[i][j]);
    printf("\n");
  }

}

int can_pair(char*input, int a, int b){

   return (((
   (input[a] == 'A' &&  input[b] == 'U') || (input[a] == 'U' &&  input[b] == 'A') ||
  (input[a] == 'G' &&  input[b] == 'C') ||  (input[a] == 'C' &&  input[b] == 'G') ||
  (input[a] == 'G' &&  input[b] == 'U') || (input[a] == 'U' &&  input[b] == 'G')
   )) )/* && (a < b - 1))) */ ? 1 : 0;
}



void readRNA(char *filename){

    int i, offset;

    FILE *plik = fopen(filename, "r");


    fscanf(plik, "%*[^\n]\n", NULL);  //skip first line FASTA
    offset = ftell(plik);
    fseek(plik, 0L, SEEK_END);
    N = ftell(plik) - offset+1;
    printf("Sequence length: %d\n", N);

    rewind(plik);
    fscanf(plik, "%*[^\n]\n", NULL);  //skip first line FASTA

    RNA = new char[N+5];

    for(i=0; i<N; i++){
      fscanf(plik, "%c", &RNA[i]);
      if(RNA[i] == 'T') RNA[i] = 'U';
    }

    for(i=0; i<N; i++){
     //    printf("%c", RNA[i]);
    }

    fclose(plik);


}


/*

srand(time(NULL));
for(i=0; i<N; i++){
   ll = rand()%4;
   switch(ll){
     case 0 : RNA[i] = 'A'; break;
     case 1 : RNA[i] = 'G'; break;
     case 2 : RNA[i] = 'C'; break;
     case 3 : RNA[i] = 'U'; break;
   }
//   printf("%c", RNA[i]);
  RNA[i] = ' ';
}

*/
