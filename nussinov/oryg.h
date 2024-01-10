void oryg(){

int i,j,k;



    for (i = N-1; i >= 0; i--) {
     if(i % 100==0)
       printf("%i \n", i);
     for (j = i+1; j < N; j++) {
      for (k = 0; k < j-i; k++) {
        S[i][j] = max(S[i][k+i] + S[k+i+1][j], S[i][j]);
      }
      for (k = 0; k < 1; k++) {
       S[i][j] = max(S[i][j], S[i+1][j-1]  + can_pair(RNA, i, j));

     }
    }
   }


}
