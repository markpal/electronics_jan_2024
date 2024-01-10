void pluto()
{


  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if (N >= 2) {
  for (t2=max(-1,ceild(-N-13,16));t2<=floord(N-1,16);t2++) {
    lbp=max(0,t2);
    ubp=min(floord(N-1,16),floord(16*t2+N+13,16));
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9,t10)
    for (t4=lbp;t4<=ubp;t4++) {
      for (t5=max(max(-N+2,16*t2-16*t4),-16*t4-14);t5<=min(0,16*t2-16*t4+15);t5++) {
        for (t7=max(16*t4,-t5+1);t7<=min(N-1,16*t4+15);t7++) {
          for (t9=0;t9<=t5+t7-1;t9++) {
            S[-t5][t7] = MAX(S[-t5][t9+-t5] + S[t9+-t5+1][t7], S[-t5][t7]);;
          }
          S[-t5][t7] = MAX(S[-t5][t7], S[-t5+1][t7-1] + can_pair(RNA, -t5, t7));;
        }
      }
    }
  }
}
/* End of CLooG code */























}
