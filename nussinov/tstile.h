void tstile()
{

int c0,c1,c2,c3,c5,c6,c7,c9,c11,c10,c4,c12;

if(1==1)
for( c0 = 0; c0 <= floord(N - 2, 8); c0 += 1)
  #pragma omp parallel for schedule(dynamic, 1) shared(c0) private(c1,c3,c4,c6,c10)
  for( c1 = (c0 + 1) / 2; c1 <= min(c0, (N - 1) / 16); c1 += 1)
    for( c3 = 16 * c0 - 16 * c1 + 1; c3 <= min(min(N - 1, 16 * c1 + 15), 16 * c0 - 16 * c1 + 16); c3 += 1) {
      for( c4 = 0; c4 <= c0 - c1; c4 += 1)
        for( c6 = max(-N + 16 * c1 + 1, -N + c3 + 1); c6 <= min(0, -N + 16 * c1 + 16); c6 += 1) {
          for( c10 = 16 * c4; c10 <= min(c3 - 1, 16 * c4 + 15); c10 += 1)
            S[(-c6)][(c3-c6)] = MAX(S[(-c6)][c10+(-c6)] + S[c10+(-c6)+1][(c3-c6)], S[(-c6)][(c3-c6)]);
          if (c1 + c4 == c0 && 16 * c0 + c6 + 15 >= 16 * c1 + c3)
            S[(-c6)][(c3-c6)] = MAX(S[(-c6)][(c3-c6)], S[(-c6)+1][(c3-c6)-1] + can_pair(RNA, (-c6), (c3-c6)));
        }
      for( c4 = max(c0 - c1 + 1, -c1 + (N + c3) / 16 - 1); c4 <= min((N - 1) / 16, -c1 + (N + c3 - 1) / 16); c4 += 1)
        for( c6 = max(max(-N + 16 * c1 + 1, -N + c3 + 1), c3 - 16 * c4 - 15); c6 <= min(-N + 16 * c1 + 16, c3 - 16 * c4); c6 += 1)
          S[(-c6)][(c3-c6)] = MAX(S[(-c6)][(c3-c6)], S[(-c6)+1][(c3-c6)-1] + can_pair(RNA, (-c6), (c3-c6)));
    }

                                                                            
}
