#define sigma(i, j) (can_pair(RNA, i, j))
void dapt(){


int n = N;
int c0,c1,c2,c3,c5,c6,c7,c9,c11,c10,c4,c12;
for (int c0 = floord(-31 * n + 115, 3132) + 2; c0 <= floord(79 * n - 158, 2436) + 2; c0 += 1) {
    #pragma omp parallel for 
    for (int c1 = max(-c0 - (n + 52) / 54 + 2, -((n + 114) / 116)); c1 <= min(min(-c0 + (n - 2) / 42 + 1, c0 + ((-4 * c0 + 3)/31) - 1), (-21 * c0 + 20)/79); c1 += 1) {
      for (int c2 = max(-c0 + c1 + floord(21 * c0 - 17 * c1 - 21, 48) + 1, -c0 - c1 - (n - 42 * c0 - 42 * c1 + 136) / 96 + 1); c2 <= min(min(-1, -c0 - c1), -((27 * c0 - 31 * c1 + 54) / 69) + 1); c2 += 1) {
        for (int c5 = max(27 * c0 - 31 * c1 + 27 * c2 - 83, -42 * c2 - 41); c5 <= min(min(n + 54 * c0 + 54 * c1 + 54 * c2 - 1, -42 * c2), 54 * c0 - 62 * c1 + 54 * c2); c5 += 1) {
          for (int c6 = max(-54 * c0 - 54 * c1 - 54 * c2, -116 * c1 - 2 * c5 - 114); c6 <= min(min(-54 * c0 - 54 * c1 - 54 * c2 + 53, n - c5 - 1), -116 * c1 - c5); c6 += 1) {
            for (int c7 = max(-116 * c1 - 115, c5 + c6); c7 <= min(min(n - 1, -116 * c1), 2 * c5 + c6 - 1); c7 += 1) {
              if (2 * c5 + c6 >= c7 + 2) {
                S[c6][c7] = MAX(S[c6][-c5 + c7] + S[-c5 + c7 + 1][c7], S[c6][c7]);
                if (c7 == c5 + c6) {
                  S[c6][c5 + c6] = MAX(S[c6][c5 + c6], S[c6 + 1][c5 + c6 - 1] +  sigma(c6, c5 + c6));
                }
              }
              S[c6][c7] = MAX(S[c6][c5 + c6 - 1] + S[c5 + c6][c7], S[c6][c7]);
              if (c7 == c5 + c6) {
                S[c6][c5 + c6] = MAX(S[c6][c5 + c6], S[c6 + 1][c5 + c6 - 1] + sigma(c6, c5 + c6));
              }
            }
          }
        }
      }
    }
  }


}