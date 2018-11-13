#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

void mat_initialize(double **mat, int N) {
  for (int i = N - 1; i >= 0; --i)
    for (int j = N - 1; j >= 0; --j)
      mat[i][j] = (1 + cos(2 * i) + sin(j)), mat[i][j] *= mat[i][j];
}

inline void find_small2(const double &a, const double &b, const double &c,
                        const double &d, double &res) {
  double slot[4];
  if (a < b)
    slot[0] = a, slot[1] = b;
  else
    slot[0] = b, slot[1] = a;

  if (c < d)
    slot[2] = c, slot[3] = d;
  else
    slot[2] = d, slot[3] = c;

  res = slot[0] < slot[2] ? fmin(slot[1], slot[2]) : fmin(slot[0], slot[3]);
}

void matrix_update(int N) {
  double **A = (double **)malloc(N * sizeof(double *));
  double **B = (double **)malloc(N * sizeof(double *));

  for (int i = 0; i < N; i++) A[i] = (double *)malloc(N * sizeof(double));
  for (int i = 0; i < N; i++) B[i] = (double *)malloc(N * sizeof(double));

  // initialize
  mat_initialize(A, N);

  std::clock_t start = std::clock();

  for (int it = 0; it < 10; ++it) {
    for (int i = N - 2; i > 0; --i)
      for (int j = N - 2; j > 0; --j)
        find_small2(A[i - 1][j - 1], A[i - 1][j + 1], A[i + 1][j - 1],
                    A[i + 1][j + 1], B[i][j]);

    for (int i = N - 2; i > 0; --i)
      for (int j = N - 2; j > 0; --j) A[i][j] += B[i][j];
  }

  double sum{0};
  for (int i = N - 1; i >= 0; --i)
    for (int j = N - 1; j >= 0; --j) sum += A[i][j];

  /* end timing */
  double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  cout << "sum = " << sum << " A[m][m] " << A[N / 2][N / 2] << " A[37][47] "
       << A[37][47] << " running time: " << duration << endl;

  for (int i = 0; i < N; i++) free((double *)A[i]);
  for (int i = 0; i < N; i++) free((double *)B[i]);

  free((double **)A);
  free((double **)B);
}

int main() { matrix_update(500); }
