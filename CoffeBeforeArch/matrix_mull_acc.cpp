#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_matrix(int *m, int N) {
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        m[i] = rand() % 100;
    }
}

int main() {
    int N = 1 << 10;

    int *a = (int*)malloc(N * N * sizeof(int));
    int *b = (int*)malloc(N * N * sizeof(int));
    int *c = (int*)malloc(N * N * sizeof(int));
    if (!a || !b || !c) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    init_matrix(a, N);
    init_matrix(b, N);

    #pragma acc kernels copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
        #pragma acc loop independent
        for (int i = 0; i < N; ++i) {
            #pragma acc loop independent
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                #pragma acc loop independent reduction(+:sum)
                for (int k = 0; k < N; ++k) {
                    sum += a[i * N + k] * b[N * k + j];
                }
                c[i * N + j] = sum;
            }
        }
    }

    free(a);
    free(b);
    free(c);

    return 0;
}
