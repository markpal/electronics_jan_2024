#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Mock definition for can_pair, replace with actual logic
int can_pair(int* RNA, int a, int b) {
    return a + b; // Placeholder
}

__device__ int can_pair_device(int* RNA, int a, int b) {
    return a + b; // Placeholder, update with actual logic
}

__global__ void computeS_kernel(int N, int t2, int* d_S, int* d_RNA) {
    int t4 = blockIdx.x * blockDim.x + threadIdx.x + t2;
    
    if(t4 <= N-1) {
        for (int t6=0; t6<=t2-1; t6++) {
            d_S[(-t2+t4)*N + t4] = MAX(d_S[(-t2+t4)*N + t6+(-t2+t4)] + d_S[(t6+(-t2+t4)+1)*N + t4], d_S[(-t2+t4)*N + t4]);
        }
        d_S[(-t2+t4)*N + t4] = MAX(d_S[(-t2+t4)*N + t4], d_S[(-t2+t4+1)*N + t4-1] + can_pair_device(d_RNA, (-t2+t4), t4));
    }
}

int main() {
    int N =10000;
    int *h_S, *d_S, *h_RNA, *d_RNA, *cpu_S;

    // Initialize and allocate memory
    h_S = (int*)malloc(N * N * sizeof(int));
    cpu_S = (int*)malloc(N * N * sizeof(int));
    h_RNA = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N * N; i++) {
        h_S[i] = rand() % 100;
        cpu_S[i] = h_S[i];
    }
    for (int i = 0; i < N; i++) {
        h_RNA[i] = rand() % 100;
    }

    cudaMalloc(&d_S, N * N * sizeof(int));
    cudaMalloc(&d_RNA, N * sizeof(int));

    cudaMemcpy(d_S, h_S, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RNA, h_RNA, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks;

    if (N >= 2 && 1 == 0) {
        for (int t2=1; t2<=N-1; t2++) {
            numBlocks = (N - t2 + threadsPerBlock - 1) / threadsPerBlock;
            computeS_kernel<<<numBlocks, threadsPerBlock>>>(N, t2, d_S, d_RNA);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_S, d_S, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU version
    if(1==1)
    for (int t2=1; t2<=N-1; t2++) {
        for (int t4=t2; t4<=N-1; t4++) {
            for (int t6=0; t6<=t2-1; t6++) {
                cpu_S[(-t2+t4)*N + t4] = MAX(cpu_S[(-t2+t4)*N + t6+(-t2+t4)] + cpu_S[(t6+(-t2+t4)+1)*N + t4], cpu_S[(-t2+t4)*N + t4]);
            }
            cpu_S[(-t2+t4)*N + t4] = MAX(cpu_S[(-t2+t4)*N + t4], cpu_S[(-t2+t4+1)*N + t4-1] + can_pair(h_RNA, (-t2+t4), t4));
        }
    }

    // Validate results
    for (int i = 0; i < N * N; i++) {
        assert(h_S[i] == cpu_S[i]);
    }

    printf("Validation successful!\n");

    cudaFree(d_S);
    cudaFree(d_RNA);

    free(h_S);
    free(cpu_S);
    free(h_RNA);

    return 0;
}
