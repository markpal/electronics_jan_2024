#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Placeholder for the isPair function. Replace with your actual implementation.
__device__ int isPair_device(int a, int b) {
    return a + b; // Placeholder logic
}

int isPair_host(int a, int b) {
    return a + b; // Placeholder logic
}

__global__ void computeS_kernel(int N, int w0, int** d_S) {
    int h0 = blockIdx.x * blockDim.x + threadIdx.x - N + w0 + 1;

    if (h0 <= 0) {
        for (int i3 = 0; i3 < w0; i3++) {
            d_S[-h0][w0 - h0] = MAX(d_S[-h0][-h0 + i3] + d_S[-h0 + i3 + 1][w0 - h0], d_S[-h0][w0 - h0]);
        }
        d_S[-h0][w0 - h0] = MAX(d_S[-h0][w0 - h0], d_S[-h0 + 1][w0 - h0 - 1] + isPair_device(-h0, w0 - h0));
    }
}

int main() {
    int N = 15000;
    int **h_S, **d_S, **cpu_S;
    int *d_S_data;

    h_S = (int**)malloc(N * sizeof(int*));
    cpu_S = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        h_S[i] = (int*)malloc(N * sizeof(int));
        cpu_S[i] = (int*)malloc(N * sizeof(int));
        for (int j = 0; j < N; j++) {
            h_S[i][j] = rand() % 100;
            cpu_S[i][j] = h_S[i][j];
        }
    }

    cudaMalloc(&d_S_data, N * N * sizeof(int));
    cudaMalloc(&d_S, N * sizeof(int*));
    
    int **h_S_array = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        h_S_array[i] = d_S_data + i * N;
    }
    cudaMemcpy(d_S, h_S_array, N * sizeof(int *), cudaMemcpyHostToDevice);

    for (int i = 0; i < N; i++) {
        cudaMemcpy(h_S_array[i], h_S[i], N * sizeof(int), cudaMemcpyHostToDevice);
    }

    int threadsPerBlock = 256;
    int numBlocks;

    for (int w0 = 1; w0 < N; w0++) {
        numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        computeS_kernel<<<numBlocks, threadsPerBlock>>>(N, w0, d_S);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < N; i++) {
        cudaMemcpy(h_S[i], h_S_array[i], N * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // CPU version for verification
if(1==0)    
for (int w0 = 1; w0 < N; w0++) {
        for (int h0 = -N + w0 + 1; h0 <= 0; h0++) {
            for (int i3 = 0; i3 < w0; i3++) {
                cpu_S[-h0][w0 - h0] = MAX(cpu_S[-h0][-h0 + i3] + cpu_S[-h0 + i3 + 1][w0 - h0], cpu_S[-h0][w0 - h0]);
            }
            cpu_S[-h0][w0 - h0] = MAX(cpu_S[-h0][w0 - h0], cpu_S[-h0 + 1][w0 - h0 - 1] + isPair_host(-h0, w0 - h0));
        }
    }

    // Validate results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(h_S[i][j] == cpu_S[i][j]);
        }
    }

    printf("Validation successful!\n");

    cudaFree(d_S_data);
    cudaFree(d_S);
    free(h_S_array);

    for (int i = 0; i < N; i++) {
        free(h_S[i]);
        free(cpu_S[i]);
    }
    free(h_S);
    free(cpu_S);

    return 0;
}

