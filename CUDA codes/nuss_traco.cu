#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // for random numbers
#include <omp.h>

// Assuming some upper limit for the random values
#define MAX_RANDOM_VALUE 10
#define BLOCK_SIZE 16


// Implementation of sigma as a device function (placeholder; replace as needed)
__device__ int sigma(int a, int b) {
    return a + b;  // Placeholder logic.
}

__device__ int isPair(char a, char b) {
    if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
        return 1;
    }
    return 0;
}

int h_isPair(char a, char b) {
    if ((a == 'A' && b == 'U') || (a == 'U' && b == 'A') || (a == 'C' && b == 'G') || (a == 'G' && b == 'C')) {
        return 1;
    }
    return 0;
}



int host_sigma(int a, int b) {
    return a + b;  // Placeholder logic. Adjust as needed.
}





__global__ void dynamicProgrammingKernel(int** d_S, int n, int c1, int chunk_size, char* sequence) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int c3_base = globalThreadIdx + max(0, -n + c1 + 1);
    
    for (int offset = 0; offset < chunk_size && (c3_base + offset) < (c1 + 1) / 2; offset++) {
        int c3 = c3_base + offset * blockDim.x * gridDim.x;
        if(c3 < (c1 + 1) / 2) {
            for (int c5 = 0; c5 <= c3; c5++) {
                d_S[n-c1+c3-1][n-c1+2*c3] = max(d_S[n-c1+c3-1][n-c1+c3+c5-1] + d_S[n-c1+c3+c5-1+1][n-c1+2*c3], d_S[n-c1+c3-1][n-c1+2*c3]);
            }
            d_S[n-c1+c3-1][n-c1+2*c3] = max(d_S[n-c1+c3-1][n-c1+2*c3], d_S[n-c1+c3-1+1][n-c1+2*c3-1] + isPair(sequence[n-c1+c3-1], sequence[n-c1+2*c3]));
        }
    }
}









int main() {
    int n = 5000;  // Example size; adjust as needed.

    char h_sequence[n] = "AUCGAUCG";
    char *d_sequence;


    int* flatArray_S = new int[n * n];
    int* flatArray_S_CPU = new int[n * n];

    // Allocate 2D host array for CPU and GPU
    int** S = new int*[n];
    int** S_CPU = new int*[n];

    for(int i = 0; i < n; i++) {
        S[i] = &flatArray_S[i * n];
        S_CPU[i] = &flatArray_S_CPU[i * n];
    }

    for(int i = 0; i < n; i++) {
    
        // Initialize the array elements with random values
        for(int j = 0; j < n; j++) {
            S[i][j] = rand() % MAX_RANDOM_VALUE;
            S_CPU[i][j] = S[i][j];  // Copy the values
        }
    }

    // Run the CPU code
double cpu_time = omp_get_wtime();
if(1==0)
    for (int c1 = 1; c1 < 2 * n - 2; c1++) {
        #pragma omp parallel shared(c1, n) 
        for(int c3 = max(0, -n + c1 + 1); c3 < (c1 + 1) / 2; c3++) {
            for(int c5 = 0; c5 <= c3; c5++) {
                S_CPU[n-c1+c3-1][n-c1+2*c3] = max(S_CPU[n-c1+c3-1][n-c1+c3+c5-1] + S_CPU[n-c1+c3+c5-1+1][n-c1+2*c3], S_CPU[n-c1+c3-1][n-c1+2*c3]);
            }
            S_CPU[n-c1+c3-1][n-c1+2*c3] = max(S_CPU[n-c1+c3-1][n-c1+2*c3], S_CPU[n-c1+c3-1+1][n-c1+2*c3-1] + h_isPair(h_sequence[n-c1+c3-1], h_sequence[n-c1+2*c3]));  // Assuming sigma function is similarly defined on the host.
        }
    }

printf("cpu ended\n");
printf("CPU Time taken: %f seconds\n", omp_get_wtime()-cpu_time);
    // GPU execution code 
    int* flat_d_S;
    int** d_S;

double start_time = omp_get_wtime();

    cudaMalloc(&d_sequence, n);
    cudaMalloc(&flat_d_S, n * n * sizeof(int));
    cudaMalloc(&d_S, n * sizeof(int*));
    
    int* h_S[n];
    for(int i = 0; i < n; i++) {
        h_S[i] = flat_d_S + i * n;
    }
    cudaMemcpy(d_S, h_S, n * sizeof(int*), cudaMemcpyHostToDevice);
cudaMemcpy(d_sequence, h_sequence, n, cudaMemcpyHostToDevice);


    // Copy host data to device before entering the loop
    cudaMemcpy(flat_d_S, &S[0][0], n * n * sizeof(int), cudaMemcpyHostToDevice);
int threadsNeeded = (n + 1) / 2;
    int numBlocks = (threadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int chunk_size = (threadsNeeded + numBlocks * BLOCK_SIZE - 1) / (numBlocks * BLOCK_SIZE);



    for (int c1 = 1; c1 < 2 * n - 2; c1++) {
dynamicProgrammingKernel<<<numBlocks, BLOCK_SIZE>>>(d_S, n, c1, chunk_size, d_sequence);

//        dynamicProgrammingKernel<<<1, (n + 1) / 2>>>(d_S, n, c1);
        cudaDeviceSynchronize();

    }

    // Copy results back to the host after all kernel invocations
    cudaMemcpy(&S[0][0], flat_d_S, n * n * sizeof(int), cudaMemcpyDeviceToHost);

double end_time = omp_get_wtime();
double elapsed_time = end_time - start_time;
printf("Time taken: %f seconds\n", elapsed_time);

printf("gpu ended\n");


    // Compare the results
    bool valid = true;
    for(int i = 0; i < n && valid; i++) {
        for(int j = 0; j < n; j++) {
            if(S[i][j] != S_CPU[i][j]) {
                valid = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): GPU = " << S[i][j] << ", CPU = " << S_CPU[i][j] << std::endl;
                break;
            }
        }
    }

    if(valid) {
        std::cout << "Results are identical!" << std::endl;
    } else {
        std::cout << "Results differ!" << std::endl;
    }


    // Cleanup memory
//    for(int i = 0; i < n; i++) {
//        delete[] S[i];
//        delete[] S_CPU[i];
//    }
    delete[] S;
    delete[] S_CPU;

    cudaFree(d_S);
    cudaFree(flat_d_S);

    return 0;
}
