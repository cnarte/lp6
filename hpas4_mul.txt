//Include header files
#include <iostream>
#include <cstdlib>
#include <bits/stdc++.h>

using namespace std;

//Function for matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int N)
{
    //Calculate the data index by multiplying the block index with block dimension and adding thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Perform row column multiplication by multiplying and summing
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    //4X4 matrix used
    int N = 4; 

    int *a, *b, *c; 
    int *d_a, *d_b, *d_c; 
    //Calculate size required in bytes
    int matrixSize = N * N * sizeof(int);

    //Create arrays on host to store numbers
    a = (int*)malloc(matrixSize);
    b = (int*)malloc(matrixSize);
    c = (int*)malloc(matrixSize);

    //Generate numbers
    for (int i = 0; i < N * N; ++i) {
        a[i] = rand()%1000;
        b[i] = rand()%1000;
    }

    //Allocate memory in CUDA for required bytes
    cudaMalloc((void**)&d_a, matrixSize);
    cudaMalloc((void**)&d_b, matrixSize);
    cudaMalloc((void**)&d_c, matrixSize);

    //Copy data from Host(CPU) to Device(GPU)
    cudaMemcpy(d_a, a, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, matrixSize, cudaMemcpyHostToDevice);

    //Define the number of threads per block
    dim3 threadsPerBlock(2, 2);
    //Calculate blocks per grid
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //Call matrixMultiplication global function
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, matrixSize, cudaMemcpyDeviceToHost);

    //Display the first elements of results
    for (int i = 0; i < N * N; ++i) {
        std::cout << c[i] << " ";
        if ((i + 1) % N == 0)
            cout<<endl;
    }

    //Free pointers and cuda space
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;}

