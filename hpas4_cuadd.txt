#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;

//Function for addition of vector
__global__ void vectorAdd(const float* a, const float* b, float* c, int size)
{
    //Calculate the data index by multiplying the block index with block dimension and adding thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    //Initialize size/digits of number
    int size = 1000000; 
    //Calculate size required in bytes
    size_t bytes = size * sizeof(float);

    //Create arrays on host to store numbers
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];

    //Generate numbers
    for (int i = 0; i < size; ++i) {
        h_a[i] = rand()%1000;
        h_b[i] = rand()%1000;
    }

    //Allocate memory in CUDA for required bytes
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    //Copy data from Host(CPU) to Device(GPU)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    //Define the number of threads per block
    int threadsPerBlock = 256;
    //Calculate blocks per grid
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    //Call vectorAdd global function
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    //Copy results from device to host again
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //Display first 10 elements of data in host
    for (int i = 0; i < 10; ++i) {
        cout<<h_c[i] << " ";
    }
    std::cout << std::endl;

    //Free stack and pointer memory, and free cuda memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
