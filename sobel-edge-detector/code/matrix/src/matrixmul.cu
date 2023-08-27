#include <stdio.h>
#include <stdlib.h>

#include "matrixmul.h"
#include "timer.h"
#define BLOCK_SIZE 16
// #define BLOCK_SIZE 32

__global__ void block_mm_kernel(const float* A, const float* B, float* C, int N, int M) 
{
	// N * M    M * N
	// ROW COL  ROW COL 
 	const int WARP_SIZE = BLOCK_SIZE;
	
    __shared__ float sub_A[WARP_SIZE][WARP_SIZE];
    __shared__ float sub_B[WARP_SIZE][WARP_SIZE];

	int col = blockIdx.x * blockDim.x + threadIdx.x; // block matrix
	int row = blockIdx.y * blockDim.y + threadIdx.y;  //block matrix 
	float temp = 0.0f;  
	int index_a;
	int index_b;

	for(int i = 0; i<M/WARP_SIZE; i++)
	{	
		index_a = row * M + i * WARP_SIZE + threadIdx.x;
		//Get element in matrix for each thread within each warp of threads
		if (index_a < N*M)
		{
			sub_A[threadIdx.y][threadIdx.x] = A[index_a];
		}
		else
		{
			sub_A[threadIdx.y][threadIdx.x]  = 0;
		}
		
		index_b = (i * WARP_SIZE + threadIdx.y) * N + col;
		if (index_b < N*M)
		{
			sub_B[threadIdx.y][threadIdx.x] = B[index_b];
		}
		else
		{
			sub_B[threadIdx.y][threadIdx.x]  = 0;
		}
		__syncthreads(); 
		
		for (int j=0; j<WARP_SIZE; j++)
		{
			temp += sub_A[threadIdx.y][j] * sub_B[j][threadIdx.x];
		}
		__syncthreads();
	}
		
	if (row < N && col < N)
	{
		C[row*N + col]= temp;
	}
}

inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

float run_mm_gpu(const float* A, const float* B, float* C, int N, int M)
{
	Timer gpu_timer;
	gpu_timer.start();

	//TODO: launch the kernel function
	const int grid_x = BLOCK_SIZE; // 32 blocks
	const int grid_y = BLOCK_SIZE; // 32 blocks
	dim3 grid(divup(N, grid_x), divup(N,grid_y),1); 
	dim3 block(grid_x, grid_y, 1); 
	block_mm_kernel<<<grid,block>>>(A, B, C, N, M);
	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}


