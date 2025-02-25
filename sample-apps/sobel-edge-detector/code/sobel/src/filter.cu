#include "filter.h"
#include "timer.h"

#include <iostream>

using namespace std;

int sobel_kernel_x[3][3] = {
		{ 1, 0,-1},
		{ 2, 0,-2},
		{ 1, 0,-1}};

int sobel_kernel_x_flip[3][3] = {
		{ 1, 0,-1},
		{ 0, 0,-2},
		{ 1, 0,-1}};

int sobel_kernel_y[3][3] = {
		{ 1, 2, 1},
		{ 0, 0, 0},
		{-1,-2,-1}};

int sobel_kernel_y_flip[3][3] = {
		{-1,-2,-1},
		{ 0, 0, 0},
		{ 1, 2, 1}};
// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b) 
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	int32_t sum_x[4]={0}, sum_y[4] = {0};
	int rows = height;
	int cols = width;

	int64_t temp[3] = {0};
	int index = 0;

	// loop through each row of input image
	for (int x = 0; x < rows; x++)
	{
		//printf("SRC Index, %d\n", x);
		// loop through each column of input image
		for (int y = 0; y < cols; y+=4)
		{
			index = x * width + y;			
		    //printf("SRC Index, %i\n", y);
            sum_x[0]+=   sobel_kernel_x[0][0] * input[index]
                       + sobel_kernel_x[0][1] * input[index+1]
                       + sobel_kernel_x[0][2] * input[index+2]
                       + sobel_kernel_x[1][0] * input[index+rows]
                       + sobel_kernel_x[1][1] * input[index+rows + 1]
                       + sobel_kernel_x[1][2] * input[index+rows + 2]
                       + sobel_kernel_x[2][0] * input[index+2*rows]
                       + sobel_kernel_x[2][1] * input[index+2*rows+1]
                       + sobel_kernel_x[2][2] * input[index+2*rows+2];

            sum_x[1]+= sobel_kernel_x[0][0] * input[index+1]
                       + sobel_kernel_x[0][1] * input[index+2]
                       + sobel_kernel_x[0][2] * input[index+3]
                       + sobel_kernel_x[1][0] * input[index+1*rows+1]
                       + sobel_kernel_x[1][1] * input[index+1*rows+2]
                       + sobel_kernel_x[1][2] * input[index+1*rows+3]
                       + sobel_kernel_x[2][0] * input[index+2*rows+1]
                       + sobel_kernel_x[2][1] * input[index+2*rows+2]
                       + sobel_kernel_x[2][2] * input[index+2*rows+3];

            sum_x[2]+= sobel_kernel_x[0][0] * input[index+2]
                       + sobel_kernel_x[0][1] * input[index+3]
                       + sobel_kernel_x[0][2] * input[index+4]
                       + sobel_kernel_x[1][0] * input[index+1*rows+2]
                       + sobel_kernel_x[1][1] * input[index+1*rows+3]
                       + sobel_kernel_x[1][2] * input[index+1*rows+4]
                       + sobel_kernel_x[2][0] * input[index+2*rows+2]
                       + sobel_kernel_x[2][1] * input[index+2*rows+3]
                       + sobel_kernel_x[2][2] * input[index+2*rows+4];

			if (x <= rows && y < cols) 
			{
				sum_x[3]+= sobel_kernel_x[0][0] * input[index+3]
						   + sobel_kernel_x[0][1] * input[index+4]
						   + sobel_kernel_x[0][2] * input[index+5]
						   + sobel_kernel_x[1][0] * input[index+1*rows+3]
						   + sobel_kernel_x[1][1] * input[index+1*rows+4]
						   + sobel_kernel_x[1][2] * input[index+1*rows+5]
						   + sobel_kernel_x[2][0] * input[index+2*rows+3]
						   + sobel_kernel_x[2][1] * input[index+2*rows+4]
						   + sobel_kernel_x[2][2] * input[index+2*rows+5];
			}

            sum_y[0]+= sobel_kernel_y_flip[0][0] * input[index]
                       + sobel_kernel_y_flip[0][1] * input[index+1]
                       + sobel_kernel_y_flip[0][2] * input[index+2]
                       + sobel_kernel_y_flip[1][0] * input[index+1*rows]
                       + sobel_kernel_y_flip[1][1] * input[index+1*rows+1]
                       + sobel_kernel_y_flip[1][2] * input[index+1*rows+2]
                       + sobel_kernel_y_flip[2][0] * input[index+2*rows]
                       + sobel_kernel_y_flip[2][1] * input[index+2*rows+1]
                       + sobel_kernel_y_flip[2][2] * input[index+2*rows+2];

            sum_y[1]+= sobel_kernel_y_flip[0][0] * input[index+1]
                       + sobel_kernel_y_flip[0][1] * input[index+2]
                       + sobel_kernel_y_flip[0][2] * input[index+3]
                       + sobel_kernel_y_flip[1][0] * input[index+1*rows+1]
                       + sobel_kernel_y_flip[1][1] * input[index+1*rows+2]
                       + sobel_kernel_y_flip[1][2] * input[index+1*rows+3]
                       + sobel_kernel_y_flip[2][0] * input[index+2*rows+1]
                       + sobel_kernel_y_flip[2][1] * input[index+2*rows+2]
                       + sobel_kernel_y_flip[2][2] * input[index+2*rows+3];

            sum_y[2]+= sobel_kernel_y_flip[0][0] * input[index+2]
                       + sobel_kernel_y_flip[0][1] * input[index+3]
                       + sobel_kernel_y_flip[0][2] * input[index+4]
                       + sobel_kernel_y_flip[1][0] * input[index+1*rows+2]
                       + sobel_kernel_y_flip[1][1] * input[index+1*rows+3]
                       + sobel_kernel_y_flip[1][2] * input[index+1*rows+4]
                       + sobel_kernel_y_flip[2][0] * input[index+2*rows+2]
                       + sobel_kernel_y_flip[2][1] * input[index+2*rows+3]
                       + sobel_kernel_y_flip[2][2] * input[index+2*rows+4];

			if (x <= rows && y < cols) 
			{
				sum_y[3]+= sobel_kernel_y_flip[0][0] * input[index+3]
						   + sobel_kernel_y_flip[0][1] * input[index+4]
						   + sobel_kernel_y_flip[0][2] * input[index+5]
						   + sobel_kernel_y_flip[1][0] * input[index+1*rows+3]
						   + sobel_kernel_y_flip[1][1] * input[index+1*rows+4]
						   + sobel_kernel_y_flip[1][2] * input[index+1*rows+5]
						   + sobel_kernel_y_flip[2][0] * input[index+2*rows+3]
						   + sobel_kernel_y_flip[2][1] * input[index+2*rows+4]
						   + sobel_kernel_y_flip[2][2] * input[index+2*rows+5];
			}
			//printf("Img X, Img Y = %i, %i\ n", x,y);
			temp[0] = sqrt(pow(sum_x[0],2)+pow(sum_y[0],2));
			temp[1] = sqrt(pow(sum_x[1],2)+pow(sum_y[1],2));
			temp[2] = sqrt(pow(sum_x[2],2)+pow(sum_y[2],2));
			if (x <= rows && y < cols) 
			{
			temp[3] = sqrt(pow(sum_x[3],2)+pow(sum_y[3],2));
			}
			
			if (temp[0]<0)
			{
				output[index]   = (temp[0] < 0) ? 0 : (temp[0]);
			}
			else
			{
				output[index]   = (temp[0] > 255) ? 255 : (temp[0]);
			}


			if (temp[1]<0)
			{
				output[index+1] = (temp[1] < 0) ? 0 : (temp[1]);
			}
			else
			{
				output[index+1] = (temp[1] > 255) ? 255 : (temp[1]);
			}


			if (temp[2]<0)
			{
				output[index+2]  = (temp[2] < 0) ? 0 : (temp[2]);
			}
			else
			{
				output[index+2]  = (temp[2] > 255) ? 255 : (temp[2]);
			}

			if (x <= rows && y < cols) 
			{
				if (temp[3]<0)
				{
					output[index+3]  = (temp[3] < 0) ? 0 : (temp[3]);
				}
				else
				{
					output[index+3]  = (temp[3] > 255) ? 255 : (temp[3]);
				}
			}
			sum_x[0] = 0;
			sum_x[1] = 0;
			sum_x[2] = 0;
			sum_x[3] = 0;
			sum_y[0] = 0;
			sum_y[1] = 0;
			sum_y[2] = 0;
			sum_y[3] = 0;
			temp[0]  = 0;
			temp[1]  = 0;
			temp[2]  = 0;
			temp[3]  = 0;
		}
	}

}

// =================== GPU Kernel Functions ===================
__global__ void sobel_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	//threadIdx = Thread Index in Block
	//blockIdx = Block index in Grid
	//blockDim = Size of block ( # threads in block )
	// gridDim = dimension of Grid 
	// Memory Bus Width : 128-bit 
	
	// 1-D Indexing
	const int x = blockIdx.x * blockDim.x + threadIdx.x;         
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	
	// Accumulator 
	int32_t sum_x=0, sum_y=0;
	
	// Size of output image
	
	float magnitude= 0;
	
	// Sobel Kernel X 
	int sobel_kernel_x[3][3] = {
		{ 1, 0,-1},
		{ 2, 0,-2},
		{ 1, 0,-1}};

	// Sobel Kernel Y Flipped for Convolution
	int sobel_kernel_y_flip[3][3] = {
		{-1,-2,-1},
		{ 0, 0, 0},
		{ 1, 2, 1}};

	if (x >= 0 && x < width && y >= 0 && y < height) 
	{
		sum_x+=      sobel_kernel_x[0][0] * input[y*width + x]
				   + sobel_kernel_x[0][1] * input[y*width + (x+1)]
				   + sobel_kernel_x[0][2] * input[y*width + (x+2)]
				   + sobel_kernel_x[1][0] * input[(y+1)*width + x]
				   + sobel_kernel_x[1][1] * input[(y+1)*width + (x+1)]
				   + sobel_kernel_x[1][2] * input[(y+1)*width + (x+2)]
				   + sobel_kernel_x[2][0] * input[(y+2)*width + x]
				   + sobel_kernel_x[2][1] * input[(y+2)*width + (x+1)]
				   + sobel_kernel_x[2][2] * input[(y+2)*width + (x+2)];
												 
		sum_y+=      sobel_kernel_y_flip[0][0] * input[y*width + x]
				   + sobel_kernel_y_flip[0][1] * input[y*width + (x+1)]
				   + sobel_kernel_y_flip[0][2] * input[y*width + (x+2)]
				   + sobel_kernel_y_flip[1][0] * input[(y+1)*width + x]
				   + sobel_kernel_y_flip[1][1] * input[(y+1)*width + (x+1)]
				   + sobel_kernel_y_flip[1][2] * input[(y+1)*width + (x+2)]
				   + sobel_kernel_y_flip[2][0] * input[(y+2)*width + x]
				   + sobel_kernel_y_flip[2][1] * input[(y+2)*width + (x+1)]
				   + sobel_kernel_y_flip[2][2] * input[(y+2)*width + (x+2)];

		magnitude = sqrt(pow(sum_x,2) + pow(sum_y,2));  
		if (magnitude<0)
		{
			output[y*width + x]  = (magnitude < 0) ? 0 : (magnitude);
		}
		else
		{
			output[y*width + x]  = (magnitude > 255) ? 255 : (magnitude);
		}
   
	
	}
	//}
}

// =================== GPU Host Functions ===================
void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	//TODO: launch kernel function // 512/8 
	const int grid_x = 8; // 32 blocks
	const int grid_y = 8; // 32 blocks

	dim3 block(grid_x, grid_y, 1); // 32,32 blocks  
	dim3 grid(divup(width, grid_x), divup(height,grid_y),1); // 512/32 = 16x16=256 threads per grid block 

	sobel_gpu<<<grid,block>>>(input,output,height,width);

	cudaDeviceSynchronize(); // everytime launch kernel 
}
