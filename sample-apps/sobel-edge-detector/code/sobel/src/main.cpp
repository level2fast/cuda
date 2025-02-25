#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "filter.h"
#include "timer.h"

//#define UNIFIED_MEM 

using namespace std;
using namespace cv;

enum SobelType
{
	SOBEL_OPENCV,
	SOBEL_CPU,
	SOBEL_GPU
};


int main(int argc, const char * argv[])
{
    //Uncomment the following line to use the external camera.
	//VideoCapture cap(1); 

    //Comment this line if you're using the external camera.
    VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;

	SobelType sobel_type = SOBEL_GPU;


	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 2)
	{
		WIDTH = atoi(argv[1]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 3)
	{
		HEIGHT = atoi(argv[2]);
	}

	// 3 arguments on command line: WIDTH = arg1, HEIGHT = arg2, type = arg3
	if(argc >= 4)
	{
		sobel_type = static_cast<SobelType>(atoi(argv[3]));
	}


	switch(sobel_type)
	{
		case SOBEL_OPENCV:
			cout << "Using OpenCV" << endl;
			break;
		case SOBEL_CPU:
			cout << "Using CPU" << endl;
			break;
		case SOBEL_GPU:
			cout << "Using GPU" << endl;
			break;
	}


	// Profiling
	Timer timer;
	double time_elapsed = 0;

	//TODO: allocate unified memory for the input and result matrices.
 	unsigned char* gray_device;
	unsigned char* sobel_out_device;
	
#ifndef UNIFIED_MEM
    //TODO: Allocate memory on the GPU device.
    //TODO: Declare the host image result matrices
    cudaMalloc((void **)&gray_device,WIDTH*HEIGHT*sizeof(unsigned char));
    cudaMalloc((void **)&sobel_out_device,WIDTH*HEIGHT*sizeof(unsigned char));

	Mat gray      = Mat(HEIGHT, WIDTH, CV_8U);
	Mat sobel_out = Mat(HEIGHT, WIDTH, CV_8U);

#else
    //TODO: Allocate unified memory for the necessary matrices
    //TODO: Declare the image matrices which point to the unified memory
	cudaMallocManaged((void **)&gray_device,WIDTH*HEIGHT*sizeof(unsigned char));
    cudaMallocManaged((void **)&sobel_out_device,WIDTH*HEIGHT*sizeof(unsigned char));

	Mat gray      = Mat(HEIGHT, WIDTH, CV_8U);
	Mat sobel_out = Mat(HEIGHT, WIDTH, CV_8U);
	
#endif
	 

	// More declarations
	Mat frame, s_x, s_y;

	char key = 0;
	int count = 0;


	// Main loop
	while(key != 'q')
	{
		// Get frame
		cap >> frame;

		// If no more frames, wait and exit
		if(frame.empty())
		{
			waitKey();
			break;
		}

		// Resize and grayscale
		resize(frame, frame, Size(WIDTH, HEIGHT));
#ifdef OPENCV4
                cvtColor(frame, gray, COLOR_BGR2GRAY);
#else
                cvtColor(frame, gray, CV_BGR2GRAY);
#endif

		// OpenCV Sobel
		switch(sobel_type)
		{
			case SOBEL_OPENCV:
				timer.start();
				Sobel(gray, s_x, CV_8U, 1, 0, 3, 1, 0, BORDER_ISOLATED);
				Sobel(gray, s_y, CV_8U, 0, 1, 3, 1, 0, BORDER_ISOLATED);
				addWeighted(s_x, 0.5, s_y, 0.5, 0, sobel_out);
				timer.stop();
				break;
			case SOBEL_CPU:
				timer.start();
				//TODO: call the sobel CPU function
				sobel_filter_cpu(gray.ptr<uchar>(), sobel_out.ptr<uchar>(), HEIGHT, WIDTH); 
				timer.stop();
				break;

			case SOBEL_GPU:
				timer.start();
				//TODO: call the sobel GPU function
				
#ifndef UNIFIED_MEM
                /* TODO: 1) Copy data from host to device
                 *       2) Call GPU host function with device data
                 *       3) Copy data from device to host
                */
		//cudaMemcpy(<TO ADDRESS>,<FROM ADDR>,<Size>,cudaMemcpyHostToDevice);
		cudaMemcpy(gray_device,gray.ptr<uchar>(), WIDTH*HEIGHT*sizeof(unsigned char), cudaMemcpyHostToDevice);
		sobel_filter_gpu(gray_device, sobel_out_device, HEIGHT, WIDTH);
		cudaMemcpy(sobel_out.ptr<uchar>(),sobel_out_device,WIDTH*HEIGHT*sizeof(unsigned char), cudaMemcpyDeviceToHost);
#else
                /* TODO: 1) Call GPU host function with unified memory allocated data
                */
		sobel_filter_gpu(gray.ptr<uchar>(), sobel_out.ptr<uchar>(), HEIGHT, WIDTH); 

	
#endif
				
				timer.stop();
				break;
		}

		time_elapsed += timer.getElapsed();

		count++;

		if(count % 10 == 0)
		{
			time_elapsed = time_elapsed/10.;
			cout << "Execution time (ms) = " << time_elapsed << endl;
			time_elapsed = 0;
		}


		// Display results
		if(gray.cols <= 1024 || gray.rows <= 1024)
		{
			imshow("Input", gray);
			imshow("Sobel", sobel_out);
			if(count <= 1) { moveWindow("Sobel", WIDTH, 0); }
		}

		key = waitKey(1);
	}

	//TODO: free the device memory
#ifndef UNIFIED_MEM
	cudaFree(gray_device);
	cudaFree(sobel_out_device);
#endif 
	return 0;
}
