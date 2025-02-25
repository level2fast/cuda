This project is an application that uses the CUDA library to implement an algorithm that performs edge detection using the sobel filter. An image is convolved with the sobel filter and stored on the JetsonTX2 board for analysis. CUDA is used to speed up the processing of the image and a set of tests were ran to determine the best block size to use when processing. 

A report was produced comparing the two different implementations of the algorithm. One using CUDA and the GPU. The other using the CPU and the OpenCV library. 
