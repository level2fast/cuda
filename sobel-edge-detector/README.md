# **Sobel Edge Detector**  

![GitHub License](https://img.shields.io/github/license/level2fast/cuda)<br/>
![GitHub contributors](https://img.shields.io/github/contributors/level2fast/cuda) <br/>
![GitHub top language](https://img.shields.io/github/languages/top/level2fast/cuda)<br/>
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/level2fast/cuda) <br/>
![GitHub repo size](https://img.shields.io/github/repo-size/level2fast/cuda)<br/>

## **📝 Project Description**  
This project is an application that uses the CUDA library to implement an algorithm that performs edge detection using the sobel filter. An image is convolved with the sobel filter and stored on the JetsonTX2 board for analysis. CUDA is used to speed up the processing of the image and a set of tests were ran to determine the best block size to use when processing. 
 
## **Sobel Operator** 
The operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical.The Sobel operator uses two 3×3 kernels to approximate the gradient of the image intensity.  

The horizontal (`Fx`) and vertical (`Fy`) Sobel filters are defined as:

### **Sobel Operator - Horizontal (`Fx`)**
| -1  |  0  | +1  |
|----|----|----|
| -2  |  0  | +2  |   
| -1  |  0  | +1  |

### **Sobel Operator - Vertical (`Fy`)**
| -1  | -2  | -1  |
|----|----|----|
|  0  |  0  |  0  |   
| +1  | +2  | +1  |

If we define **A** as the source image, and **Gx** and **Gy** as two images which at each point contain the horizontal and vertical derivative approximations respectively, the computations are as follows:

**Gx** = **Fx** * **A** <br/>
**Gy** = **Fy** * **A**

where `*` represents the **2D convolution operation** applied to the image.

## **Profile Report**
A [report](https://github.com/level2fast/cuda/blob/main/sobel-edge-detector/docs/Report.pdf) was produced comparing the speed of the two different implementations of the algorithm. One using CUDA and the GPU. The other using the CPU and the OpenCV library. 

---

 ## **🚀 Live Demo**  
[🔗 Click here to check out the live version](https://youtu.be/UHnwwBcvIdk)

---

## **📸 Screenshots**  
Snapshot of 1 frame of a test video used in this project with sobel operator applied to it.  

![Screenshot](sobel-edge-detector/code/sobel/images/sobel_snapshot.png)  

---

## **🛠️ Features**  
✅ Feature 1 – *Profiling of Sobel Algorithm execution*  
✅ Feature 2 – *Command Line Args to select Sobel Operator Type(OpenCV, CPU, or GPU)*  
✅ Feature 3 – *Image View: Input image vs. Sobel Output*  

---

## **📦 Tech Stack**  
- **Language:** C++, CUDA
- **Libraries:** OpenCV, Stdlib
- **Hardware:** Jetson Nano
---

## **📥 Installation & Setup**  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/level2fast/cuda.git
cd sobel-edge-detector/code/sobel
sudo apt install -y nvidia-jetpack
make 500 500 2 # WIDTH = arg1, HEIGHT = arg2, proc_type(CPU(0)|OPENCV(1)|GPU(2)) = arg3
``` 
