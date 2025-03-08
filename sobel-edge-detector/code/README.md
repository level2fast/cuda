# **Sobel Edge Detector**  

![GitHub License](https://img.shields.io/github/license/level2fast/cuda)<br/>
![GitHub contributors](https://img.shields.io/github/contributors/level2fast/cuda) <br/>
![GitHub top language](https://img.shields.io/github/languages/top/level2fast/cuda)<br/>
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/level2fast/cuda) <br/>
![GitHub repo size](https://img.shields.io/github/repo-size/level2fast/cuda)<br/>

## **📝 Project Description**  
This project is an application that uses the CUDA library to implement an algorithm that performs edge detection using the sobel filter. An image is convolved with the sobel filter and stored on the JetsonTX2 board for analysis. CUDA is used to speed up the processing of the image and a set of tests were ran to determine the best block size to use when processing. 
 
A [report](https://github.com/level2fast/cuda/blob/main/sobel-edge-detector/docs/Report.pdf) was produced comparing the two different implementations of the algorithm. One using CUDA and the GPU. The other using the CPU and the OpenCV library. 

---

<!-- ## **🚀 Live Demo**  
[🔗 Click here to check out the live version](https://your-live-demo-url.com) *(if applicable)*  

---

## **📸 Screenshots**  
Include relevant screenshots or GIFs showcasing your project’s interface and functionality.  

![Screenshot](https://your-screenshot-url.com/image.png)  

--- -->

## **🛠️ Features**  
✅ Feature 1 – *Profiling of Sobel Algorithm execution*  
✅ Feature 2 – *Command Line Args to select Sobel Operator Type(OpenCV, CPU, or GPU)*  
✅ Feature 3 – *Image View: Input image vs. Sobel Output*  

---

## **📦 Tech Stack**  
- **Language:** C++, CUDA
- **Libraries:** OpenCV, Stdlib
- **Hardware:** CUDA Enabled Device
---

<!-- ## **📥 Installation & Setup**  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
npm install
``` -->
