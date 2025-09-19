# CUDA-Accelerated Neural Network for MNIST

This project is a high-performance implementation of a Multi-Layer Perceptron (MLP) neural network, trained on the MNIST dataset and accelerated using CUDA C++. The primary goal was to explore and apply advanced CUDA optimization techniques to significantly reduce the training time.

---

## Project Overview

This project was a fantastic learning experience for me. I started with a foundational knowledge of neural networks and a desire to dive deep into GPU programming. My journey began by learning the fundamentals of CUDA—how to write kernels, manage memory between the host (CPU) and device (GPU), and integrate powerful libraries like **cuBLAS** for matrix multiplication.

My initial version was functional but sequential. The real challenge and breakthrough came when I began learning about **CUDA Streams**. The concept of creating asynchronous, overlapping timelines for GPU operations was a game-changer. I redesigned the training loop to process multiple data batches concurrently, effectively hiding the data transfer latency behind the GPU's computation time. This involved using **pinned memory** for faster host-to-device transfers and carefully synchronizing the streams to prevent race conditions—a challenge that really solidified my understanding of parallel programming.

Finally, to validate the performance gains, I implemented AI to create a detailed timing analysis. I refined this analysis to accurately measure the time spent on different parts of the program, confirming the incredible efficiency of the final, stream-optimized version.

---

## Performance Results

The final version of the code, running on an **NVIDIA RTX 2060**, achieved the following performance for a full training run (4 epochs on 10,000 images):

| Metric                  | Time Taken |
| :---------------------- | :--------- |
| **GPU Compute Time** | 1.778 s    |
| **Host Computation** | 0.385 s    |
| **Inferred Memory Transfers** | 0.252 s    |
| **Total Run Time** | **2.415 s**|

---

## Key Features & Concepts Implemented

* **Custom CUDA Kernels**: Wrote custom kernels for activation functions (ReLU), their derivatives, and other element-wise operations.
* **cuBLAS Integration**: Leveraged the highly optimized cuBLAS library for all matrix multiplication tasks.
* **CUDA Streams**: Implemented a double-buffered pipeline using two CUDA streams to overlap H2D data transfers, kernel executions, and weight updates concurrently.
* **Pinned Memory**: Utilized `cudaMallocHost` to allocate page-locked host memory, enabling asynchronous memory copies and maximizing data transfer bandwidth.

---

## Neural Network Architecture

The model is a simple Multi-Layer Perceptron (MLP) with one hidden layer, designed to classify the 28x28 handwritten digit images from the MNIST dataset.

* **Input Layer**: 784 neurons (representing each pixel in a flattened 28x28 image).
* **Hidden Layer**: 256 neurons with a ReLU activation function.
* **Output Layer**: 10 neurons (one for each digit, 0-9) with a Softmax activation function to produce a probability distribution.

The core of the forward pass involves two matrix multiplications:
1.  **Input to Hidden**: `[Batch Size, 784] @ [784, 256] = [Batch Size, 256]`
2.  **Hidden to Output**: `[Batch Size, 256] @ [256, 10] = [Batch Size, 10]`

---

## Dependencies & Setup

This project requires both a Python environment for data preparation and a C++/CUDA environment for the main program.

### Before You Begin: System Requirements

To compile and run the C++ portion, you will need the NVIDIA CUDA Toolkit, which includes the compiler (`nvcc`), system libraries (`cuBLAS`), and the GPU driver.
* **NVIDIA GPU Driver**: Ensure you have the latest drivers for your GPU.
* **NVIDIA CUDA Toolkit**: Download and install the toolkit from the official NVIDIA site: [NVIDIA CUDA Toolkit Download Page](https://developer.nvidia.com/cuda-downloads) (Select version 11.0 or newer).
* **(For Windows Users) WSL 2**: This project was developed on Windows 11 using the Windows Subsystem for Linux (WSL 2) with an Ubuntu distribution. If you are on Windows, it is highly recommended you use WSL. [Official Microsoft Guide to Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install).

### How to Run This Project

Follow these steps in your terminal.

1.  **Clone the Repository**
    First, download the project files from GitHub.
    ```bash
    git clone [https://github.com/hasan-a5/CUDA-Accelerated-MNIST-MLP.git](https://github.com/hasan-a5/CUDA-Accelerated-MNIST-MLP.git)
    cd CUDA-Accelerated-MNIST-MLP
    ```

2.  **Set Up Python and Download Data**
    This step uses Python to download the MNIST dataset and convert it into the binary format our program needs.
    ```bash
    # Create and activate a Python virtual environment
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Install the required Python packages
    pip install -r requirements.txt
    
    # Run the downloader script
    python3 downloader.py
    ```
    After this, you will have a `data/` folder in your project directory.

3.  **Compile the CUDA Program**
    Now, compile the main C++ file using NVIDIA's `nvcc` compiler.
    ```bash
    nvcc -o final_code final_code.cu -lcublas
    ```
    **Note**: The `-lcublas` flag is essential. It tells the compiler to link against the NVIDIA cuBLAS library, which contains the optimized matrix multiplication functions. This command will create a new executable file named `final_code`.

4.  **Run the Training**
    Execute the program you just compiled to start training the neural network.
    ```bash
    ./final_code
    ```
    You will see the training progress for each epoch, followed by the final performance report.
