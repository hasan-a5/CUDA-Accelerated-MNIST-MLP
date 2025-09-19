#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <cublas_v2.h>
#include <chrono>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define BATCH_SIZE 8
#define EPOCHS 4
#define TRAINING_SIZE 10000
#define TEST_SIZE 1000

typedef struct {
    float *weights1, *weights2, *bias1, *bias2;
    float *grad_weights1, *grad_weights2, *grad_bias1, *grad_bias2;
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
} NeuralNetwork;

// Load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr,"Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr,"Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// Load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr,"Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr,"Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// init function for weights using kaiming initialization
void initialize_weights(float *weights, int size) {
    float stddev = sqrt(2.0f / size);
    for (int i = 0; i < size; i++){
        weights[i] = ((float)rand() / RAND_MAX) * stddev - (stddev / 2.0f);
    }
}

// init for biases to zero
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++){
        bias[i] = 0.0f;
    }
}

// CUDA ReLU activation
__global__ void relu_kernal(float *x, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// CUDA bias addition
__global__ void add_bias_kernel(float *x, float *bias, int batch_size, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if(b < batch_size && i < size){
        x[idx] += bias[i];
    }
}

// CUDA softmax
__global__ void softmax_kernal(float *x, int batch_size, int size){
    int b = blockIdx.x;
    if(b < batch_size){
        float max_val = x[b*size];
        for(int i = 1; i < size; i++){ // loop to find true max
            max_val = fmaxf(max_val, x[b*size + i]);
        }

        float sum = 0.0f; // compute exp(x_i - max_val) for each element and sum them up
        for (int i = 0; i < size; i++) {
            x[b*size + i] = expf(x[b*size + i] - max_val);
            sum += x[b*size + i];
        }

        for (int i = 0; i < size; i++) { // divide each exponentiated value by the sum
            x[b*size + i] = fmaxf(x[b*size + i] / sum, 1e-8); // Clamp the min value to 1e-8 to avoid log(0)
        }
    }
}

// CUDA zeroing out gradients
__global__ void zero_grad_kernel(float *grad, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// CUDA ReLU derivative
__global__ void relu_derivative_kernel(float *x, float *d_ReLU, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_ReLU[idx] = x[idx] > 0.0f ? 1.0f : 0.0f;}
}

// CUDA element-wise gradient multiplication (used to find dX2 * d_ReLU)
__global__ void multiply_gradients_kernel(float *grad1, float *grad2, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad1[idx] *= grad2[idx];
    }
}

// C(m,n) = A(m,k) @ B(k,n) for row-major matrices
void matmul_a_b(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // By passing B then A, cuBLAS treats our row-major data as column-major B^T and A^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
}

// C(m,n) = A(m,k) @ B(n,k)^T for row-major matrices
void matmul_a_bt(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // C^T = (B^T)^T @ A^T = B @ A^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, B, k, A, k, &beta, C, n);
}

// C(m,n) = A(k,m)^T @ B(k,n) for row-major matrices
void matmul_at_b(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // C^T = B^T @ (A^T)^T = B^T @ A
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, B, n, A, m, &beta, C, n);
}

void relu(float *x, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu_kernal<<<numBlocks, blockSize, 0, stream>>>(x, size);
}

void bias_add(float *x, float *bias, int batch_size, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (batch_size * size + blockSize - 1) / blockSize;
    add_bias_kernel<<<numBlocks, blockSize, 0, stream>>>(x, bias, batch_size, size);
}

void softmax(float *x, int batch_size, int size, cudaStream_t stream) {
    softmax_kernal<<<batch_size, 1, 0, stream>>>(x, batch_size, size);
}

void zero_grad(float *grad, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    zero_grad_kernel<<<numBlocks, blockSize, 0, stream>>>(grad, size);
}

void relu_derivative(float *x, float *d_ReLU, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu_derivative_kernel<<<numBlocks, blockSize, 0, stream>>>(x, d_ReLU, size);
}

void multiply_gradients(float *grad1, float *grad2, int size, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    multiply_gradients_kernel<<<numBlocks, blockSize, 0, stream>>>(grad1, grad2, size);
}

// Forward pass
void forward(cublasHandle_t handle, NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size, cudaStream_t stream) {
    // Set the cuBLAS stream for this forward pass
    cublasSetStream(handle, stream);

    // Input to Hidden
    matmul_a_b(handle, d_input, nn->d_weights1, d_hidden, batch_size, HIDDEN_SIZE, INPUT_SIZE);
    bias_add(d_hidden, nn->d_bias1, batch_size, HIDDEN_SIZE, stream);
    relu(d_hidden, batch_size * HIDDEN_SIZE, stream);

    // Hidden to Output
    matmul_a_b(handle, d_hidden, nn->d_weights2, d_output, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    bias_add(d_output, nn->d_bias2, batch_size, OUTPUT_SIZE, stream);
    softmax(d_output, batch_size, OUTPUT_SIZE, stream);
}

// Cross-entropy loss
float cross_entropy_loss(float *output, int *labels, int batch_size){
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++){
        total_loss -= logf(fmaxf(output[i * OUTPUT_SIZE + labels[i]], 1e-8f)); // Clamp the min value to 1e-8 to avoid log(0)
    }
    return total_loss / batch_size;
}

// compute output gradients
__global__ void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * OUTPUT_SIZE) {
        int b = idx / OUTPUT_SIZE;
        grad_output[idx] = output[idx];
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// Backward pass
void backward(cublasHandle_t handle, NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels, int batch_size, cudaStream_t stream) {
    // Set the cuBLAS stream for this backward pass
    cublasSetStream(handle, stream);

    // Zero out gradients
    zero_grad(nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE, stream);
    zero_grad(nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE, stream);
    zero_grad(nn->d_grad_bias1, HIDDEN_SIZE, stream);
    zero_grad(nn->d_grad_bias2, OUTPUT_SIZE, stream);

    // allocate device memory for gradients
    float *d_grad_output;
    cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * sizeof(float));

    // Compute output gradients
    compute_output_gradients<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256, 0, stream>>>(d_grad_output, d_output, d_labels, batch_size);

    // Update gradients for W2 (W2grad = Dloss.T @ H)
    matmul_at_b(handle, d_hidden, d_grad_output, nn->d_grad_weights2, HIDDEN_SIZE, OUTPUT_SIZE, batch_size);

    // Compute dX2
    float *d_dX2;
    cudaMalloc(&d_dX2, batch_size * HIDDEN_SIZE * sizeof(float));
    matmul_a_bt(handle, d_grad_output, nn->d_weights2, d_dX2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);

    // Compute d_ReLU_out
    float *d_grad_hidden;
    cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_SIZE * sizeof(float));
    relu_derivative(d_hidden, d_grad_hidden, batch_size * HIDDEN_SIZE, stream); // d_ReLU
    multiply_gradients(d_dX2, d_grad_hidden, batch_size * HIDDEN_SIZE, stream); // dX2 * d_ReLU

    // update gradients for W1 (W1grad = d_reLU_out @ X)
    matmul_at_b(handle, d_input, d_dX2, nn->d_grad_weights1, INPUT_SIZE, HIDDEN_SIZE, batch_size);

    // Free allocated memory
    cudaFree(d_grad_output);
    cudaFree(d_dX2);
    cudaFree(d_grad_hidden);
}

// Gradient descent step
__global__ void update_weights(float *weights, float *grad_weights, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= grad_weights[idx] * LEARNING_RATE;
    }
}

// Initialize neural network
void initialize_nn(NeuralNetwork *nn) {
    // Allocate host memory
    nn->weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->grad_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases
    initialize_weights(nn->weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);

    // Allocate device memory
    cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float));

    // Copy weights and biases to device
    cudaMemcpy(nn->d_weights1, nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->d_weights2, nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->d_bias1, nn->bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nn->d_bias2, nn->bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
}

void train(cublasHandle_t handle, NeuralNetwork *nn, float *X_train, int *y_train, float* gpu_time, float* host_time){
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Create events to time GPU work
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float elapsed_ms;

    float *d_X_train_batch1, *d_X_train_batch2, *d_hidden1, *d_hidden2, *d_output1, *d_output2;
    int *d_y_train_batch1, *d_y_train_batch2;

    cudaMalloc(&d_X_train_batch1, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_X_train_batch2, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_hidden2, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output1, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_y_train_batch1, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_y_train_batch2, BATCH_SIZE * sizeof(int));

    int number_of_batches = TRAINING_SIZE / BATCH_SIZE;
    for(int epoch = 0; epoch < EPOCHS; epoch++){
        float total_loss = 0.0f;
        int correct = 0;
        for(int batch = 0; batch < number_of_batches; batch += 2){
            int start1 = batch * BATCH_SIZE;
            int start2 = (batch + 1) * BATCH_SIZE;

            cudaEventRecord(start_event);

            // Stream 1
            cudaMemcpyAsync(d_X_train_batch1, &X_train[start1 * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(d_y_train_batch1, &y_train[start1], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
            forward(handle, nn, d_X_train_batch1, d_hidden1, d_output1, BATCH_SIZE, stream1);
            backward(handle, nn, d_X_train_batch1, d_hidden1, d_output1, d_y_train_batch1, BATCH_SIZE, stream1);

            // Stream 2
            if (batch + 1 < number_of_batches) {
                cudaMemcpyAsync(d_X_train_batch2, &X_train[start2 * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream2);
                cudaMemcpyAsync(d_y_train_batch2, &y_train[start2], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
                forward(handle, nn, d_X_train_batch2, d_hidden2, d_output2, BATCH_SIZE, stream2);
                backward(handle, nn, d_X_train_batch2, d_hidden2, d_output2, d_y_train_batch2, BATCH_SIZE, stream2);
            }

            cudaStreamSynchronize(stream1);
            update_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256, 0, stream1>>>(nn->d_weights1, nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE);
            update_weights<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256, 0, stream1>>>(nn->d_weights2, nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE);
            update_weights<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream1>>>(nn->d_bias1, nn->d_grad_bias1, HIDDEN_SIZE);
            update_weights<<<(OUTPUT_SIZE + 255) / 256, 256, 0, stream1>>>(nn->d_bias2, nn->d_grad_bias2, OUTPUT_SIZE);
            if (batch + 1 < number_of_batches) {
                cudaStreamSynchronize(stream2);
                update_weights<<<(INPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256, 0, stream2>>>(nn->d_weights1, nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE);
                update_weights<<<(HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256, 256, 0, stream2>>>(nn->d_weights2, nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE);
                update_weights<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream2>>>(nn->d_bias1, nn->d_grad_bias1, HIDDEN_SIZE);
                update_weights<<<(OUTPUT_SIZE + 255) / 256, 256, 0, stream2>>>(nn->d_bias2, nn->d_grad_bias2, OUTPUT_SIZE);
            }
            
            cudaEventRecord(stop_event);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
            *gpu_time += elapsed_ms;
            
            // host timer for D2H copies and CPU calculations
            auto start_host = std::chrono::high_resolution_clock::now();

            // Loss and Accuracy (on CPU)
            float *output1 = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
            cudaMemcpy(output1, d_output1, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            int *labels1 = (int *)malloc(BATCH_SIZE * sizeof(int));
            cudaMemcpy(labels1, d_y_train_batch1, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
            total_loss += cross_entropy_loss(output1, labels1, BATCH_SIZE);
            for(int i = 0; i < BATCH_SIZE; i++){ int p=0; for(int j=1; j<OUTPUT_SIZE; j++){ if(output1[i*OUTPUT_SIZE+j]>output1[i*OUTPUT_SIZE+p]){ p=j; } } if(p==labels1[i]){ correct++; } }
            free(output1); free(labels1);

            if (batch + 1 < number_of_batches) {
                float *output2 = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
                cudaMemcpy(output2, d_output2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                int *labels2 = (int *)malloc(BATCH_SIZE * sizeof(int));
                cudaMemcpy(labels2, d_y_train_batch2, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                total_loss += cross_entropy_loss(output2, labels2, BATCH_SIZE);
                for(int i = 0; i < BATCH_SIZE; i++){ int p=0; for(int j=1; j<OUTPUT_SIZE; j++){ if(output2[i*OUTPUT_SIZE+j]>output2[i*OUTPUT_SIZE+p]){ p=j; } } if(p==labels2[i]){ correct++; } }
                free(output2); free(labels2);
            }

            auto stop_host = std::chrono::high_resolution_clock::now();
            *host_time += std::chrono::duration<float, std::milli>(stop_host - start_host).count();

            if ((batch + 2) % 100 == 0 || (epoch == 0 && batch == 0)) { printf("Epoch %d/%d, Batch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, EPOCHS, batch + 2, number_of_batches, total_loss / (batch + 2), (float)correct / ((batch + 2) * BATCH_SIZE) * 100.0f); }
        }
        printf("Epoch %d completed. Average Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, total_loss / number_of_batches, (float)correct / TRAINING_SIZE * 100.0f);
    }

    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    cudaFree(d_X_train_batch1); cudaFree(d_X_train_batch2);
    cudaFree(d_hidden1); cudaFree(d_hidden2);
    cudaFree(d_output1); cudaFree(d_output2);
    cudaFree(d_y_train_batch1); cudaFree(d_y_train_batch2);
}

int main() {
    srand(time(NULL));
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    NeuralNetwork nn;
    initialize_nn(&nn);
    float *X_train; int *y_train;
    cudaMallocHost((void**)&X_train, TRAINING_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost((void**)&y_train, TRAINING_SIZE * sizeof(int));
    load_data("data/X_train.bin", X_train, TRAINING_SIZE * INPUT_SIZE);
    load_labels("data/y_train.bin", y_train, TRAINING_SIZE);

    float gpu_compute_ms = 0;
    float host_computation_ms = 0;

    auto total_start = std::chrono::high_resolution_clock::now();

    train(cublas_handle, &nn, X_train, y_train, &gpu_compute_ms, &host_computation_ms);

    auto total_stop = std::chrono::high_resolution_clock::now();
    float total_time_ms = std::chrono::duration<float, std::milli>(total_stop - total_start).count();
    
    float memory_transfers_ms = total_time_ms - gpu_compute_ms - host_computation_ms;

    printf("\n--- Timing Results ---\n");
    printf("Memory Transfers (Inferred): %.3f s\n", memory_transfers_ms / 1000.0f);
    printf("GPU Compute Time:            %.3f s\n", gpu_compute_ms / 1000.0f);
    printf("Host Computation:            %.3f s\n", host_computation_ms / 1000.0f);
    printf("--------------------------------\n");
    printf("Total Run Time:              %.3f s\n", total_time_ms / 1000.0f);
    printf("\nNote: 'Memory Transfers' is inferred because H2D copies overlap with GPU compute.\n");

    cudaFreeHost(X_train); cudaFreeHost(y_train);
    free(nn.weights1); free(nn.weights2); free(nn.bias1); free(nn.bias2);
    free(nn.grad_weights1); free(nn.grad_weights2); free(nn.grad_bias1); free(nn.grad_bias2);
    cudaFree(nn.d_weights1); cudaFree(nn.d_weights2); cudaFree(nn.d_bias1); cudaFree(nn.d_bias2);
    cudaFree(nn.d_grad_weights1); cudaFree(nn.d_grad_weights2); cudaFree(nn.d_grad_bias1); cudaFree(nn.d_grad_bias2);
    cublasDestroy(cublas_handle);
    
    return 0;
}