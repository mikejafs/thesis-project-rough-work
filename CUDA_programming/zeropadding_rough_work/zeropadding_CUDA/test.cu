#include <iostream>
#include <cuda_runtime.h>

__device__ int max_diff(const int* edges, int size) {
    int max_val = edges[1] - edges[0];
    for (int i = 1; i < size - 1; i++) {
        int diff = edges[i + 1] - edges[i];
        if (diff > max_val) {
            max_val = diff;
        }
    }
    return max_val;
}

__global__ void zeroPadKernel(int* array, int* edges, int edges_size, int** out, int n_blocks, int largest_block) {
    int block_idx = blockIdx.x;

    if (block_idx < n_blocks) {
        int start = edges[block_idx];
        int stop = edges[block_idx + 1];
        int block_size = stop - start;

        for (int i = 0; i < largest_block; i++) {
            if (i < block_size) {
                out[block_idx][i] = array[start + i];
            } else {
                out[block_idx][i] = 0;  // Fill with zeros for padding
            }
        }
    }
}

void zeroPad(int* array, const int* edges, int edges_size, int**& out, int& out_n_blocks, int& out_largest_block) {
    out_n_blocks = edges_size - 1;
    out_largest_block = edges[edges_size - 1] - edges[0];  // Largest block size

    // Allocate device memory
    int* d_array;
    int* d_edges;
    int** d_out;
    cudaMalloc(&d_array, n_rows * n_cols * sizeof(int));
    cudaMalloc(&d_edges, edges_size * sizeof(int));
    cudaMalloc(&d_out, out_n_blocks * sizeof(int*));

    // Allocate memory for output blocks
    int* h_out[n_blocks]; // Temporary host pointers
    for (int i = 0; i < out_n_blocks; i++) {
        cudaMalloc(&h_out[i], out_largest_block * sizeof(int));
        cudaMemcpy(&d_out[i], &h_out[i], sizeof(int*), cudaMemcpyHostToDevice);
    }

    // Copy data to device
    cudaMemcpy(d_array, array, n_rows * n_cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, edges_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    zeroPadKernel<<<out_n_blocks, 1>>>(d_array, d_edges, edges_size, d_out, out_n_blocks, out_largest_block);
    cudaDeviceSynchronize();

    // Copy back the output
    out = (int**)malloc(out_n_blocks * sizeof(int*));
    cudaMemcpy(out, d_out, out_n_blocks * sizeof(int*), cudaMemcpyDeviceToHost);
    
    // Copy each block back to host
    for (int i = 0; i < out_n_blocks; i++) {
        out[i] = (int*)malloc(out_largest_block * sizeof(int));
        cudaMemcpy(out[i], h_out[i], out_largest_block * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(h_out[i]); // Free device memory for each block
    }

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_edges);
    cudaFree(d_out);
}

int main() {
    const int n_rows = 20;
    const int n_cols = 4;
    int array[n_rows][n_cols];
    int edges[] = {0, 4, 15, 18, n_rows};
    int edges_size = sizeof(edges) / sizeof(edges[0]);

    // Fill the array with random numbers
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            array[i][j] = rand() % 100;
        }
    }

    std::cout << "The original array was: \n";
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            std::cout << array[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nAfter zeropadding: \n";
    int** total_result;
    int out_n_blocks, out_largest_block;

    zeroPad((int*)array, edges, edges_size, total_result, out_n_blocks, out_largest_block);

    for (int col = 0; col < out_n_blocks; col++) {
        std::cout << "Column " << col << ": ";
        for (int j = 0; j < out_largest_block; j++) {
            std::cout << total_result[col][j] << " ";
        }
        std::cout << "\n";
        free(total_result[col]);  // Free allocated memory for each block
    }
    free(total_result);  // Free the array of pointers

    return 0;
}
