#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

// Helper function to calculate the maximum difference in the "edges" array
int max_diff(const int* edges, int size) {
    int max_val = edges[1] - edges[0];
    for (int i = 1; i < size - 1; i++) {
        int diff = edges[i + 1] - edges[i];
        if (diff > max_val) {
            max_val = diff;
        }
    }
    return max_val;
}

// Function for zero-padding an array based on edges
double** zeropad(double* array, int array_rows, int array_cols, const int* edges, int edges_size, int* out_rows, int* out_cols) {
    // Calculate the largest block and the number of blocks
    int largest_block = max_diff(edges, edges_size);
    int n_blocks = edges_size - 1;
    
    // Set the output dimensions
    *out_rows = n_blocks;
    *out_cols = largest_block;

    // Allocate memory for the output array
    double** out = (double**)malloc(n_blocks * sizeof(double*));
    for (int i = 0; i < n_blocks; i++) {
        out[i] = (double*)calloc(largest_block, sizeof(double));  // Zero-initialized (all values 0)
    }

    // Fill the output array based on the edges
    for (int block = 0; block < n_blocks; block++) {
        int start = edges[block];
        int stop = edges[block + 1];
        int block_size = stop - start;

        // Fill the block in the output array
        for (int i = 0; i < block_size; i++) {
            if (array_cols == 1) {
                // If it's a 1D input array (noise matrix case)
                out[block][i] = array[start + i];
            } else {
                // If it's a 2D input array (diffuse or source matrix case)
                out[block][i] = array[(start + i) * array_cols];  // Assume the input is flattened 2D array
            }
        }
    }
    
    return out;
}

// Test the function
int main() {
    // Example input arrays
    int edges[] = {0, 2, 4};  // Edges of the blocks
    int edges_size = sizeof(edges) / sizeof(edges[0]);
    
    // 1D array example (noise matrix)
    double array[] = {{1.0, 2.0, 3.0, 4.0}};
    int array_rows = 4;
    int array_cols = 1;

    // Output array dimensions
    int out_rows, out_cols;
    
    // Call the zeropad function
    double** result = zeropad(array, array_rows, array_cols, edges, edges_size, &out_rows, &out_cols);
    
    // Print the output array
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            printf("%.2f ", result[i][j]);
        }
        printf("\n");
    }
    
    // Free allocated memory
    for (int i = 0; i < out_rows; i++) {
        free(result[i]);
    }
    free(result);
    
    return 0;
}
