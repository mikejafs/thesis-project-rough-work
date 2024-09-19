#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
The goal of this file is to modify the original zeropad function to 
be able to handle 2-d arrays, with the added ability to do it dynamically.
*/

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

int** zeroPad(int* array, int array_cols, const int* edges, int edges_size, int* out_rows, int* out_cols){
    int largest_block = max_diff(edges, edges_size);
    int n_blocks = edges_size - 1;

    *out_rows = n_blocks;
    *out_cols = largest_block;

    // int** out = (int**)malloc(n_blocks * sizeof(int*));
    // for (int i = 0; i < n_blocks; i++) {
    //     out[i] = (int*)calloc(largest_block, sizeof(int));  // calloc caused zero-initialized (all values 0)
    // }

    // Allocate a contiguous block of memory for the entire 2D array
    int** out = (int**)malloc(n_blocks * sizeof(int*));
    int* data = (int*)malloc(n_blocks * largest_block * sizeof(int));

    // Set the pointers for each row
    for (int i = 0; i < n_blocks; i++) {
        out[i] = &data[i * largest_block]; // Point each row to the corresponding part of the contiguous block
    }

    // Initialize the 2D array to zero
    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < largest_block; j++) {
            out[i][j] = 0;  // Optionally initialize each element to zero
        }
    }

    // perform the zero-padding
    for (int block = 0; block < n_blocks; block++){
        int start = edges[block];
        int stop = edges[block + 1];
        int block_size = stop - start;

        for (int i = 0; i < block_size; i++){
            for (int j = 0; j < array_cols; j++) {
                // Access the flattened array using the formula: index = row * total_columns + column
                out[block][i] = array[(start + i) * array_cols + j];
            }
        }
    }
    return out;
}

int main(){

    int n_cols = 2;
    int n_rows = 20;
    int array[n_rows][n_cols];

    int edges[] = {0, 5, 12, 18, n_rows};
    int edges_size = sizeof(edges) / sizeof(edges[0]);

    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            array[i][j] = rand() % 10;       
        }
    }

    printf("The original array was: \n");
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("\n");

    // int array_rows = n;
    // int array_cols = 1;
    int out_rows, out_cols;

    int** result = zeroPad(array, n_cols, edges, edges_size, &out_rows, &out_cols);

    printf("After zeropadding: \n");
    for (int i = 0; i < out_rows; i++) {
        for (int j = 0; j < out_cols; j++) {
            printf("%d ", result[i][j]); // Dereferences to get the actual integer value
        }
        printf("\n");
    }

    for (int i = 0; i < out_rows; i++) {
        free(result[i]);
    }
    free(result);
    
    return 0;
}





