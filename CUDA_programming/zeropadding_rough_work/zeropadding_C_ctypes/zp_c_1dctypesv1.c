#include <stdio.h>
#include <stdlib.h>

/*
Same as contiguous zeropad function but designed to be run from a python script
using ctypes functionality. Essentially only difference is we don't need a main() function.
*/
 
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

// Rewriting zeroPad to use a single pointer for a 1D array
int* zeroPad(int* array, 
    const int* edges, 
    int edges_size, 
    int* out_rows, 
    int* out_cols
    ) {
        int largest_block = max_diff(edges, edges_size);
        int n_blocks = edges_size - 1;

        *out_rows = n_blocks;
        *out_cols = largest_block;

        // Allocate memory for a single 1D array to store the padded output
        int* out = (int*)calloc(n_blocks * largest_block, sizeof(int)); // Single pointer, flat memory

        // Perform the zero-padding
        for (int block = 0; block < n_blocks; block++) {
            int start = edges[block];
            int stop = edges[block + 1];
            int block_size = stop - start;

            // Copy the elements from the original array to the appropriate position in the 1D array
            for (int i = 0; i < block_size; i++) {
                out[block * largest_block + i] = array[start + i];
            }
        }
        return out;
}

// A function to free the memory allocated by zeroPad
void free_memory(int* ptr) {
    free(ptr);
}


