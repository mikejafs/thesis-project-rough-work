#include <stdio.h>
#include <stdlib.h>

/*
Modifying the contiguous 1d zeropad code so that ALL initialization
will happen from within the Python cell
*/

void zeroPad(
    double* in_array,
    double* out_array,
    long* edges,
    int n_blocks,
    int largest_block
){
    for (int block = 0; block < n_blocks; block++){
        // printf("Test 1");
        int start = edges[block];
        int stop = edges[block + 1];
        int block_size = stop - start;
        for (int i = 0; i < block_size; i++){
            out_array[block*largest_block + i] = in_array[start + i];
        } 
    }
}

