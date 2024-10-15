#include <stdio.h>
#include <stdlib.h>

/*
Modifying the contiguous 1D zero-padding code so that all initialization
will happen from within the Python cell.
*/

void zeroPad(
    double* in_array,
    double* out_array,
    unsigned long* edges,   // Use unsigned long for edges
    unsigned long n_blocks, // Use unsigned long for n_blocks
    unsigned long largest_block // Use unsigned long for largest_block
){
    printf("Welcome 1\n");
    for (unsigned long block = 0; block < n_blocks; block++){
        printf("Welcome 2\n");
        unsigned long start = edges[block];
        unsigned long stop = edges[block + 1];
        unsigned long block_size = stop - start;
        for (unsigned long i = 0; i < block_size; i++){
            out_array[block * largest_block + i] = in_array[start + i];
        }
    }
}
