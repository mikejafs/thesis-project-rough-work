#include <stdio.h>


//For the 1D case (note this will still be 2d since we need to treat the blocks as new rows)
//Begin by just doing for a single block, then move to multiple blocks

// For just the case of a single block:
void zeropad(
    double *in,
    double *out,
    long *edges,
    int n_blocks,
    int largest_block)
    {
        
        // allocate memory for size of largest block * n_blocks
        // double* temp = (double*)calloc(largest_block, sizeof(double));

        int block_size = edges[1] - edges[0];   //for the case of only a single block
        for (int i = 0; i < block_size; i++){
            out[i] = in[i];
        }

    }