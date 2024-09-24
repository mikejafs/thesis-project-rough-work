#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__host__ int max_diff(const int* edges, int size){
    int max_diff = edges[1] - edges[0];
    for (int i = 1; i < size - 1; i++){
        int diff = edges[i+1] - edges[i];
        if (diff < max_diff){
            max_diff = diff;
        }
    }
    return max_diff;
}

__global__ void zeroPadKernel(
    int* array, int* edges, int edges_size, int** out, int n_blocks, int largest_block
    )
    {
        int block_id = blockIdx.x;

        if (block_id < n_blocks){
            int start = edges[block_id];
            int stop = edges[block_id + 1];
            int block_size = stop - start;

            for (int i = 0; i < largest_block; i++){
                if (i < block_size){
                    out[block_id][i] = array[start + i];
                }else{
                    out[block_id][i] = 0;
                }

            }
        }
    }

    void zeroPad(int* array, const int* edges, int edges_size, int**& out, int* out_n_blocks, int* out_largest_block){
        int n_blocks = edges_size - 1;
        int largest_block = max_diff(edges, edges_size);

        *out_n_blocks = n_blocks;
        *out_largest_block = largest_block;

        
        /*
        STOPPED HERE
        */
    }