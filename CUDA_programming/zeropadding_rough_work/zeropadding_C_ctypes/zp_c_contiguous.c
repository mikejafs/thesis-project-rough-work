#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
Original zeropad algorithm, but only uses a single pointer definition.
As such, the pointer is only "de-referenced" once when assigning data from the 
input array to the intitialized array of pure zeros.
*/

//function for max diff between edges array
int max_diff(const int* edges, int size){
    int max_val = edges[1] - edges[0];
    for (int i = 1; i < size - 1; i++){
        int diff = edges[i + 1] - edges[i];
        if (diff > max_val){
            max_val = diff;
        }
    }
    return max_val;
}

int* zeroPad(
    int* array,
    const int* edges,
    int edges_size,
    int* out_rows,   //really only define these to be able to return them
    int* out_cols   //and print out the zeropadding solution
){
    int largest_block = max_diff(edges, edges_size);
    int n_blocks = edges_size - 1;

    *out_rows = n_blocks;
    *out_cols = largest_block;

    int* out = (int*)calloc(n_blocks * largest_block, sizeof(int));

    for (int block = 0; block < n_blocks; block ++){
        int start = edges[block]; 
        int stop = edges[block + 1];
        int block_size = stop - start;
        for (int i = 0; i < block_size; i++){
            out[block*largest_block + i] = array[start + i];
        }
    }
    return out;
}

int main(){
    int n = 20;
    int array[n];

    int edges[] = {0, 5, 12, 18, n};
    int edges_size = sizeof(edges) / sizeof(edges[0]);

    // Initialize the original array with random values
    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100;
    }

    printf("The original array was:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }

    printf("\n\n");

    int out_rows, out_cols;
    int* result = zeroPad(
        array,
        edges,
        edges_size,
        &out_rows,
        &out_cols
    );

    for (int row = 0; row < out_rows; row ++){
        for (int col = 0; col < out_cols; col ++){
            printf("%d ", result[row*out_cols + col]);
        }
    }

    free(result);
    return 0;
}