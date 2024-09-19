#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
Attempt at writing the 2D version of the zeropad function from the zeropad_1.0 file.

Uses the notion of a triple pointer to dereference three times, allowing us to loop
over columns of the 2D array.

It is now essential to add another layer of data de-alloc ('freeing') at the end now.
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

int** zeroPad(int* array, const int* edges, int edges_size, int* out_n_blocks, int* out_largest_block){
    int largest_block = max_diff(edges, edges_size);
    int n_blocks = edges_size - 1;

    // *out_rows = n_blocks;
    // *out_cols = largest_block;

    // Return n_blocks and largest block in main function below
    *out_n_blocks = n_blocks;
    *out_largest_block = largest_block;

    int** out = (int**)malloc(n_blocks * sizeof(int*));
    for (int i = 0; i < n_blocks; i++) {
        out[i] = (int*)calloc(largest_block, sizeof(int));  // calloc caused zero-initialized (all values 0)
    }

    // perform the zero-padding
    for (int block = 0; block < n_blocks; block++){
        int start = edges[block];
        int stop = edges[block + 1];
        int block_size = stop - start;

        for (int i = 0; i < block_size; i++){
            out[block][i] = array[start + i];
        }
    }
    return out;
}

int main(){

    int n_rows = 20;
    int n_cols = 4;
    int array[n_rows][n_cols];

    int edges[] = {0, 4, 15, 18, n_rows};
    int edges_size = sizeof(edges) / sizeof(edges[0]);

    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            array[i][j] = rand() % 100;
        } 
    }

    printf("The original array was: \n");
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++){
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }

    printf("\n \n");
    printf("After zeropadding: \n");

    int out_n_blocks, out_largest_block;
    int*** total_result = (int***)malloc(n_cols * sizeof(int**));

        for (int col = 0; col < n_cols; col ++){

            // Create a temp flattened array for representing each column
            int* flat_array = malloc(n_rows * sizeof(int));
            for (int i = 0; i < n_rows; i++){
                flat_array[i] = array[i][col];  //Assign the flattened array to the current column being looped through
            }

            // Feed the flattened array to the zeroPad function above and proceed as in the 1D case, but do for each column
            total_result[col] = zeroPad(flat_array, edges, edges_size, &out_n_blocks, &out_largest_block);
            free(flat_array);

            // Print a space between each new column
            printf("\n");

            // Identify each new column by its index while printing
            printf("Column %d: \n", col);    // Uncomment to have a better picture of the zeropadding (each zeropadded column on a new line)
            for (int i = 0; i < out_n_blocks; i++) {
                for (int j = 0; j < out_largest_block; j++) {
                // dereference result with i, j, k
                    printf("%d ", total_result[col][i][j]); // Dereferences to get the actual integer value
                }
            // printf("\n");   // Uncomment to have a better picture of the zeropadding (each zeropadded block on a new line)
            }

            // Print each new column on a new line
            printf("\n");

            //would need another layer of freeing
            for (int i = 0; i < out_n_blocks; i++) {
                free(total_result[col][i]);
            }
            free(total_result[col]);
        }

    free(total_result);
    
    return 0;
}





