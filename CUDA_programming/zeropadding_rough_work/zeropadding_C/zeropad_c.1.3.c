#include <stdio.h>
#include <stdlib.h>

void zeropad(double *array, int *edges, int n_edges, double ***out, int **out_shape, int array_dim) {
    int largest_block = 0;
    for (int i = 1; i < n_edges; i++) {
        int block_size = edges[i] - edges[i - 1];
        if (block_size > largest_block) {
            largest_block = block_size;
        }
    }
    
    int n_blocks = n_edges - 1;
    
    // Allocate memory for output shape
    *out_shape = (int *)malloc(2 * sizeof(int)); // Change size based on array_dim
    if (array_dim > 1) {
        *out_shape = (int *)malloc(3 * sizeof(int));
        (*out_shape)[2] = array_dim;
    }
    (*out_shape)[0] = n_blocks;
    (*out_shape)[1] = largest_block;

    if (array_dim == 1) { // Noise matrix case
        *out = (double **)malloc(n_blocks * sizeof(double *));
        for (int i = 0; i < n_blocks; i++) {
            (*out)[i] = (double *)malloc(largest_block * sizeof(double));
            for (int j = 0; j < largest_block; j++) {
                (*out)[i][j] = 0.0; // Initialize to zero
            }
        }
    } else { // Diffuse/source matrices case
        *out = (double **)malloc(n_blocks * sizeof(double *));
        for (int i = 0; i < n_blocks; i++) {
            (*out)[i] = (double *)malloc(largest_block * array_dim * sizeof(double));
            for (int j = 0; j < largest_block * array_dim; j++) {
                (*out)[i][j] = 0.0; // Initialize to zero
            }
        }
    }

    for (int block = 0; block < n_blocks; block++) {
        int start = edges[block];
        int stop = edges[block + 1];
        for (int j = 0; j < (stop - start); j++) {
            (*out)[block][j] = array[start + j];
        }
    }
}

// Example usage
int main() {
    double array[] = {1, 2, 3, 4, 5, 6};
    int edges[] = {0, 2, 5};
    double **out;
    int *out_shape;
    int n_edges = sizeof(edges) / sizeof(edges[0]);
    int array_dim = 1; // Change this to 2 for diffuse/source matrices

    zeropad(array, edges, n_edges, &out, &out_shape, array_dim);

    // Print output
    for (int i = 0; i < out_shape[0]; i++) {
        for (int j = 0; j < out_shape[1]; j++) {
            printf("%lf ", out[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < out_shape[0]; i++) {
        free(out[i]);
    }
    free(out);
    free(out_shape);

    return 0;
}
