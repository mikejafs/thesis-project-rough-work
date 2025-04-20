#include <stdio.h>
#include <stdlib.h>

#define ROWS 3
#define COLS 4

int main(void) {
    int i, j;

    // ----------------------------
    // Contiguous 2D Array Allocation
    // ----------------------------
    // Allocate one block for ROWS x COLS integers.
    int *contiguous = malloc(ROWS * COLS * sizeof(int));
    if (contiguous == NULL) {
        perror("Failed to allocate contiguous memory");
        return EXIT_FAILURE;
    }

    // Initialize the contiguous 2D array.
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            contiguous[i * COLS + j] = i * COLS + j;
        }
    }

    // Print the contiguous array.
    printf("Contiguous 2D Array:\n");
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            printf("%2d ", contiguous[i * COLS + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Free the contiguous array.
    free(contiguous);

    // ----------------------------
    // Pointers-to-Pointers 2D Array Allocation
    // ----------------------------
    // Allocate an array of ROWS pointers.
    int **ptrArray = malloc(ROWS * sizeof(int *));
    if (ptrArray == NULL) {
        perror("Failed to allocate pointer array");
        return EXIT_FAILURE;
    }

    // For each row, allocate memory for COLS integers.
    for (i = 0; i < ROWS; i++) {
        ptrArray[i] = malloc(COLS * sizeof(int));
        if (ptrArray[i] == NULL) {
            perror("Failed to allocate a row");
            // Clean up already allocated rows before exiting.
            for (int k = 0; k < i; k++) {
                free(ptrArray[k]);
            }
            free(ptrArray);
            return EXIT_FAILURE;
        }
    }

    // Initialize the pointer-to-pointer 2D array.
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            ptrArray[i][j] = i * COLS + j;
        }
    }

    // Print the pointer-to-pointer array.
    printf("Pointer-to-Pointer 2D Array:\n");
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            printf("%2d ", ptrArray[i][j]);
        }
        printf("\n");
    }

    // Free memory allocated for the pointers-to-pointers array.
    for (i = 0; i < ROWS; i++) {
        free(ptrArray[i]);
    }
    free(ptrArray);

    return 0;
}
