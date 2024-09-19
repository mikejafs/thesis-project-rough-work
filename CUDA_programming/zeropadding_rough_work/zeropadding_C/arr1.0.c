#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// void printArray(int arr[], int size){
//     for (int i = 0; i < size; i++){
//         printf("%d", arr[i]);
//     }
//     printf("\n");
// }


// int main(){
//     int array[] = {{1, 2},
//     {3, 4}};
//     int size = sizeof(array) / sizeof(array[0]);

//     printArray(array, size);

//     return 0;
// }

// #define ROWS = 2
// #define COLS = 3


// int main()
// {
//     // int ROWS = 2;
//     // int COLS = 3;
//     int array[2][3] = {
//                         {1, 2, 3}, 
//                         {4, 5, 6}
//                         };

//     int out[2][3]; 
//     for (int i = 0; i < 2; i++)
//     {
//         for (int j = 0; j < 3; j++)
//         {
//             printf("%d", array[i][j]);
//         }
//             out[i][j] = array[i][j];

//     }
//     print(out[][]);
// };

// int main(){

//     int n_rows = 10;
//     int n_cols = 2;
//     int array[n_rows][n_cols];
    
//     for (int i = 0; i < n_rows; i++){
//         for (int j = 0; j < n_cols; j++){
//             array[i][j] = rand() % 10;       
//         }
//     }

//     for (int i = 0; i < n_rows; i++){
//         for (int j = 0; j < n_cols; j++){
//             printf("%d ", array[i][j]);
//         }
//         // printf("\n");
//     }
// }
    

//     #include <stdio.h>

// int main() {
//     // Define the dimensions of the 3D matrix
//     int depth = 2;
//     int rows = 3;
//     int cols = 4;

//     // Declare and initialize a 3D array
//     int matrix[depth][rows][cols];

//     // Populate the 3D array with values
//     for (int d = 0; d < depth; d++) {
//         for (int r = 0; r < rows; r++) {
//             for (int c = 0; c < cols; c++) {
//                 matrix[d][r][c] = d * rows * cols + r * cols + c; // Just an example value
//             }
//         }
//     }

//     // Print the 3D matrix
//     for (int d = 0; d < depth; d++) {
//         printf("Depth %d:\n", d);
//         for (int r = 0; r < rows; r++) {
//             for (int c = 0; c < cols; c++) {
//                 printf("%d ", matrix[d][r][c]);
//             }
//             printf("\n");
//         }
//         printf("\n"); // New line after each depth layer
//     }

//     return 0;
// }


int main()
{
    
    int largest_block = 3;
    int n_blocks = 4;

    int out[largest_block * n_blocks][2];

    for (int i = 0; i < (largest_block * n_blocks); i++){
        for (int j = 0; j < 2; j++){
            out[i][j] = 0;
        }
    }

    
    for (int i = 0; i < (largest_block * n_blocks); i++){
        for (int j = 0; j < 2; j++){
            printf("%d ", out[i][j]);
        }
        printf("\n");
    }

    return 0;
}

