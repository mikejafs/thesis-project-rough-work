#include <stdio.h>

//A file containing a main function that copies entries of one array to 
//another array

int main()
{

    int a[5] = {3, 4, 6, 7, 8}, n = 5;
    int b[n], i;

    for (i = 0; i < n; i++){
        b[i] = a[i]; 
    }

    printf("The first array is : ");
    for (i = 0; i < n; i++){
        printf("%d", a[i]);
    }

    printf("\nThe second array is :");
    for (i = 0; i < n; i++){
        printf("%d", b[i]);
    }
    printf("\n");  //leave a line at the end of the code since Ubuntu is annoying

    return 0;
}