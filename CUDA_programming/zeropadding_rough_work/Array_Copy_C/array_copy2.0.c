//Final approach for writing an array copy routine that uses recursion

#include <stdio.h>

int copy_array(int a[], int b[], int n, int i)
{
    if (i < n){
        b[i] = a[i];
        copy_array(a, b, n, ++i);
    }
}

int array(int a[], int n)
{
    int i;
    for (i = 0; i < n; i++){
        printf("%d", a[i]);
    }
}

int main()
{
    int k[5] = {3, 6, 9, 2, 5}, n = 5;
    int l[n], i;

    copy_array(k, l, n, 0);

    printf("first array:");
    array(k, n);

    printf("\nsecond array:");
    array(l, n);
    printf("\n");

    return 0;
}