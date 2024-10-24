#include <stdio.h>

int main(){
    const int n = 20;
    int array[n] = {0};

    int edges[] = {0, 5, 6, 8, 18, 20};
    int edges_size = sizeof(edges) / sizeof(edges[0]);
    int size = sizeof(edges[0]);
    printf("%d\n", size);
    printf("\n");
    for (int i = 0; i < n; i++){
        printf("%d", array[i]);
    }   
}

