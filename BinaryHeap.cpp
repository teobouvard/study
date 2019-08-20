#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    int allocated; /* current allcoation of array */
    int filled;    /* number of items present in the binheap */
    int *array;    /* array of values */
} BinaryHeap;

/* Init allocates the structure BinaryHeap and
 * also the membre array with the given size
 * it also fill allocated (size) and intializes
 * filled to 0 */
BinaryHeap * Init(int size);

/* InsertValue insert value into the binary heap
 * the array is reallocated if necessary (allocated changed
 * with respect to the new size )
 * filled is incremented by 1 */
bool InsertValue(BinaryHeap * heap, int value);

/* ExtractMax returns 0 if the binary heap is empty
 * otherwise it returns 1 and fills *val with the maximum
 * value present in the binary heap
 * filled is decremented by 1  and the max value is removed
 * from the binary heap */
int ExtractMax(BinaryHeap * heap, int * val);

/* Destroy frees the structure and the array */
void Destroy(BinaryHeap * heap);


int main(void)
{
    char lecture[100];
    int val;
    BinaryHeap * heap;
    heap = Init(10);
    
    fscanf(stdin, "%99s", lecture);
    while (strcmp(lecture, "bye") != 0) {
        if (strcmp(lecture, "insert") == 0) {
            fscanf(stdin, "%99s", lecture);
            val = (int)strtol(lecture, NULL, 10);
            InsertValue(heap, val);
            
        }
        else if (strcmp(lecture, "extract") == 0) {
            if (ExtractMax(heap, &val))
            {
                printf("%d\r\n", val);
            }
        }
        fscanf(stdin, "%99s", lecture);
    }
    Destroy(heap);
    return 0;
}

BinaryHeap * Init(int size)
{
    BinaryHeap * heap;
    heap = (BinaryHeap *)malloc(sizeof(BinaryHeap));
    heap->allocated = size;
    heap->filled = 0;
    heap->array = (int*)malloc(sizeof(int)*size);
    
    return heap;
}

bool InsertValue(BinaryHeap * heap, int value)
{
    bool ans = false;
    
    int indice, pere, tmp;
    if (heap->filled == heap->allocated) {
        heap->allocated += 10;
        heap->array = (int*)realloc(heap->array, sizeof(int)*heap->allocated);
    } else {
        heap->array[heap->filled] = value;
        indice = heap->filled;
        heap->filled++;
        pere = (indice - 1) / 2;
        while (indice > 0 && heap->array[pere] < heap->array[indice]) {
            tmp = heap->array[pere];
            heap->array[pere] = heap->array[indice];
            heap->array[indice] = tmp;
            indice = pere;
            pere = (indice - 1) / 2;
        }
        ans = true;
    }
    
    return ans;
    
    
}

int ExtractMax(BinaryHeap * heap, int *res)
{
    if (heap->filled == 0){
        return 0;
    }
    else{
        
        *res = heap->array[0];
        heap->array[0] = heap->array[heap->filled - 1];
        heap->filled--;
        
        int indice = 0;
        int filsGauche = 1;
        int filsDroit = 2;
        int tmp;                                                            //variable temporaire pour échanger deux cellules
        
        while(indice < heap->filled){                                       //tant que l'on a pas descendu tout le tas
            if(filsDroit >= heap->filled){                                  //si il n'y a qu'un fils à gauche
                if(heap->array[filsGauche] > heap->array[indice]){          //on compare la cellule courante avec le fils gauche
                    tmp = heap->array[indice];
                    heap->array[indice] = heap->array[filsGauche];          //on échange si l'ordre n'est pas le bon
                    heap->array[filsGauche] = tmp;
                }
                return 1;
            }
            else{
                if(heap->array[filsGauche] > heap->array[filsDroit]){
                    tmp = heap->array[indice];
                    heap->array[indice] = heap->array[filsGauche];
                    heap->array[filsGauche] = tmp;
                    indice = 2*indice + 1;
                }
                else{
                    tmp = heap->array[indice];
                    heap->array[indice] = heap->array[filsDroit];
                    heap->array[filsDroit] = tmp;
                    indice = 2*indice + 2;
                }
            }
        }
    }
    return 1;
}

void Destroy(BinaryHeap * heap)
{
    /* put your destruction code here */
}
