#define MAP

#include <iostream>


using namespace std;

void afficherTableau(int * array, int n);
void bubbleSort(int * array, int n);


int main() {
	
	int nbElements;
	
	cin >> nbElements;
	
	int tab[nbElements];
	
	//initialisation du tableau
	for (int i = 0; i < nbElements; i++) {
		cin >> tab[i];
	}
	
	 
	afficherTableau(tab, nbElements);
	bubbleSort(tab, nbElements);
	afficherTableau(tab, nbElements);
	
	return 0;
}

void afficherTableau(int * array, int n){
	
	for (int i = 0; i < n; i++) {
		cout << array[i];
	}
	cout << endl;
}

void bubbleSort(int array[], int n){
	
	int m = n;
	
	for (int j = 0; j < n - 1; j++){
		for (int i = 0; i < n - 1 - i; i++) {
			if (array[i+1] < array[i]){
				int tmp = array[i];
				array[i] = array[i+1];
				array[i+1] = tmp;
			}
		}
	}
}



