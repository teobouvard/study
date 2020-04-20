#define BIGNUMBER 1000000
#include <iostream>
#include <math.h>
using namespace std;


int maxVecteur(int vect[], int taille){
	int ans = vect[0];
	
	for (int i = 0; i < taille; i++){
		if (vect[i] > ans){
			ans = vect[i];
		}
	}

	return ans;
}

int minVecteur(int vect[], int taille){
	int ans = vect[0];
	
	for (int i = 0; i < taille; i++){
		if (vect[i] < ans){
			ans = vect[i];
		}
	}

	return ans;
}

int maxNegatifVecteur(int vect[], int taille){
	int ans = -BIGNUMBER;
	
	for (int i = 0; i < taille; i++){
		if (vect[i] > ans && vect[i] < 0){
			ans = vect[i];
		}
	}
	
	return ans;
}

int minPositifVecteur(int vect[], int taille){
	int ans = BIGNUMBER;
	
	for (int i = 0; i < taille; i++){
		if (vect[i] < ans && vect[i] > 0){
			ans = vect[i];
		}
	}
	
	return ans;
}


int main() {
	
	int nbElements = 0;
	int lecture;
	int centroid, diameter, result;
	
	cin >> nbElements;
	
	int vecteur[nbElements];
	
	for (int i = 0; i < nbElements; i++){
		cin >> lecture;
		vecteur[i] = lecture;
	}
	
	diameter = maxVecteur(vecteur, nbElements) - minVecteur(vecteur, nbElements);
	centroid = minPositifVecteur(vecteur, nbElements) - maxNegatifVecteur(vecteur, nbElements);
	
	result = diameter - centroid;
	
	cout << result << "\r\n";
	
	return 0;
}

