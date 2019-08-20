#include <iostream>
#include <math.h>
using namespace std;


int moyenneVect(int vect[], int taille){
	
	int moyenne = 0;
	int somme = 0;
	
	for (int i = 0; i < taille; i++) {
		somme += vect[i];
	}
	
	if (taille != 0){
		moyenne = (int)(somme/taille);
	}
	else {
		moyenne = 0;
	}
	
	return moyenne;
}

int main() {
	
	int nbEntiers = 0;
	int lecture,moyenne;
	int result = 0;
	
	cin >> nbEntiers;
	
	int vecteur[nbEntiers];
	
	for (int i = 0; i < nbEntiers; i++){
		cin >> lecture;
		vecteur[i] = lecture;
	}
	
	moyenne = moyenneVect(vecteur, nbEntiers);
	
	for (int i = 0; i < nbEntiers; i++) {
		if (vecteur[i] >= moyenne && vecteur[i] > 0){
			result++;
		}
	}
	
	cout << result << "\r\n";
	
	return 0;
}

