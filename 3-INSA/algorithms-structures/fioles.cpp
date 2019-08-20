#include <iostream>
#include <math.h>

#define MAP

using namespace std;


int moyenneVect(const int vect[],const int taille){
	
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
	
	int capacite,nbFioles,lecture;
	int result = 0;
	
	cin >> capacite;
	cin >> nbFioles;
	
	int vecteur[nbFioles];
	
	for (int i = 0; i < nbFioles; i++){
		cin >> lecture;
		vecteur[i] = lecture;
		
		#ifdef MAP
		cout << "vecteur[" << i << "] = " << lecture << endl;
		#endif
	}
	
	
	
	cout << result << "\r\n";
	
	return 0;
}

