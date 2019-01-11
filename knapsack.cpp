#include <iostream>
using namespace std;


int main() {
	
	int capacite = 0;
	int nbObjets = 1;
	int objets[100] = {0};
	int lecture;
	
	cin >> capacite;
	
	cin >> lecture;
	while (lecture != -1){
		objets[nbObjets] = lecture;
		nbObjets++;
		cin >> lecture;
	}
	
	
	//initialisation du tableau de remplissage (ajout d'un objet de taille nulle et d'une capacité nulle)
	capacite++;
	int remplissage[nbObjets][capacite];
	
	for (int i = 0; i < nbObjets; i++) {
		for (int j = 0; j < capacite; j++) {
			remplissage[i][j] = 0;
		}
	}
	
	remplissage[0][0] = 1;
	for (int i = 1; i < nbObjets; i++) {
		for (int j = 0; j < capacite; j++) {
			if (remplissage[i-1][j] == 1 || ((j - objets[i] >= 0) && (remplissage[i-1][j-objets[i]] == 1))) {
				remplissage[i][j] = 1;
				
			}
		}
	}

	
	if (remplissage[nbObjets-1][capacite-1] == 0){
		cout << "NON" << "\r\n";
	}
	else{
		cout << "OUI" << "\r\n";
	}
	
	//affichage du tableau pour debug
	for (int i = 0; i < nbObjets; i++){
		for (int j = 0; j < capacite ; j++){
			if (j == (capacite - 1)){
				cout << remplissage[i][j] << endl;
			}
			else{
				cout << remplissage[i][j] << " ";
			}
		}
	}
	
	
	return 0;
}

