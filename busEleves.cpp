//#define MAP

#include <iostream>
#include <math.h>
using namespace std;

int main() {
	
	int capacite = 0;
	int nbEleves = 0;
	int nb3IF = 1; //avant y'avait 1, je sais plus pourquoi ? 3IF de poids nul ?
	int nb4IF = 1;
	int tab3IF[100] = {0};
	int tab4IF[100] = {0};
	int poids, promotion;
	int capaciteMax3IF = 0;
	int capaciteMax4IF = 0;
	
	cin >> capacite;
	cin >> nbEleves;
	
	for (int i = 1; i <= nbEleves; i++){
		cin >> poids >> promotion;
		
		if (promotion == 3){
			tab3IF[nb3IF] = poids;
			nb3IF++;
		} else {
			tab4IF[nb4IF] = poids;
			nb4IF++;
		}
	}
	
	
	//initialisation du tableau de remplissage (ajout d'un objet de taille nulle et d'une capacité nulle)
	capacite++; //afin d'ajouter une colonne de capacité nulle
	
	int remplissage3IF[nb3IF][capacite];
	int remplissage4IF[nb4IF][capacite];
	
	//pour un bus de 3IF
	for (int i = 0; i < nb3IF; i++){
		for (int j = 0; j < capacite; j++){
			remplissage3IF[i][j] = 0;
		}
	}
	
	for (int i = 0; i < nb3IF; i++){
		remplissage3IF[i][0] = 1;
	}
	
	
	for (int i = 1; i < capacite; i++){
		remplissage3IF[0][i] = 0;
	}
	
	
	//calcul du remplissage effectif
	for (int i = 1; i < nb3IF; i++){
		for (int j = 1; j < capacite ; j++){
			if (remplissage3IF[i-1][j] == 1 || ((j-tab3IF[i] >= 0) && (remplissage3IF[i-1][j-tab3IF[i]] == 1))){
				remplissage3IF[i][j] = 1;
				capaciteMax3IF = j;
			}
		}
	}
	
	//pour un bus de 4IF
	for (int i = 0; i < nb4IF; i++){
		for (int j = 0; j < capacite; j++){
			remplissage4IF[i][j] = 0;
		}
	}
	
	for (int i = 0; i < nb4IF; i++){
		remplissage4IF[i][0] = 1;
	}
	
	
	for (int i = 1; i < capacite; i++){
		remplissage4IF[0][i] = 0;
	}
	
	
	//calcul du remplissage effectif
	for (int i = 1; i < nb4IF; i++){
		for (int j = 1; j < capacite ; j++){
			if (remplissage4IF[i-1][j] == 1 || ((j-tab4IF[i] >= 0) && (remplissage4IF[i-1][j-tab4IF[i]] == 1))){
				remplissage4IF[i][j] = 1;
				capaciteMax4IF = j;
			}
		}
	}
	
	if (capaciteMax3IF > capaciteMax4IF){
		cout << "3" << "\r\n";
	} else if (capaciteMax3IF < capaciteMax4IF){
		cout << "4" << "\r\n";
	} else {
		cout << "3 4" << "\r\n";
	}
	

	{
	#ifdef MAP
	
	cout << "tableau du bus de 3IF" << endl;
	for (int i = 0; i < nb3IF; i++){
		for (int j = 0; j < capacite ; j++){
			if (j == (capacite - 1)){
				cout << remplissage3IF[i][j] << endl;
			}
			else{
				cout << remplissage3IF[i][j] << " ";
			}
		}
	}
	
	cout << "tableau du bus de 4IF" << endl;
	for (int i = 0; i < nb4IF; i++){
		for (int j = 0; j < capacite ; j++){
			if (j == (capacite - 1)){
				cout << remplissage4IF[i][j] << endl;
			}
			else{
				cout << remplissage4IF[i][j] << " ";
			}
		}
	}
	#endif
	} //	affichage du tableau pour debug (def MAP)
	
	return 0;
}

