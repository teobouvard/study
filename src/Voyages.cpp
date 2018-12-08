/*************************************************************************
testVoyage  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Réalisation du module <testVoyage> (fichier testVoyage.cpp) ---------------

/////////////////////////////////////////////////////////////////  INCLUDE
//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;
//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"
#include "TrajetSimple.h"
#include "TrajetCompose.h"
#include "Catalogue.h"

#define BIGNUMBER 99999999

///////////////////////////////////////////////////////////////////  PRIVE
//------------------------------------------------------------- Constantes

const int TAILLE_MAX_STRING = 20;

//------------------------------------------------------------------ Types

//---------------------------------------------------- Variables statiques

//------------------------------------------------------ Fonctions privées

//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques

void init(){
	cout << "Bienvenue dans le Gestionnaire de Trajets" << endl << endl;
}

void annonce(){
	cout << "Afficher le catalogue : 0 | Ajouter un trajet : 1 | Rechercher un trajet : 2 | Rechercher un trajet (avancé) : 3 | Quitter cette app agile : 9" << endl << endl;
}

//retourne un pointeur sur un trajet simple créé lors de la fonction
Trajet* CreerTrajetSimple(){
	char* ville1 = new char[TAILLE_MAX_STRING];
	char* ville2 = new char[TAILLE_MAX_STRING];
	char* mdTransport = new char[TAILLE_MAX_STRING];

	cout << "Ville de départ ?"<< endl;
	cin >> ville1;
	cout << "Ville d'arrivée ?"<< endl;
	cin >> ville2;
	cout << "Mode de Transport ?"<< endl;
	cin >> mdTransport;

	Trajet* trajet = new TrajetSimple(ville1, ville2, mdTransport);

	delete [] ville1;
	delete [] ville2;
	delete [] mdTransport;

	return trajet;
}

//fonction récursive qui ajoute des trajets simple à la collection de base
//si un trajet composé contient un trajet composé, la fonction s'auto-appelle
//option : 0-> trajet simple 		1->trajet composé
void ajoutCollection(Collection * c, int option){
	if(option == 0){
		c->Ajouter(CreerTrajetSimple());
	} else if(option == 1){
		int nEscales;
		cout << "Nombre d'escales?" << endl;
		while(!(cin >> nEscales) || nEscales < 2){
			cin.clear();
			cin.ignore(BIGNUMBER, '\n');
			cout << "Entrée invalide. Réessayez" << endl;
		}
		Collection* collectionTrajets = new Collection;
		for(int i = 0 ; i < nEscales; i++){
			cout << "Trajet simple : 0 | Trajet composé 1" << endl;
			int choix;
			while(!(cin >> choix)){
				cin.clear();
				cin.ignore(BIGNUMBER, '\n');
				cout << "Entrée invalide. Réessayez" << endl;
			}
			ajoutCollection(collectionTrajets,choix);
		}
		TrajetCompose* trajet = new TrajetCompose(collectionTrajets);
		c->Ajouter(trajet);
	}
	else if(option == 2){
		cout << endl;
	}
}


int main()
{
	int lecture;
	char* ville1 = new char[TAILLE_MAX_STRING];
	char* ville2 = new char[TAILLE_MAX_STRING];
	Catalogue catalogue;
	Collection c;

	init();
	annonce();

	while(!(cin >> lecture)){
		cin.clear();
		cin.ignore(BIGNUMBER, '\n');
		cout << "Entrée invalide. Réessayez" << endl;
	}

	while(lecture != 9){

		switch (lecture){

			case 0 :
			catalogue.AfficherCatalogue();
			break;

			case 1 :
			cout << endl << "Trajet simple : 0 | Trajet composé : 1 | Annuler : 2" << endl;
			while(!(cin >> lecture)){
				cin.clear();
				cin.ignore(BIGNUMBER, '\n');
				cout << "Entrée invalide. Réessayez" << endl;
			}
			catalogue.addOneTrajet();
			ajoutCollection(catalogue.getCollection(),lecture);
			break;

			case 2 :
			cout << "Ville de départ ?"<< endl;
			cin >> ville1;
			cout << "Ville d'arrivée ?"<< endl;
			cin >> ville2;
			cout << endl;
			catalogue.RechercherTrajet(ville1, ville2);
			catalogue.RaZ_nbOption();
			break;

			case 3 :
			cout << "Ville de départ ?"<< endl;
			cin >> ville1;
			cout << "Ville d'arrivée ?"<< endl;
			cin >> ville2;
			cout << endl;
			catalogue.RechercherTrajetAvance(ville1, ville2,0,&c);
			catalogue.RaZ_nbOption();
			break;

			default :
			cout << "Entrée invalide. Réessayez" << endl;
			break;

		}

		annonce();

		while(!(cin >> lecture)){
			cin.clear();
			cin.ignore(BIGNUMBER, '\n');
			cout << "Entrée invalide. Réessayez" << endl;
		}
	}

	cout << "Au revoir !" << endl << endl;

	delete [] ville1;
	delete [] ville2;

	return 0;
}
