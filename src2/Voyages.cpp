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

#define BIGNUMBER 999999

///////////////////////////////////////////////////////////////////  PRIVE
//------------------------------------------------------------- Constantes

const int TAILLE_MAX_STRING = 20;

//------------------------------------------------------------------ Types

//---------------------------------------------------- Variables statiques

//------------------------------------------------------ Fonctions privées

//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques

void annonce(){
	cout << "Afficher le catalogue : 0 | Ajouter un trajet : 1 | Rechercher un trajet : 2 | Rechercher un trajet (avancé) : 3 | Quitter cette app agile : 9" << endl << endl;
}

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

void creationTrajet(Collection * c, int option){
	if(option == 1){
		c->Ajouter(CreerTrajetSimple());
	} else if(option == 2){
		int nEscales;
		cout << "Nombre d'escales?" << endl;
		cin >> nEscales;
		Collection* collectionTrajets = new Collection;
		for(int i = 0 ; i < nEscales; i++){
			cout << "Escale n°" << i+1 << endl;
			cout << "Trajet simple : 1 | Trajet composé 2" << endl;
			int choix;
			cin >> choix;
			creationTrajet(collectionTrajets,choix);
		}
		TrajetCompose* trajet = new TrajetCompose(collectionTrajets);
		c->Ajouter(trajet);
	}
	else if(option == 0){
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

	cout << "Bienvenue dans le Gestionnaire de Trajets" << endl << endl;

	annonce();

	cin >> lecture;

	while(lecture != 9){

		switch (lecture){

			case 0 :
			catalogue.AfficherCatalogue();
			break;

			case 1 :
			cout << endl << "Trajet simple : 1 | Trajet composé : 2 | Annuler : 0" << endl;
			cin >> lecture;
			creationTrajet(catalogue.getCollection(),lecture);
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

		cin >> lecture;
	}

	cout << "Au revoir !" << endl << endl;

	delete [] ville1;
	delete [] ville2;

	return 0;
}
