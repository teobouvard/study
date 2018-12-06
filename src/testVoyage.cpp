/*************************************************************************
testVoyage  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation du module <testVoyage> (fichier testVoyage.cpp) ---------------

/////////////////////////////////////////////////////////////////  INCLUDE
//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;
#include <cstring>
//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"
#include "TrajetSimple.h"
#include "TrajetCompose.h"
#include "Catalogue.h"

#define BIGNUMBER 9999999999

///////////////////////////////////////////////////////////////////  PRIVE
//------------------------------------------------------------- Constantes

const int TAILLE_MAX_STRING = 20;

//------------------------------------------------------------------ Types

//---------------------------------------------------- Variables statiques

//------------------------------------------------------ Fonctions privées

//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques

void testTrajetCompose(){
	TrajetSimple* TS2 = new TrajetSimple("Bordeaux","Brest","Voiture");
	TrajetSimple* TS3 = new TrajetSimple("Brest","Lille","Train");

	Collection* C1 = new Collection;
	C1->Ajouter(TS2);
	C1->Ajouter(TS3);

	TrajetCompose* TC1 = new TrajetCompose(C1);

	C1->AfficherCollection();

	delete C1;
	delete TC1;
}

void testEgaliteTrajet(){
	TrajetSimple* TS1 = new TrajetSimple("Lyon","Bordeaux","Train");
	TrajetSimple* TS2 = TS1;

	delete TS1;

	TS2->Afficher();

	delete TS2;
}

void testCatalogue(){
	Catalogue* catalogue = new Catalogue;

	TrajetSimple* TS2 = new TrajetSimple("Bordeaux","Brest","Voiture");
	TrajetSimple* TS3 = new TrajetSimple("Brest","Lille","Train");
	TrajetSimple* TS4 = new TrajetSimple("Bordeaux","Brest","Voiture");
	TrajetSimple* TS5 = new TrajetSimple("Brest","Lille","Train");

	Collection* C1 = new Collection;
	C1->Ajouter(TS4);
	C1->Ajouter(TS5);

	TrajetCompose* TC1 = new TrajetCompose(C1);

	catalogue->AjouterTrajet(TS2);
	catalogue->AjouterTrajet(TS3);
	catalogue->AjouterTrajet(TC1);

	catalogue->AfficherCatalogue();

	delete catalogue;
}

void init(){
	cout << "Bienvenue dans le Gestionnaire de Trajets" << endl << endl;
}

void annonce(){
	cout << "Afficher le catalogue : 0 | Ajouter un trajet : 1 | Rechercher un trajet : 2 | Rechercher un trajet (avancé) : 3 | Quitter cette app agile : 9" << endl << endl;
}

//retourne un pointeur sur un trajet simple créé lors de la fonction
Trajet* CreerTrajet(){
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
		c->Ajouter(CreerTrajet());
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
}

//premier étage de l'ajout d'un trajet au catalogue
//si on crée un trajet composé, appel à ajoutCollection
void ajoutCatalogue(Catalogue * c, int option){
	if(option == 0){
		c->AjouterTrajet(CreerTrajet());
	} else if(option == 1){
		int nEscales;
		cout << "Nombre d'escales?" << endl;
		while(!(cin >> nEscales)){
			cin.clear();
			cin.ignore(BIGNUMBER, '\n');
			cout << "Entrée invalide. Réessayez" << endl;
		}
		Collection* collectionTrajets = new Collection;
		for (int i = 0 ; i < nEscales; i++){
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
		c->AjouterTrajet(trajet);
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
			ajoutCatalogue(&catalogue,lecture);
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
