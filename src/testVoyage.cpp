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
#include <cstring>
//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"
#include "TrajetSimple.h"
#include "TrajetCompose.h"
#include "Catalogue.h"

///////////////////////////////////////////////////////////////////  PRIVE
//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

//---------------------------------------------------- Variables statiques

//------------------------------------------------------ Fonctions privées

//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques

void testCopieCollection(){
	TrajetSimple* TS1 = new TrajetSimple("Lyon","Bordeaux","Train");
	TrajetSimple* TS2 = new TrajetSimple("Bordeaux","Brest","Voiture");

	Collection* C1 = new Collection;
	C1->Ajouter(TS1);
	C1->Ajouter(TS2);

	Collection* C2 = new Collection(*C1);
	delete C1;

	C2->AfficherCollection();
	delete C2;
}

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

int main()
{

	//testTrajetCompose();
	testCatalogue();

	return 0;
}
