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

	C2->Afficher();
	delete C2;
}

void testTrajetCompose(){
	TrajetSimple* TS2 = new TrajetSimple("Bordeaux","Brest","Voiture");
	TrajetSimple* TS3 = new TrajetSimple("Brest","Lille","Train");

	Collection* C1 = new Collection;
	C1->Ajouter(TS2);
	C1->Ajouter(TS3);

	TrajetCompose* TC1 = new TrajetCompose(C1);
	TC1->Afficher();

	delete C1;
	delete TC1;
}


int main()
{

	testTrajetCompose();

	return 0;
}
