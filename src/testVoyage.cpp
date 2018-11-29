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

//------------------------------------------------------ Include personnel
#include <iostream>
#include <cstring>
#include "Trajet.h"
#include "TrajetSimple.h"
#include "TrajetCompose.h"
#include "Collection.h"

///////////////////////////////////////////////////////////////////  PRIVE
//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

//---------------------------------------------------- Variables statiques

//------------------------------------------------------ Fonctions privées
//static type nom ( liste de paramètres )
// Mode d'emploi :
//
// Contrat :
//
// Algorithme :
//
//{
//} //----- fin de nom

//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques
int main()
{
	
	TrajetSimple* TS1 = new TrajetSimple("Lyon","Bordeaux","Train");
	TrajetSimple* TS2 = new TrajetSimple("Bordeaux","Brest","Voiture");
	TrajetSimple* TS3 = new TrajetSimple("Brest","Lille","Train");
	
	Collection C1;
	C1.Ajouter(TS1);
	C1.Ajouter(TS2);
	C1.Ajouter(TS3);
	C1.Afficher();

	//TrajetCompose TC2("Paris","Marseille",1,tc2);
	//TS1.Afficher();
	
	return 0;
}

