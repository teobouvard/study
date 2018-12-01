/*************************************************************************
Trajet  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation de la classe <Trajet> (fichier Trajet.cpp) ------------

//---------------------------------------------------------------- INCLUDE
//-------------------------------------------------------- Include système
#include <string>
#include <iostream>
using namespace std;

//------------------------------------------------------ Include personnel
#include "Trajet.h"

//------------------------------------------------------------- Constantes

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Trajet::Afficher() const
{
	cout << "Ville de depart : " << villeDepart << endl;
	cout << "Ville d'arrivée : " << villeArrivee << endl;
}

//------------------------------------------------- Surcharge d'opérateurs
/*Trajet & Trajet::operator = ( const Trajet & unTrajet )
// Algorithme :
//
{
} //----- Fin de operator =*/


//-------------------------------------------- Constructeurs - destructeur
Trajet::Trajet ( const Trajet & unTrajet )
{
	villeDepart = new char[strlen(unTrajet.villeDepart) + 1];
	villeArrivee = new char[strlen(unTrajet.villeArrivee) + 1];

	strcpy(villeDepart,unTrajet.villeDepart);
	strcpy(villeArrivee,unTrajet.villeArrivee);
	{
		#ifdef MAP
		cout << "Appel au constructeur de copie de <Trajet>" << endl;
		#endif
	} //----- Fin de Trajet (constructeur de copie)
}


Trajet::Trajet (const char * villeDep, const char * villeArr)
{
	villeDepart = new char[strlen(villeDep) + 1];
	villeArrivee = new char[strlen(villeArr) + 1];

	strcpy(villeDepart,villeDep);
	strcpy(villeArrivee,villeArr);
	
	#ifdef MAP
	cout << "Appel au constructeur de <Trajet>" << endl;
	#endif
	//----- Fin de Trajet
}


Trajet::~Trajet ( )
{
	delete [] villeDepart;
	delete [] villeArrivee;

	#ifdef MAP
	cout << "Appel au destructeur de <Trajet>" << endl;
	#endif
} //----- Fin de ~Trajet


//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées
