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
//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Trajet::Afficher() const
{
	cout << villeDepart << " -> " << villeArrivee;
}

char* Trajet::getVille(int depart_arrivee) const
{
	if(depart_arrivee == 0){
		return villeDepart;
	}
	else if (depart_arrivee == 1){
		return villeArrivee;
	}
	else{
		cout << "Erreur du getVille" << endl;
		return nullptr;
	}
}

//-------------------------------------------- Constructeurs - destructeur
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
