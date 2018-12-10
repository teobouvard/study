/*************************************************************************
TrajetSimple  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Réalisation de la classe <TrajetSimple> (fichier TrajetSimple.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
#include <cstring>
using namespace std;
//------------------------------------------------------ Include personnel
#include "TrajetSimple.h"
//----------------------------------------------------------------- PUBLIC
//----------------------------------------------------- Méthodes publiques

void TrajetSimple::Afficher() const
{
	cout << "Trajet Simple : ";
	Trajet::Afficher();
	cout << " en " << modeTransport << endl;
}

Trajet* TrajetSimple::clone() const{
	return new TrajetSimple(villeDepart,villeArrivee,modeTransport);
}

//-------------------------------------------- Constructeurs - destructeur

TrajetSimple::TrajetSimple (const char * villeDep, const char * villeArr, const char * modeTrans ) : Trajet(villeDep, villeArr)
{
	modeTransport = new char[strlen(modeTrans) + 1];
	strcpy(modeTransport,modeTrans);
	#ifdef MAP
	cout << "Appel au constructeur de <TrajetSimple>" << endl;
	#endif
} //----- Fin de TrajetSimple


TrajetSimple::~TrajetSimple ( )
{
	delete [] modeTransport;
	#ifdef MAP
	cout << "Appel au destructeur de <TrajetSimple>"<< endl;
	#endif
} //----- Fin de ~TrajetSimple
