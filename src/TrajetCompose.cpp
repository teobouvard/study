/*************************************************************************
TrajetCompose  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation de la classe <TrajetCompose> (fichier TrajetCompose.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;
//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"
#include "TrajetCompose.h"
//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void TrajetCompose::Afficher() const
{
	cout << "Trajet Composé comportant " << nombreEscales << " escales." << endl;
	cout << "Ville de départ : " << escales->getElement(0)->getVille(0) << "  ";
	cout << "Ville d'arrivée : " << escales->getElement(nombreEscales-1)->getVille(1) << endl;

	for (int i = 0; i < nombreEscales; i++){
		cout << "	";
		escales->getElement(i)->Afficher();
	}
	cout << endl;
}

Trajet* TrajetCompose::clone() const{
	
	Collection* c = escales->cloneCollection();
	return new TrajetCompose(c);
}

//-------------------------------------------- Constructeurs - destructeur

TrajetCompose::TrajetCompose (Collection* c)
: Trajet(c->getElement(0)->getVille(0),c->getElement(c->getNbElements()-1)->getVille(1))
{
	nombreEscales = c->getNbElements();
	escales = c;
	#ifdef MAP
	cout << "Appel au constructeur de <TrajetCompose>" << endl;
	#endif
} //----- Fin de TrajetCompose

TrajetCompose::~TrajetCompose ( )
{
	delete escales;
	#ifdef MAP
	cout << "Appel au destructeur de <TrajetCompose>" << endl;
	#endif
} //----- Fin de ~TrajetCompose


//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées
