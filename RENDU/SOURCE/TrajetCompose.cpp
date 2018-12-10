/*************************************************************************
TrajetCompose  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Réalisation de la classe <TrajetCompose> (fichier TrajetCompose.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;
//------------------------------------------------------ Include personnel
#include "TrajetCompose.h"
//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void TrajetCompose::Afficher() const
{
	cout << "Trajet Composé comportant " << escales->getNbElements() << " escales." << endl;
	cout << "Ville de départ : " << escales->getElement(0)->getVille(0) << "  ";
	cout << "Ville d'arrivée : " << escales->getElement(escales->getNbElements()-1)->getVille(1) << endl;

	for (int i = 0; i < escales->getNbElements(); i++){
		cout << "	";
		escales->getElement(i)->Afficher();
	}
	cout << endl;
}

Trajet* TrajetCompose::clone() const
{
	Collection* c = escales->cloneCollection();
	return new TrajetCompose(c);
}

//-------------------------------------------- Constructeurs - destructeur

TrajetCompose::TrajetCompose (Collection* c)
: Trajet(c->getElement(0)->getVille(0),c->getElement(c->getNbElements()-1)->getVille(1))
{
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
