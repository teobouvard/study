/*************************************************************************
Trajet  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <Trajet> (fichier Trajet.h) ----------------
#ifndef Trajet_H
#define Trajet_H

//--------------------------------------------------- Interfaces utilisées
#include <cstring>
using namespace std;

//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

class Trajet
{
	//----------------------------------------------------------------- PUBLIC
	//afin d'afficher les villes du trajet dans l'affichage d'un TrajetCompose
	friend class TrajetCompose;

	//afin d'accéder aux villes lors du constructeur de copie de Collection
	//a résoudre
	friend class Collection;

public:
	//----------------------------------------------------- Méthodes publiques
	//pas virtuelle pure sinon on ne peut pas instancier Trajet dans un constructeur
	virtual void Afficher() const; //= 0;
	//-------------------------------------------------- Surcharge d'opérateurs
	//Trajet & operator = ( const Trajet & unTrajet );

	//-------------------------------------------- Constructeurs - destructeur
	Trajet ( const Trajet & unTrajet );
	Trajet (const char * villeDep, const char * villeArr);
	virtual ~Trajet ( );

	//------------------------------------------------------------------ PRIVE

protected:
	//----------------------------------------------------- Méthodes protégées

	//----------------------------------------------------- Attributs protégés
	char * villeDepart;
	char * villeArrivee;
};

#endif // Trajet_H
