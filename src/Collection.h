/*************************************************************************
Collection  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <Collection> (fichier Collection.h) ----------------
#ifndef Collection_H
#define Collection_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
#include "Collection.h"

// Rôle de la classe <Collection>


//------------------------------------------------------------------------

class Collection
{
	//----------------------------------------------------------------- PUBLIC
public:
	//----------------------------------------------------- Méthodes publiques

	//procédure permettant de réallouer le tableau de trajets lorsqu'il est plein
	//la taille du tableau est doublée à chaque appel à cette procédure
	void Resize();

	void AfficherCollection() const;
	void Ajouter(Trajet * t);
	Trajet* getElement(int i) const;
	int getNbElements() const;

	//fonction qui crée une collection identique à celle sur laquelle elle est appellée
	Collection* cloneCollection() const;

	//-------------------------------------------- Constructeurs - destructeur
	Collection ( const Collection & unCollection );
	Collection ();
	virtual ~Collection ( );
	//------------------------------------------------------------------ PRIVE

protected:
	//----------------------------------------------------- Méthodes protégées

	//----------------------------------------------------- Attributs protégés
	int nbElements;
	int tailleTableau;
	Trajet** elements;
};

#endif // Collection_H
