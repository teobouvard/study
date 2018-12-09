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
//-----------------------------------------------------------------------------
/* Rôle de la classe <Collection>

La classe <Collection> correspond à la structure de données utilisée pour ce TP.
Elle est constituée d'un tableau dynamique de pointeurs sur des trajets. La taille
actuelle du tableau dynamique est stockée dans l'attribut nbElementsMax, et le
nombre de trajets qu'elle contient dans l'attribut nbElements.
*/

//------------------------------------------------------------------------

class Collection
{
	//----------------------------------------------------------------- PUBLIC
public:
	//----------------------------------------------------- Méthodes publiques

	/*
	- procédure permettant de réallouer le tableau de trajets lorsqu'il est plein
	- la taille du tableau dynamique est doublée à chaque appel à cette procédure
	- la taille initiale est fixée par la constante TAILLE_INITIALE
	*/
	void Resize();

	//affiche chaque trajet contenu dans le tableau dynamique
	void AfficherCollection() const;

	//ajoute le pointeur sur le trajet t au tableau dynamique
	void Ajouter(Trajet * t);

	/*
	- retourne le pointeur sur le trajet contenu à l'indice i du tableau dynamique
	- utilisé dans le constructeur et l'affichage de TrajetCompose ainsi que les
	  méthodes de recherche du Catalogue
	*/
	Trajet* getElement(int i) const;

	/*
	- retourne le nombre d'élements contenus du tableau dynamique
	- utilisé dans le constructeur de TrajetCompose, les recherches et l'affichage
		du Catalogue
	*/
	int getNbElements() const;

	/*
	- crée une collection identique à celle sur laquelle elle est appellée
	- utilisé dans la recherche avancée du Catalogue et le clonage d'un
		TrajetCompose
	*/
	Collection* cloneCollection() const;

	//-------------------------------------------- Constructeurs - destructeur

	//construction d'une collection vide de taille TAILLE_INITIALE
	Collection ();

	/*
	après avoir détruit tous les pointeurs de Trajet du tableau dynamique, celui
	ci est détruit aussi
	*/
	virtual ~Collection ( );
	//------------------------------------------------------------------ PRIVE

protected:
	//----------------------------------------------------- Méthodes protégées
	//----------------------------------------------------- Attributs protégés
	int nbElements;
	int nbElementsMax;
	Trajet** elements;
};

#endif // Collection_H
