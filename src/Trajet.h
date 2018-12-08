/*************************************************************************
Trajet  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <Trajet> (fichier Trajet.h) ----------------
#ifndef Trajet_H
#define Trajet_H
//--------------------------------------------------- Interfaces utilisées
/*	Rôle de la classe <Trajet>

La classe <Trajet> est une classe abstraite qui représente un Trajet comme un
objet ayant une ville de départ et une ville d'arrivée.
*/

class Trajet
{
	//----------------------------------------------------------------- PUBLIC
public:
	//----------------------------------------------------- Méthodes publiques

	//méthode virtuelle pure qui permet d'afficher les attributs d'un trajet
	virtual void Afficher() const = 0;

	/*
	- méthode qui renvoie la ville de départ (depart_arrivee = 0) ou d'arrivée
	(depart_arrivee = 1) d'un trajet
	- utilisée dans les méthodes de recherche du Catalogue et l'affichage des
	trajets composés
	*/
	char* getVille(int depart_arrivee) const;

	/*
	- méthode virtuelle pure qui permet de cloner un trajet
	- utilisée dans les méthodes de recherche du Catalogue, le clonage d'une Collection
	et implémentée dans TrajetSimple et TrajetCompose
	*/
	virtual Trajet* clone() const = 0;

	//-------------------------------------------- Constructeurs - destructeur

	//construction d'un trajet grâce à une ville de départ et une ville d'arrivée
	Trajet (const char * villeDep, const char * villeArr);

	//destruction des deux pointeurs attributs de la classe
	virtual ~Trajet ( );

	//------------------------------------------------------------------ PRIVE
protected:
	//----------------------------------------------------- Attributs protégés
	char * villeDepart;
	char * villeArrivee;
};

#endif // Trajet_H
