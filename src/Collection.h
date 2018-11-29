/*************************************************************************
Collection  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <Collection> (fichier Collection.h) ----------------
#if ! defined ( Collection_H )
#define Collection_H

//--------------------------------------------------- Interfaces utilisées

//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

//------------------------------------------------------------------------
// Rôle de la classe <Collection>
//
//
//------------------------------------------------------------------------

class Collection
{
	//----------------------------------------------------------------- PUBLIC

public:
	//----------------------------------------------------- Méthodes publiques

	//procédure permettant de réallouer le tableau de trajets lorsqu'il est plein
	//la taille du tableau est doublée à chaque appel à cette procédure
	//arr : tableau de trajets 		n : taille actuelle du tableau
	void Resize();

	void Afficher() const;
	void Ajouter(Trajet * t);

	//------------------------------------------------- Surcharge d'opérateurs
	//Collection & operator = ( const Collection & unCollection );

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
