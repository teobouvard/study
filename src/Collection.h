/*************************************************************************
Collection  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <Collection> (fichier Collection.h) ----------------
#ifndef Collection_H
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

//afin d'accéder aux attributs d'une collection dans le constructeur de TrajetCompose
friend class TrajetCompose;

public:
	//----------------------------------------------------- Méthodes publiques

	//procédure permettant de réallouer le tableau de trajets lorsqu'il est plein
	//la taille du tableau est doublée à chaque appel à cette procédure
	void Resize();

	void AfficherCollection() const;
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
