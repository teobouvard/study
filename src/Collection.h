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
