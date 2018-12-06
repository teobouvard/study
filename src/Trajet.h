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

class Trajet
{
	//----------------------------------------------------------------- PUBLIC

public:
	//----------------------------------------------------- Méthodes publiques
	virtual void Afficher() const = 0;
	char* getVille(int depart_arrivee) const;
	virtual Trajet* clone() const = 0;

	//-------------------------------------------- Constructeurs - destructeur
	Trajet (const char * villeDep, const char * villeArr);
	virtual ~Trajet ( );

	//------------------------------------------------------------------ PRIVE
protected:
	//----------------------------------------------------- Attributs protégés
	char * villeDepart;
	char * villeArrivee;
};

#endif // Trajet_H
