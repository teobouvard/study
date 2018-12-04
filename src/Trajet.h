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
