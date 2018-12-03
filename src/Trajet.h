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
	//afin d'afficher les villes du trajet dans l'affichage d'un TrajetCompose
	//friend class TrajetCompose;

public:
	//----------------------------------------------------- Méthodes publiques
	virtual void Afficher() const = 0;
	char* getVille(int depart_arrivee) const;

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
