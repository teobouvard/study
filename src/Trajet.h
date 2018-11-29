/*************************************************************************
                           Trajet  -  description
                             -------------------
    début                : $DATE$
    copyright            : (C) $YEAR$ par $AUTHOR$
    e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <Trajet> (fichier Trajet.h) ----------------
#if ! defined ( Trajet_H )
#define Trajet_H

//--------------------------------------------------- Interfaces utilisées
#include <cstring>
using namespace std;

//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

//------------------------------------------------------------------------
// Rôle de la classe <Trajet>
//
//
//------------------------------------------------------------------------

class Trajet
{
//----------------------------------------------------------------- PUBLIC

public:
//----------------------------------------------------- Méthodes publiques
    // type Méthode ( liste des paramètres );
    // Mode d'emploi :
    //
    // Contrat :
    //
	
	virtual void Afficher();


//------------------------------------------------- Surcharge d'opérateurs
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

//-------------------------------- Autres définitions dépendantes de <Trajet>

#endif // Trajet_H

