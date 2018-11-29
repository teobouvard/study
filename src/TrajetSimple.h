/*************************************************************************
                           TrajetSimple  -  description
                             -------------------
    début                : $DATE$
    copyright            : (C) $YEAR$ par $AUTHOR$
    e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <TrajetSimple> (fichier TrajetSimple.h) ----------------
#if ! defined ( TrajetSimple_H )
#define TrajetSimple_H

//--------------------------------------------------- Interfaces utilisées

//------------------------------------------------------------- Constantes

//------------------------------------------------------------------ Types

//------------------------------------------------------------------------
// Rôle de la classe <TrajetSimple>
//
//
//------------------------------------------------------------------------

class TrajetSimple : public Trajet
{
//----------------------------------------------------------------- PUBLIC

public:
//----------------------------------------------------- Méthodes publiques
    // type Méthode ( liste des paramètres );
    // Mode d'emploi :
    //
    // Contrat :
    //


//------------------------------------------------- Surcharge d'opérateurs
    //TrajetSimple & operator = ( const TrajetSimple & unTrajetSimple );

    
//-------------------------------------------- Constructeurs - destructeur
    TrajetSimple ( const TrajetSimple & unTrajetSimple );


    TrajetSimple ( const char * villeDep, const char * villeArr, const char * modeTrans );
    
    void Afficher();


    virtual ~TrajetSimple ( );


//------------------------------------------------------------------ PRIVE

protected:
//----------------------------------------------------- Méthodes protégées
char * modeTransport;
//----------------------------------------------------- Attributs protégés

};

//-------------------------------- Autres définitions dépendantes de <TrajetSimple>

#endif // TrajetSimple_H

