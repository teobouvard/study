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
    // type Méthode ( liste des paramètres );
    // Mode d'emploi :
    //
    // Contrat :
    //
	
	void Afficher() const;
	void Ajouter(Trajet * t);
	
//------------------------------------------------- Surcharge d'opérateurs
    Collection & operator = ( const Collection & unCollection );
    // Mode d'emploi :
    //
    // Contrat :
    //


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

//-------------------------------- Autres définitions dépendantes de <Collection>

#endif // Collection_H

