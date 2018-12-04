/*************************************************************************
                           Catalogue  -  description
                             -------------------
    début                : $DATE$
    copyright            : (C) $YEAR$ par $AUTHOR$
    e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <Catalogue> (fichier Catalogue.h) ----------------
#ifndef Catalogue_H
#define Catalogue_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
#include "Collection.h"

// Rôle de la classe <Catalogue>

//------------------------------------------------------------------------
class Catalogue
{
//----------------------------------------------------------------- PUBLIC
public:
//----------------------------------------------------- Méthodes publiques

void AjouterTrajet(Trajet* unTrajet);
void AfficherCatalogue();
void RechercherTrajet(char* depart, char* arrivee) const;
void RechercherTrajetAvance(char* depart, char* arrivee, int profondeurRecherche, Collection* c) const;

//procédure utilisé pour l'affichage du nombre d'option dans les recherches
void RaZ_nbOption();


//-------------------------------------------- Constructeurs - destructeur

    Catalogue ( );
    virtual ~Catalogue ( );

//------------------------------------------------------------------ PRIVE
protected:
//----------------------------------------------------- Attributs protégés
  Collection* collectionTrajets;
  int nbTrajets;
};

#endif // Catalogue_H
