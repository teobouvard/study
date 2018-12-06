/*************************************************************************
                           Catalogue  -  description
                             -------------------
    début                : Novembre 2018
    copyright            : Mathis Guilhin & Téo Bouvard
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
Collection* getCollection();

//procédure utilisé pour réinitialiser l'affichage du nombre d'options dans les recherches
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
