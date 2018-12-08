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
//-----------------------------------------------------------------------------
/* Rôle de la classe <Catalogue>

La classe <Catalogue> correspond à la structure de données utilisée pour ce TP.
Elle est constituée d'un tableau dynamique de pointeurs sur des trajets. La taille
actuelle du tableau dynamique est stockée dans l'attribut tailleTableau, et le
nombre de trajets qu'elle contient dans l'attribut nbElements.
*/
//------------------------------------------------------------------------
class Catalogue
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques

  void AfficherCatalogue();
  void RechercherTrajet(char* depart, char* arrivee) const;
  void RechercherTrajetAvance(char* depart, char* arrivee, int profondeurRecherche, Collection* c) const;
  Collection* getCollection();

  //procédure utilisée pour réinitialiser l'affichage du nombre d'options dans les recherches
  void RaZ_nbOption();

  void addOneTrajet();

  //-------------------------------------------- Constructeurs - destructeur

  Catalogue ( );
  virtual ~Catalogue ( );

  //------------------------------------------------------------------ PRIVE
protected:
  //----------------------------------------------------- Attributs protégés
  Collection* collectionTrajets;
};

#endif // Catalogue_H
