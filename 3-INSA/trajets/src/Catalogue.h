/*************************************************************************
Catalogue  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <Catalogue> (fichier Catalogue.h) ----------
#ifndef Catalogue_H
#define Catalogue_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
#include "Collection.h"
//-----------------------------------------------------------------------------
/* Rôle de la classe <Catalogue>

La classe <Catalogue> représente un catalogue de trajets, et se base sur la
structure de données définie par la classe Collection.
*/

class Catalogue
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques

  /*
  affiche le nombre de trajets présents dans le catalogue, ainsi que les différents
  trajets qui le composent
  */
  void AfficherCatalogue();

  /*
  recherche dans le catalogue tous les trajets qui ont comme ville de départ le
  paramètre depart et comme ville d'arrivée le paramètre arrivee
  */
  void RechercherTrajet(char* depart, char* arrivee) const;

  /*
  - recherche dans le catalogue tous les trajets qui ont comme ville de départ le
  paramètre depart et comme ville d'arrivée le paramètre arrivee, en composant
  les trajets
  - cette méthode est récursive, le paramètre profondeurRecherche permet de savoir
  à quel étage de la recherche on se situe
  - le paramètre c est une collection de trajets qui correspond à une des compositions
  de trajets permettant d'arriver à destination
  */
  void RechercherTrajetAvance(char* depart, char* arrivee, int profondeurRecherche, Collection* c) const;

  /*
  - retourne la collection de trajets du Catalogue
  - utilisé dans la classe de test Voyages
  */
  Collection* getCollection();

  //procédure utilisée pour réinitialiser l'affichage du nombre d'options après une recherche
  void RaZ_nbOption();

  //-------------------------------------------- Constructeurs - destructeur

  //construction d'un catalogue vide
  Catalogue();

  //destruction du pointeur sur collection, seul attribut de la classe
  virtual ~Catalogue();

  //------------------------------------------------------------------ PRIVE
protected:
  //----------------------------------------------------- Attributs protégés
  Collection* collectionTrajets;
};

#endif // Catalogue_H
