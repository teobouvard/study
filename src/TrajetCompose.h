/*************************************************************************
TrajetCompose  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <TrajetCompose> (fichier TrajetCompose.h) ----------------
#ifndef TrajetCompose_H
#define TrajetCompose_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
#include "Collection.h"
//------------------------------------------------------------------------
// Rôle de la classe <TrajetCompose>
/*
La classe <TrajetSimple> est une sorte de Trajet qui comporte un nombre d'escales
et une collection de trajets qui correspond aux différentes escales
*/

class TrajetCompose : public Trajet
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques

  /*
  redéfinition de l'affichage d'un Trajet, permettant d'afficher le nombre
  d'escales et les différentes escales
  */
  virtual void Afficher() const;

  //redéfinition du clonage d'un trajet
  virtual Trajet* clone() const;
  //-------------------------------------------- Constructeurs - destructeur

  //construction d'un trajet composé grâce à une collection de trajets
  TrajetCompose (Collection* c);

  //destruction du pointeur attribut de la classe
  virtual ~TrajetCompose ( );

  //------------------------------------------------------------------ PRIVE
protected:
  //----------------------------------------------------- Attributs protégés
  Collection* escales;
  int nombreEscales;
};

#endif // TrajetCompose_H
