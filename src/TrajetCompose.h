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
class TrajetCompose : public Trajet
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques
  virtual void Afficher() const;
  virtual Trajet* clone() const;
  //-------------------------------------------- Constructeurs - destructeur

  TrajetCompose (Collection* c);
  virtual ~TrajetCompose ( );

  //------------------------------------------------------------------ PRIVE
protected:

  //----------------------------------------------------- Attributs protégés
  Collection* escales;
  int nombreEscales;
  char * villeDepart;
  char * villeArrivee;
};

#endif // TrajetCompose_H
