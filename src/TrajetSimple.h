/*************************************************************************
TrajetSimple  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <TrajetSimple> (fichier TrajetSimple.h) ---
#ifndef TrajetSimple_H
#define TrajetSimple_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
//------------------------------------------------------------------------
// Rôle de la classe <TrajetSimple>
/*
La classe <TrajetSimple> est une sorte de Trajet qui comporte un mode de Transport

*/

class TrajetSimple : public Trajet
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques
  virtual void Afficher() const;
  virtual Trajet* clone() const;

  //-------------------------------------------- Constructeurs - destructeur

  /*construit un Trajet Simple grâce à une ville de départ, une ville d'arrivée
  et un mode de transport*/
  TrajetSimple ( const char * villeDep, const char * villeArr, const char * modeTrans );

  //détruit le pointeur attribut de la classe
  virtual ~TrajetSimple ( );

  //------------------------------------------------------------------ PRIVE
protected:
  //----------------------------------------------------- Attributs protégés
  char * modeTransport;
};

#endif // TrajetSimple_H
