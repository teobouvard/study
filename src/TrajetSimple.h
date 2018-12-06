/*************************************************************************
TrajetSimple  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface de la classe <TrajetSimple> (fichier TrajetSimple.h) ----------------
#ifndef TrajetSimple_H
#define TrajetSimple_H
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
//------------------------------------------------------------------------
// Rôle de la classe <TrajetSimple>

class TrajetSimple : public Trajet
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques
  virtual void Afficher() const;
  virtual Trajet* clone() const;

  //-------------------------------------------- Constructeurs - destructeur
  TrajetSimple ( const TrajetSimple & unTrajetSimple );
  TrajetSimple ( const char * villeDep, const char * villeArr, const char * modeTrans );
  virtual ~TrajetSimple ( );

  //------------------------------------------------------------------ PRIVE
protected:
  //----------------------------------------------------- Méthodes protégées
  char * modeTransport;
  //----------------------------------------------------- Attributs protégés

};

#endif // TrajetSimple_H
