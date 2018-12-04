/*************************************************************************
TrajetSimple  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <TrajetSimple> (fichier TrajetSimple.h) ----------------
#ifndef TrajetSimple_H
#define TrajetSimple_H
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
