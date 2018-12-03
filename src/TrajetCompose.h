/*************************************************************************
TrajetCompose  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Interface de la classe <TrajetCompose> (fichier TrajetCompose.h) ----------------
#ifndef TrajetCompose_H
#define TrajetCompose_H

//------------------------------------------------------------------------
class TrajetCompose : public Trajet
{
  //----------------------------------------------------------------- PUBLIC
public:
  //----------------------------------------------------- Méthodes publiques
  virtual void Afficher() const;

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
