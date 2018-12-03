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
