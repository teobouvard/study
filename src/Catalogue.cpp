/*************************************************************************
Catalogue  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation de la classe <Catalogue> (fichier Catalogue.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;
//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"
#include "Catalogue.h"
//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Catalogue::RechercherTrajet(char* depart, char* arrivee) const{
  for (int i = 0; i < nbTrajets; i++){
    Trajet* trajetEvalue = collectionTrajets->getElement(i);
    if(strcmp(depart,trajetEvalue->getVille(0))==0 && strcmp(arrivee,trajetEvalue->getVille(1))==0){
      trajetEvalue->Afficher();
    }
  }
}

void Catalogue::AjouterTrajet(Trajet* unTrajet)
{
  collectionTrajets->Ajouter(unTrajet);
  nbTrajets++;
}

void Catalogue::AfficherCatalogue(){
  cout << "Le catalogue comporte " << nbTrajets << " trajets" << endl << endl;
  collectionTrajets->Collection::AfficherCollection();
}

//-------------------------------------------- Constructeurs - destructeur

Catalogue::Catalogue ()
{
  nbTrajets = 0;
  collectionTrajets = new Collection;
  #ifdef MAP
  cout << "Appel au constructeur de <Catalogue>" << endl;
  #endif
} //----- Fin de Catalogue


Catalogue::~Catalogue ( )
{
  delete collectionTrajets;
  #ifdef MAP
  cout << "Appel au destructeur de <Catalogue>" << endl;
  #endif
} //----- Fin de ~Catalogue
