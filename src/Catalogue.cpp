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
//------------------------------------------------------------- Constantes
const int PROFONDEUR_MAXIMALE = 5;
int profondeurRecherche = 0;
//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Catalogue::RechercherTrajet(char* depart, char* arrivee) const{
  int compteur = 0;
  for (int i = 0; i < nbTrajets; i++){
    Trajet* trajetEvalue = collectionTrajets->getElement(i);
    if(strcmp(depart,trajetEvalue->getVille(0))==0 && strcmp(arrivee,trajetEvalue->getVille(1))==0){
      compteur++;
      trajetEvalue->Afficher();
    }
  }
  cout << compteur << " trajets trouvés." << endl << endl;
}

void Catalogue::RechercherTrajetAvance(char* depart, char* arrivee, int profondeurRecherche,Collection* c) const{

  //while(profondeurRecherche < PROFONDEUR_MAXIMALE){

  for (int i = 0; i < nbTrajets; i++){
    Collection* c1 = c->cloneCollection();
    Trajet* trajetEvalue = collectionTrajets->getElement(i)->clone();
    c1->Ajouter(trajetEvalue);
    if(strcmp(depart,trajetEvalue->getVille(0))==0){
      if(strcmp(arrivee,trajetEvalue->getVille(1))==0){
        c1->AfficherCollection();

      }
      else{
        if(profondeurRecherche<=5){
          Catalogue::RechercherTrajetAvance(trajetEvalue->getVille(1),arrivee,++profondeurRecherche,c1);
        }
      }
    }
    delete c1;
  }
}
//}

void Catalogue::AjouterTrajet(Trajet* unTrajet)
{
  collectionTrajets->Ajouter(unTrajet);
  nbTrajets++;
}

void Catalogue::AfficherCatalogue(){
  cout << "Le catalogue comporte " << nbTrajets << " trajets." << endl << endl;
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
