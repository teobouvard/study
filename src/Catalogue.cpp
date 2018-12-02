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

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Catalogue::RechercherTrajet(char* depart, char* arrivee){
  for (int i = 0; i < nbTrajets; i++){
    Trajet* trajetEvalue = collectionTrajets->elements[i];
    if(strcmp(depart,trajetEvalue->villeDepart)==0 && strcmp(arrivee,trajetEvalue->villeArrivee)==0){
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

//------------------------------------------------- Surcharge d'opérateurs
/*Catalogue & Catalogue::operator = ( const Catalogue & unCatalogue )
// Algorithme :
//
{
} //----- Fin de operator =*/


//-------------------------------------------- Constructeurs - destructeur
/*Catalogue::Catalogue ( const Catalogue & unCatalogue )
{
  #ifdef MAP
  cout << "Appel au constructeur de copie de <Catalogue>" << endl;
  #endif
}*/ //----- Fin de Catalogue (constructeur de copie)


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


//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées
