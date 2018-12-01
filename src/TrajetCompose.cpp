/*************************************************************************
TrajetCompose  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation de la classe <TrajetCompose> (fichier TrajetCompose.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;

//------------------------------------------------------ Include personnel
//attention à l'ordre des include !
#include "Trajet.h"
#include "Collection.h"
#include "TrajetCompose.h"


//------------------------------------------------------------- Constantes

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void TrajetCompose::Afficher() const
{
	cout << "Trajet Compose" << endl;
  cout << "Ville de départ : " << escales->elements[0]->villeDepart << "  ";
  cout << "Ville d'arrivée : " << escales->elements[nombreEscales-1]->villeArrivee << endl;
  cout << "Escales : " << endl;

  for (int i = 0; i < nombreEscales; i++){
    escales->elements[i]->Afficher();
  }
}


//------------------------------------------------- Surcharge d'opérateurs
/*TrajetCompose & TrajetCompose::operator = ( const TrajetCompose & unTrajetCompose )
// Algorithme :
//
{
} //----- Fin de operator =*/


//-------------------------------------------- Constructeurs - destructeur
/*TrajetCompose::TrajetCompose ( const TrajetCompose & unTrajetCompose ) : Trajet(unTrajetCompose)
{
nombreTrajets = unTrajetCompose.nombreTrajets;
lesTrajets = unTrajetCompose.lesTrajets;
#ifdef MAP
cout << "Appel au constructeur de copie de <TrajetCompose>" << endl;
#endif
} //----- Fin de TrajetCompose (constructeur de copie)

*/
TrajetCompose::TrajetCompose (Collection* c)
: Trajet(c->elements[0]->villeDepart,c->elements[c->nbElements-1]->villeArrivee)
{
  nombreEscales = c->nbElements;
	escales = new Collection;
  for (int i = 0; i < nombreEscales; i++){
    escales->elements[i] = c->elements[i];
  }
  #ifdef MAP
  cout << "Appel au constructeur de <TrajetCompose>" << endl;
  #endif
} //----- Fin de TrajetCompose


TrajetCompose::~TrajetCompose ( )
{
	delete escales;
  #ifdef MAP
  cout << "Appel au destructeur de <TrajetCompose>" << endl;
  #endif
  } //----- Fin de ~TrajetCompose


  //------------------------------------------------------------------ PRIVE

  //----------------------------------------------------- Méthodes protégées
