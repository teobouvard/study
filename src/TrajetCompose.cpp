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
//attention à l'ordre des include
#include "Trajet.h"
#include "Collection.h"
#include "TrajetCompose.h"


//------------------------------------------------------------- Constantes

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
// type TrajetCompose::Méthode ( liste des paramètres )
// Algorithme :
//
//{
//} //----- Fin de Méthode


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
TrajetCompose::TrajetCompose (char * villeDep, char * villeArr, int nbTrajets) : Trajet(villeDep,villeArr)
{
  nombreTrajets = nbTrajets;
  #ifdef MAP
  cout << "Appel au constructeur de <TrajetCompose>" << endl;
  #endif
} //----- Fin de TrajetCompose


TrajetCompose::~TrajetCompose ( )
{/*
  delete [] collectionTrajets;

  #ifdef MAP
  cout << "Appel au destructeur de <TrajetCompose>" << endl;
  #endif
  */} //----- Fin de ~TrajetCompose


  //------------------------------------------------------------------ PRIVE

  //----------------------------------------------------- Méthodes protégées
