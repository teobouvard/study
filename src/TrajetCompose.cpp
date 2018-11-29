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
#include "Trajet.h"
#include "TrajetCompose.h"
#include "Collection.h"

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


TrajetCompose::TrajetCompose (char * villeDep, char * villeArr, int nbTrajets) : Trajet(villeDep,villeArr), nombreTrajets(nbTrajets)
// Algorithme :
//
{
#ifdef MAP
    cout << "Appel au constructeur de <TrajetCompose>" << endl;
#endif
} //----- Fin de TrajetCompose
*/ 

TrajetCompose::~TrajetCompose ( )
// Algorithme :
//
{
#ifdef MAP
    cout << "Appel au destructeur de <TrajetCompose>" << endl;
#endif
} //----- Fin de ~TrajetCompose


//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées

