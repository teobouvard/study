/*************************************************************************
Voyage  -  description
-------------------
début                : Novembre 2018
copyright            : Mathis Guilhin & Téo Bouvard
*************************************************************************/

//---------- Interface du module <Voyage> (fichier Voyage.h) -------------------
#if ! defined ( Voyage_H )
#define Voyage_H
//------------------------------------------------------------------------
/*  Rôle du module <Voyage>

Ce module implémente l'interface graphique de l'application et la création des
trajets qui vont être ajoutés au Catalogue.
*/

/////////////////////////////////////////////////////////////////  INCLUDE
//--------------------------------------------------- Interfaces utilisées
#include "Trajet.h"
#include "Collection.h"
//////////////////////////////////////////////////////////////////  PUBLIC
//---------------------------------------------------- Fonctions publiques

//affichage du menu en boucle dans le main
static void affichageMenu();

//retourne un pointeur sur un trajet simple créé à l'intérieur de la fonction
static Trajet* creerTrajetSimple();

/*
- fonction récursive qui ajoute des trajets simples à la Collection c
- option : 0-> trajet simple 		1->trajet composé
- si un trajet composé contient un trajet composé, la fonction s'auto-appelle
*/
static void creationTrajet(Collection * c, int option);

int main();

#endif // Voyage_H
