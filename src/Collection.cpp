/*************************************************************************
Collection  -  description
-------------------
début                : $DATE$
copyright            : (C) $YEAR$ par $AUTHOR$
e-mail               : $EMAIL$
*************************************************************************/

//---------- Réalisation de la classe <Collection> (fichier Collection.cpp) ------------

//---------------------------------------------------------------- INCLUDE

//-------------------------------------------------------- Include système
#include <iostream>
using namespace std;

//------------------------------------------------------ Include personnel
#include "Trajet.h"
#include "Collection.h"

//------------------------------------------------------------- Constantes

const int TAILLE_INITIALE = 10; //doit être différent de 0 pour Resize()

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
void Collection::Resize()
{
	int newSize = 2*tailleTableau;
	Trajet** resized_arr = new Trajet* [newSize]; //deleted dans le destructeur de Trajet ?

	for(int i = 0; i < tailleTableau; i++){
		resized_arr[i] = elements[i];
	}
	tailleTableau = newSize;

	delete[] elements;
	elements = resized_arr;

}

void Collection::Afficher() const
{
	cout << "La collection comporte " << nbElements << " elements" << endl << endl;

	for (int i = 0; i < nbElements; i++)
		elements[i]->Afficher();
}

void Collection::Ajouter(Trajet * t)
{
	//ajout d'un nouveau trajet si la taille le permet
	if (nbElements < tailleTableau){
		elements[nbElements++] = t;
	}
	//sinon redimensionnement du tableau dynamique
	else{
		Collection::Resize();
		elements[nbElements++] = t;

		#ifdef MAP2
		cout << "Nouveau nbElements " << nbElements << endl;
		cout << "Nouveau TailleTableau " << tailleTableau << endl;
		#endif
	}
}
//------------------------------------------------- Surcharge d'opérateurs
//Collection & Collection::operator = ( const Collection & unCollection )

//
//----- Fin de operator =

//-------------------------------------------- Constructeurs - destructeur
Collection::Collection ( const Collection & unCollection )
{
	nbElements = unCollection.nbElements;

	for (int i = 0; i < nbElements; i++){
		*elements[i] = *unCollection.elements[i];
	}

	#ifdef MAP
	cout << "Appel au constructeur de copie de <Collection>" << endl;
	#endif
} //----- Fin de Collection (constructeur de copie)


Collection::Collection ()
{
	nbElements = 0;
	elements = new Trajet* [TAILLE_INITIALE];
	tailleTableau = TAILLE_INITIALE;

	#ifdef MAP
	cout << "Appel au constructeur de <Collection>" << endl;
	#endif
} //----- Fin de Collection


Collection::~Collection ()
{
	for (int i = 0; i < nbElements; i++){
		delete elements[i]; //ce delete cause un segmentation fault du à un invalid write and free
	}

	delete [] elements;

	#ifdef MAP
	cout << "Appel au destructeur de <Collection>" << endl;
	#endif

}//----- Fin de ~Collection

//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées
