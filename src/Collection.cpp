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

const int TAILLE_INITIALE = 2;

//----------------------------------------------------------------- PUBLIC

//----------------------------------------------------- Méthodes publiques
// type Collection::Méthode ( liste des paramètres )


void Collection::Afficher() const
{
	cout << "La collection comporte " << nbElements << " elements" << endl << endl;

	for (int i = 0; i < nbElements; i++)
	{
		elements[i]->Afficher();
	}
}


void Collection::Ajouter(Trajet * t)
{
	//ajout d'un nouveau trajet si la taille le permet
	if (nbElements < tailleTableau){
		elements[nbElements++] = t;
	}
	//sinon redimensionnement du tableau dynamique
	else{
		int nouvelleTaille = 2*nbElements;
		//LE CORE DUMP VIENT DE LÀ
		//creation d'un tableau tampon
		Trajet** buf = new Trajet* [nouvelleTaille];
		for (int i = 0; i < nbElements; i++)
		{
			buf[i] = elements[i];
		}
		
		//réattribution au nouveau tableau
		Trajet** elements = new Trajet* [nouvelleTaille];
		for (int i = 0; i < nbElements; i++)
		{
			elements[i] = buf[i];
		}
		
		elements[nbElements++] = t;
		tailleTableau = nouvelleTaille;
		
		cout << "Nouveau nbElements" << nbElements << endl;
		cout << "Nouveau TailleTableau" << tailleTableau << endl;
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

	for (int i = 0; i < nbElements; i++)
	{
		elements[i] =  unCollection.elements[i];
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


Collection::~Collection ( )
{
	for (int i = 0; i < nbElements; i++){
		delete elements[i];
	}

	delete [] elements;
	{
#ifdef MAP
		cout << "Appel au destructeur de <Collection>" << endl;
#endif
	} //----- Fin de ~Collection
}

//------------------------------------------------------------------ PRIVE

//----------------------------------------------------- Méthodes protégées

