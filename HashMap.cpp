#define _CRT_SECURE_NO_WARNINGS
#define EMPTY 0
#define DELETED 1
#define USED 2

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>


/* mettez ici vos déclarations de fonctions et types */



typedef char * Key;
unsigned int HashFunction(Key key, unsigned int size);

typedef struct Cellule {
	int status;
	char * key;
	char * val;
}Celulle;

typedef struct Hash {
	int taille;
	Cellule * tab;
}Hash;

void error(void);

int main(void)
{
	int size;
	char lecture[100];
	char * key;
	char * val;
	Hash h;
	

	if (fscanf(stdin, "%99s", lecture) != 1)
		error();
	while (strcmp(lecture, "bye") != 0)
	{
		if (strcmp(lecture, "init") == 0)
		{
			if (fscanf(stdin, "%99s", lecture) != 1)
				error();
			size = atoi(lecture);
			/* mettre le code d'initialisation ici */
			h.taille = size;
			h.tab = (Cellule*)malloc(sizeof(Cellule)*h.taille);
			for (int i = 0; i < h.taille; i++) {
				h.tab[i].status = EMPTY;
				
			}
			
		}
		else if (strcmp(lecture, "insert") == 0)
		{
			if (fscanf(stdin, "%99s", lecture) != 1)
				error();
			key = strdup(lecture);
			if (fscanf(stdin, "%99s", lecture) != 1)
				error();
			val = strdup(lecture);
			/* mettre ici le code d'insertion */
			unsigned int index = HashFunction(key, h.taille);
			int cpt = 0;
			while (h.tab[index].status != EMPTY && cpt <= h.taille && strcmp(key,  h.tab[index].key)){ //ou EMPTY || DELETED
				//std::cout << "case occupee : " << index << " statut " << h.tab[index].status << "cle 1 " << key <<" cle 2 " << h.tab[index].key <<  "\r\n";
				if (index == h.taille - 1) {
					index = 0;
				} else {
					index++;
				}
				
				cpt++;
				
			}
			if (cpt <= h.taille) {
				h.tab[index].key = key;
				h.tab[index].val = val;
				h.tab[index].status = USED;
				//std::cout << "insertion de la valeur " << key << " : " << val << " a l'index " << index << "\r\n";
			} else {
				//std::cout << "pasdplace";
			}

		}
		else if (strcmp(lecture, "delete") == 0)
		{
			if (fscanf(stdin, "%99s", lecture) != 1)
				error();
			key = strdup(lecture);
			/* mettre ici le code de suppression */
			for (int i = 0; i < h.taille; i++) {
				if (h.tab[i].status == USED && !strcmp(h.tab[i].key, key)) {
					h.tab[i].status = DELETED;

				}
			}
		}
		else if (strcmp(lecture, "query") == 0)
		{
			if (fscanf(stdin, "%99s", lecture) != 1)
				error();
			key = strdup(lecture);
			/* mettre ici le code de recherche et traitement/affichage du résultat */
			bool found = false;
			for (int i = 0; i < h.taille; i++) {
				if (h.tab[i].status == USED && !strcmp(h.tab[i].key, key)) {
					std::cout << h.tab[i].val << "\r\n";
					found = true;

				}
			}
			if (!found) {
				std::cout << "Not found" << "\r\n";
			}
		}
		else if (strcmp(lecture, "destroy") == 0)
		{
			free(h.tab);
		}
		else if (strcmp(lecture, "stats") == 0)
		{
			/* mettre ici le code de statistiques */
			int cptE = 0;
			int cptD = 0;
			int cptU = 0;
			for (int i = 0; i < h.taille; i++) {
				switch (h.tab[i].status) {
					case 0: cptE++; break;
					case 1: cptD++; break;
					case 2: cptU++; break;
				}
			}
			std::cout << "size    : " << h.taille << "\r\n";
			std::cout << "empty   : " << cptE << "\r\n";
			std::cout << "deleted : " << cptD << "\r\n";
			std::cout << "used    : " << cptU << "\r\n";
			
		}

		if (fscanf(stdin, "%99s", lecture) != 1)
			error();
	}
	return 0;
}

/* fonction de décalage de bit circulaire */
unsigned int shift_rotate(unsigned int val, unsigned int n)
{
	n = n % (sizeof(unsigned int) * 8);
	return (val << n) | (val >> (sizeof(unsigned int) * 8 - n));
}

/* fonction d'encodage d'une chaîne de caractères */
unsigned int Encode(Key key)
{
	unsigned int i;
	unsigned int val = 0;
	unsigned int power = 0;
	for (i = 0; i<strlen(key); i++)
	{
		val += shift_rotate(key[i], power * 7);
		power++;
	}
	return val;
}

/* fonction de hachage simple qui prend le modulo */
unsigned int hash(unsigned int val, unsigned int size)
{
	return val%size;
}

/* fonction de hachage complète à utiliser */
unsigned int HashFunction(Key key, unsigned int size)
{
	return hash(Encode(key), size);
}

/* placer ici vos définitions (implémentations) de fonctions ou procédures */

void error(void)
{
	printf("input error\r\n");
	exit(0);
}