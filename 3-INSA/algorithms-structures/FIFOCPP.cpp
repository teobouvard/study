/* problème numero 3 - file implementée par une liste chainée */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct Element {
	struct Element * suivant;
	int valeur;
} Element;

typedef struct {
	Element * debut; /* le début de la liste chaînée */
	Element * fin;   /* le dernier élément de la liste */
} File;

void EnFile(File * file, int valeur);
int DeFile(File * file);
int EstVide(File file);
void Init(File * file);
void Destroy(File file);

void error(void);

int main(void)
{
	int val;
	char lecture[100];
	File file;
	
	Init(&file);
	
	if (fscanf(stdin,"%99s",lecture)!=1)
		error();
	while (strcmp(lecture,"bye")!=0)
	{
		if (strcmp(lecture,"enfile")==0)
		{
			if (fscanf(stdin,"%99s",lecture)!=1)
				error();
			val = atoi(lecture);
			EnFile(&file,val);
		}
		else if (strcmp(lecture,"defile")==0)
		{
			val = DeFile(&file);
			printf("%d\r\n",val);
		}
		else if (strcmp(lecture,"estvide")==0)
		{
			printf("%s\r\n",EstVide(file)?"oui":"non");
		}
		
		if (fscanf(stdin,"%99s",lecture)!=1)
			error();
	}
	Destroy(file);
	return 0;
}

void EnFile(File * file, int valeur)
/* cette procédure enfile valeur dans la file */
/*  - file est un pointeur sur la structure de données File et
 *    représente une file bien formée
 *  - valeur est la valeur à insérer */
{
	Element * aEnfiler = malloc(sizeof(Element));
	
	aEnfiler->valeur = valeur;
	
	if(EstVide(*file)){
		file->debut = aEnfiler;
		file->fin = aEnfiler;
	}
	else{
		file->fin->suivant = aEnfiler;
		file->fin = aEnfiler;
	}
}

int DeFile(File * file)
/* cette fonction défile la valeur et la renvoie */
/*  - file est un pointeur sur la structure de données File et
 *    représente une file bien formée
 *  - si la file est vide la valeur 0 est renvoyée */
{
	if (EstVide(*file)){
		return 0;
	}
	else{
		int res = file->debut->valeur;
		file->debut = file->debut->suivant;
		if (file->debut == NULL){
			file->fin = NULL;
		}
		return res;
	}
}

int EstVide(File file)
/* cette fonction indique si la file est vide */
/*  - file est de type File et représente une file bien formée
 *  - la valeur renvoyée vaut 1 si la file est vide, 0 sinon */
{
	if(file.debut == NULL && file.fin == NULL){
		return 1;
	}
	else{
		return 0;
	}
}

void Init(File * file)
{
	file->debut = NULL;
	file->fin = NULL;
}

void Destroy(File file)
{
	Element * courant = file.debut;
	Element * suivant;
	while (courant != NULL)
	{
		suivant = courant->suivant;
		free(courant);
		courant = suivant;
	}
}

void error(void)
{
	printf("input error\r\n");
	exit(0);
}