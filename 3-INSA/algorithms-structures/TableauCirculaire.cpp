#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


typedef struct Liste{
	int valeur;
	Liste * suivant;
} Liste;

int main() {

	char lecture[100];
	int tabCirculaire[100];
	int val;
	int indiceD = 0;
	int indiceF = 0;

	fscanf(stdin, "%99s", lecture);

	while (strcmp(lecture, "bye") != 0) {
		if (strcmp(lecture, "queue") == 0) {
			fscanf(stdin, "%99s", lecture);
			val = (int)strtol(lecture, NULL, 10);
			//valeur tableau
			tabCirculaire[indiceF] = val;
			if (indiceF == 99) {
				indiceF = 0;
			}
			indiceF++;

		}
		else if (strcmp(lecture, "dequeue") == 0) {
			//sort valeur
			if (indiceD != indiceF) {
				printf("%d\r\n", tabCirculaire[indiceD]);
				tabCirculaire[indiceD] = NULL;
				if (indiceD == 99) {
					indiceD = 0;
				}
				indiceD++;
			}

		}
		fscanf(stdin, "%99s", lecture);
	}

	return 0;

}