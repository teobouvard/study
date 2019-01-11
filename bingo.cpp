#include <iostream>
#include <math.h>
using namespace std;

//#define MAP



int main() {
	
	int nbCartes = 0;
	int lecture,constanteM;
	int result = 0;
	
	cin >> constanteM;
	cin >> nbCartes;
	
	int vecteur[nbCartes];
	
	for (int i = 0; i < nbCartes; i++){
		cin >> lecture;
		
		vecteur[i] = lecture;
		
	}
	
#ifdef MAP
	for (int i = 0; i < nbCartes; i++){
		cout << "vecteur " << i << " = " << vecteur[i] << endl;
	}
#endif
	
	for (int i = 0; i < nbCartes; i++){
		for (int j = i+1; j < nbCartes; j++){
			for (int k = j+1; k < nbCartes; k++){
				if ((vecteur[i] + vecteur[j] + vecteur[k]) == constanteM){
					result++;
#ifdef MAP
					cout << "i = " << vecteur[i] << " j = " << vecteur[j] << " k = " << vecteur[k] << endl;
#endif
				}
			}
		}
	}
	
	result = (int) result;
	
	cout << result << "\r\n";
	
	return 0;
}

