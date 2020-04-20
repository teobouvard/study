
#include <iostream>
#include <math.h>
using namespace std;


int main() {
	
	int tab[100][2];
	int a,b;
	int taille = 0;
	
	cin >> a;
	cin >> b;
	
	
	for (int i = 0; i <= floor(sqrt(a)) ; i++){
		for (int j = 0; j <= floor(sqrt(a)) ; j++){
			if (pow(i,2)+pow(j,2) == a){
				if (pow(i,3)+pow(j,3) == b){
					tab[taille][0] = i;
					tab[taille][1] = j;
					taille++;
				}
			}
		}
	}
	
	if (taille == 0) {
		cout << "-" << "\r\n";
	}
	else{
		for (int i = 0; i < taille ; i++){
			cout << tab[i][0] << " " << tab[i][1] << "\r\n";
		}
	}
	
	return 0;
}

