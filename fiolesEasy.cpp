#include <iostream>
using namespace std;

#include <math.h>


int main(){
	
	int V,S,modifs,res;
	bool rempli = false;
	modifs = 0;
	res = 0;
	
	cin >> V;
	cin >> S;
	
	V++;
	
	int tab[V];
	
	//init tableau à 0
	for (int i = 0; i < V; i++){
		tab[i] = 0;
	}
	
	//remplissage tab
	tab[S+1] = 1;
	
	
	while (!rempli){
		rempli = true;
		for (int i = 0; i < V; i++){
			if (tab[i] == 1){
				
				if (((int)floor(i/5) >= 0) && tab[(int)floor(i/5)] == 0){
					tab[(int)floor(i/5)] = 1;
					modifs++;
				}
				
				else if (((int)2*i < V) && (tab[(int)2*i]) == 0){
					tab[(int)2*i] = 1;
					modifs++;
				}
				
				else if (((int)3*i < V) && (tab[(int)3*i]) == 0){
					tab[(int)3*i] = 1;
					modifs++;
					
				}
			}
		}
		if (modifs != 0){
			rempli = false;
			modifs = 0;
		}
	}

	for (int i = 0; i < V; i++){
		if (tab[i] == 1) {
			res = i;
		}
	}

	cout << res << "\r\n";
	
	return 0;
}