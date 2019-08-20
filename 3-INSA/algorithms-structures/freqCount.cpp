#include <iostream>

using namespace std;

// On considère une bijection entre l'ensembe des mots possibles et les entiers de 0 à 64
// (4*4*4 combinaisons de lettres possibles)
int main() {
	
	int mots_possibles[64] = {};
	int occurences_max = 0;
	char c1, c2, c3, separator = '-';
	
	while(separator == '-' && cin >> c1 >> c2 >> c3) {
		cin >> separator;
		
		// Interprétation d'un mot comme un entier unique
		int code_mot = (c3-'A') + (c2-'A')*4 + (c1-'A')*16;
		mots_possibles[code_mot]++;
		if(mots_possibles[code_mot] > occurences_max) {
			++occurences_max;
		}
	}
	
	cout << occurences_max << "\r\n";
	for(int i=0; i<64; ++i) {
		if(mots_possibles[i] == occurences_max && occurences_max > 0) {
			// Interprétation d'un entier comme un mot
			int n = i;
			char c1 = 'A', c2 = 'A', c3 = 'A';
			c1 += n/16;
			n %= 16;
			c2 += n/4;
			n %= 4;
			c3 += n;
			cout << c1 << c2 << c3 << "\r\n";
		}
	}
	return 0;
}