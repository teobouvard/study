
#include <iostream>
using namespace std;


int main() {
	
	int lecture;
	int tab[1000];
	int result = 1;
	int taille = 0;
	
	cin >> lecture;
	while (lecture != -1){
		tab[taille] = lecture;
		taille++;
		cin >> lecture;
	}
	
	for (int i = 0; i < (int)(taille/2) ; i++){
		if (tab[i] == tab[taille-i-1]){
			result = 1;
		}
		else {
			result = 0;
			break;
		}
	}
	
	
	
	cout << result << "\r\n";
	
	return 0;
}

