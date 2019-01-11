
#include <iostream>
using namespace std;

int factorial(int n){
	if (n==1){
		return n;
	}
	else{
		return n*factorial(n-1);
	}
}


int main() {
	
	int nombre;
	cin >> nombre;
	int result = factorial(nombre);
	cout << result << "\r\n";
    return 0;
}

