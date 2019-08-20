#include <iostream>
using namespace std;
int **& resoudre(int **  &sol, int poids[], int C, int n) {
	sol[0][0] = 1;
	for (int i = 1; i < n; i++) {
		for (int j = 0; j < C; j++) {
			if ((sol[i - 1][j] == 1) || (j - poids[i - 1] >= 0) && (sol[i - 1][j - poids[i - 1]] == 1)) {
				sol[i][j] = 1;

			}
		}
	}
	return sol;
}
int optimal(int **& solfiole1, int n1, int ** & solfiole2, int n2, int C) {
	int max = 0;
	for (int i(0); i < C; i++) {
		for (int j(0); j < C; j++) {
			if (solfiole1[n1][i] == 1 && solfiole2[n2][j] == 1 && (i+j) <= C) {
				int produit = 2 * (i < j ? i : j);
				if (produit > max)
					max = produit;
			}
		}
	}
	return max;
}
int main() {
	int C;
	int n;
	int fiole1[20];
	int fiole2[20];
	int nf1 = 0, nf2 = 0;
	cin >> C >> n;
	for (int i(0); i < n; i++) {
		int gramme, appartenance;
		cin >> gramme >> appartenance;
		if (appartenance == 1) {
			fiole1[nf1++] = gramme;
		}

		else
			fiole2[nf2++] = gramme;
	}
	int** solfiole1 = new int*[nf1 + 1];
	for (int i(0); i < nf1 + 1; i++) {
		solfiole1[i] = new int[C];
	}
	for (int i = 0; i < nf1 + 1; i++) {
		for (int y = 0; y < C; y++) {
			solfiole1[i][y] = 0;
		}
	}
	int** solfiole2 = new int*[nf2 + 1];
	for (int i(0); i < nf2 + 1; i++) {
		solfiole2[i] = new int[C];
	}
	for (int i = 0; i < nf2 + 1; i++) {
		for (int y = 0; y < C; y++) {
			solfiole2[i][y] = 0;
		}
	}
	solfiole1 = resoudre(solfiole1, fiole1, C, nf1 + 1);
	solfiole2 = resoudre(solfiole2, fiole2, C, nf2 + 1);
	cout << optimal(solfiole1, nf1, solfiole2, nf2, C) << "\r\n";
}