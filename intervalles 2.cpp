#include <iostream>
#include <vector>
#include <math.h>

int aire(int a1, int b1, int a2, int b2){
  int res = 0;

  if (a2>a1){
    if (b2>b1){
      if (b1>a2){
        res = b1-a2;
      }
      else {
        res = 0;
      }
    }
    else res = b2-a2;
  }
  else
  if (b1>b2){
    if (b2>a1){
      res = b2-a1;
    }
    else {
      res = 0;
    }
  }
  else {
    res = b1-a1;
  }

  return res;
}

int main() {

  int nbIntervalles;
  std::vector<int> a,b;

  std::cin >> nbIntervalles;

  for (int i = 0; i < nbIntervalles; i++){
    int debutIntervalle, finIntervalle;
    std::cin >> debutIntervalle >> finIntervalle;
    a.push_back(debutIntervalle);
    b.push_back(finIntervalle);
  }

  for (int j = 0; j < nbIntervalles; j++){
    if (b.at(j) >= a.at(j+1)){
      b.erase(b.begin()+j);
      a.erase(a.begin()+j+1);
      nbIntervalles--;
    }
  }

  std::cout << nbIntervalles << "\r\n";

  for (int k = 0; k < nbIntervalles; k++){
    std::cout << a.at(k) << " " << b.at(k) << "\r\n";
  }

  return 0;
}
