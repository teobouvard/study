#include <stdio.h>


int main() {
  int a;
  scanf("%d",&a);

  printf("%d\r\n",a);

  if (a<1 || a > 1000){
    return 0;
  }

  else{
    while(a!=1){
      if(a%2==0){
        a=a/2;
        printf("%d\r\n", a);
      }
      else{
        a=3*a+1;
        printf("%d\r\n", a);
      }
    }
    return 0;
  }
}
