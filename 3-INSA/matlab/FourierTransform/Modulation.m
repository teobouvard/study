function Modulation

close all;

%bornes temporelles
a = - 5;
b = 5;

%nombre d'échantillons
N = 16384;

%fréquence d'échantillonage
fe = N/(b-a);

%période d'échantillonnage
te = 1/fe;

%fréquence de coupure du filtre passe bas
f_coupure = 12*20;

%temps en abscisse : N intervalles -> il faut s'arrêter à b-te
xt = linspace(a,b-te,N);

%fréquence en abscisse : N intervalles -> il faut s'arrêter à b-te
xf = linspace(-fe/2,fe/2-1/(b-a),N);

%fréquence des deux ondes porteuses
f1 = 100;
f2 = 200;

%initialisation des signaux pour avoir une taille fixe avant la boucle
s1 = zeros(1,N);
s2 = zeros(1,N);
c = zeros(1,N);
d1 = zeros(1,N);
d2 = zeros(1,N);

for n=1:N
    
    t = (n-1)*te + a;
    
    s1(1,n)= cos(2*pi*2*t) + cos(2*pi*4*t) + cos(2*pi*6*t) + cos(2*pi*8*t) + cos(2*pi*10*t);
    s2(1,n)= 5*cos(2*pi*2*t) + 4*cos(2*pi*4*t) + 3*cos(2*pi*6*t) + 2*cos(2*pi*8*t) + cos(2*pi*10*t);
    
    %modulation
    c(1,n) = s1(1,n)*cos(2*pi*f1*t) + s2(1,n)*cos(2*pi*f2*t);
    
    %démodulation
    d1(1,n) = c(1,n)*cos(2*pi*f1*t);
    d2(1,n) = c(1,n)*cos(2*pi*f2*t);
    
end

%filtrage passe bas pour récupérer le signal d'origine
e1 = real(tfour(d1));
e1(1,1:N/2 - f_coupure) = 0;
e1(1,N/2 + f_coupure:N) = 0;

e2 = real(tfour(d2));
e2(1,1:N/2 - f_coupure) = 0;
e2(1,N/2 + f_coupure:N) = 0;

figure('units','normalized','outerposition',[0 0 1 1]);

%Partie Réelle des spectres en fréquence

subplot(4,2,1);
plot(xf,real(tfour(s1)));
title('spectre (s1)')

subplot(4,2,2);
plot(xf,real(tfour(s2)));
title('spectre (s2)')

subplot(4,2,[3,4]);
plot(xf,real(tfour(c)));
title('spectre (c)')

subplot(4,2,5);
plot(xf,real(tfour(d1)));
title('spectre (d1)')

subplot(4,2,6);
plot(xf,real(tfour(d2)));
title('spectre (d2)')

subplot(4,2,7);
plot(xf,e1);
title('spectre (e1)')

subplot(4,2,8);
plot(xf,e2);
title('spectre (e2)')


    
end
