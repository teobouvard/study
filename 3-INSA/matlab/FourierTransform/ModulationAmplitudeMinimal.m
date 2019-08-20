function [] = ModulationAmplitudeMinimal(typeSpectre)

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

%temps en abscisse :N intervalles -> il faut s'arrêter à b-te
xt = linspace(a,b-te,N);

%fréquence en abscisse :N intervalles -> il faut s'arrêter à b-te
xf = linspace(-fe/2,fe/2-1/(b-a),N);

%fréquence des deux signaux à transmettre
f1 = 10;
f2 = 20;

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

figure('units','normalized','outerposition',[0 0 1 1]);

%Amplitude/Phase
if (typeSpectre==0)
    subplot(3,4,1);
    plot(xf,real(tfour(s1)));
    title('real(s1)')
    
    subplot(3,4,2);
    plot(xf,imag(tfour(s1)));
    title('imaginary(s1)')
    
    subplot(3,4,3);
    plot(xf,real(tfour(s2)));
    title('real(s2)')
    
    subplot(3,4,4);
    plot(xf,imag(tfour(s2)));
    title('imaginary(s2)')
    
    subplot(3,4,[5,6]);
    plot(xf,real(tfour(c)));
    title('real(c)')
    
    subplot(3,4,[7,8]);
    plot(xf,imag(tfour(c)));
    title('imaginary(c)')
    
    subplot(3,4,9);
    plot(xf,real(tfour(d1)));
    title('real(d1)')
    
    subplot(3,4,10);
    plot(xf,imag(tfour(d1)));
    title('imaginary(d1)')
    
    subplot(3,4,11);
    plot(xf,real(tfour(d2)));
    title('real(d2)')
    
    subplot(3,4,12);
    plot(xf,imag(tfour(d2)));
    title('imaginary(d2)')
    
end


%Partie Réelle/Imaginaire
if (typeSpectre==1)
    subplot(3,4,1);
    plot(xf,abs(tfour(s1)));
    title('module(s1)')
    
    subplot(3,4,2);
    plot(xf,angle(tfour(s1)));
    title('argument(s2)')
    
    subplot(3,4,3);
    plot(xf,abs(tfour(s2)));
    title('module(s2)')
    
    subplot(3,4,4);
    plot(xf,angle(tfour(s2)));
    title('argument(s2)')
    
    subplot(3,4,[5,6]);
    plot(xf,abs(tfour(c)));
    title('module(c)')
    
    subplot(3,4,[7,8]);
    plot(xf,angle(tfour(c)));
    title('argument(c)')
    
    subplot(3,4,9);
    plot(xf,abs(tfour(d1)));
    title('module(d1)')
    
    subplot(3,4,10);
    plot(xf,angle(tfour(d1)));
    title('argument(d1)')
    
    subplot(3,4,11);
    plot(xf,abs(tfour(d2)));
    title('module(d2)')
    
    subplot(3,4,12);
    plot(xf,angle(tfour(d2)));
    title('argument(d2)')
    
end

end