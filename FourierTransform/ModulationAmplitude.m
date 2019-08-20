function [] = ModulationAmplitude(typeSpectre)

close all;

a = - 5;
b = 5;
N = 16384;
fe = N/(b-a);
te = 1/fe;

xt = linspace(a,b-te,N);    % N intervalles -> il faut s'arrêter à b-T
xf = linspace(-fe/2,fe/2-1/(b-a),N);    %idem

f1 = 100;
f2 = 200;

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

subplot(6,4,[1,2]);
plot(xt,s1);
title('Signal s1')

subplot(6,4,[3,4]);
plot(xt,s2);
title('Signal s2')

subplot(6,4,[9,12]);
plot(xt,c);
title('Signal c')

subplot(6,4,[17,18]);
plot(xt,d1);
title('Signal d1')

subplot(6,4,[19,20]);
plot(xt,d2);
title('Signal d2')

%Amplitude/Phase
if (typeSpectre==0)
    subplot(6,4,5);
    plot(xf,real(tfour(s1)));
    title('Partie réelle de la transformée de Fourier de s1')
    
    subplot(6,4,6);
    plot(xf,imag(tfour(s1)));
    title('Partie imaginaire de la transformée de Fourier de s1')
    
    subplot(6,4,7);
    plot(xf,real(tfour(s2)));
    title('Partie réelle de la transformée de Fourier de s2')
    
    subplot(6,4,8);
    plot(xf,imag(tfour(s2)));
    title('Partie imaginaire de la transformée de Fourier de s2')
    
    subplot(6,4,[13,14]);
    plot(xf,real(tfour(c)));
    title('Partie réelle de la transformée de Fourier de c')
    
    subplot(6,4,[15,16]);
    plot(xf,imag(tfour(c)));
    title('Partie imaginaire de la transformée de Fourier de c')
    
    subplot(6,4,21);
    plot(xf,real(tfour(d1)));
    title('Partie réelle de la transformée de Fourier de d1')
    
    subplot(6,4,22);
    plot(xf,imag(tfour(d1)));
    title('Partie imaginaire de la transformée de Fourier de d1')
    
    subplot(6,4,23);
    plot(xf,real(tfour(d2)));
    title('Partie réelle de la transformée de Fourier de d2')
    
    subplot(6,4,24);
    plot(xf,imag(tfour(d2)));
    title('Partie imaginaire de la transformée de Fourier de d2')
    
end

%Partie Réelle/Imaginaire
if (typeSpectre==1)
    subplot(3,2,3);
    plot(xf,abs(F));
    title('Module de la transformée de Fourier')
    
    subplot(3,2,4);
    plot(xf,mod(angle(F),pi)); %modulo pi pour n'avoir qu'une seule valeur ?
    title('Argument de la transformée de Fourier')
end

end