function [Fw]=tfour(vf)


%tf	Transformée de Fourier discrète.
% 	tf(vf) est la transformée de Fourier discrète de la fonction f
%       dont (ne) échantillons sont rangés dans le vecteur ligne vf. 
%	Ces échantillons correspondent à des "instants" centrés sur la 
%	valeur zéro. La valeur de f(0) doit être contenue dans 
%	l'échantillon (ne/2 + 1).
%
%  	Voir aussi tfi, IFFT, FFT2, IFFT2, FFTSHIFT.

if (size(vf,1)==1)

	ne=size(vf,2);

	i=sqrt(-1);
	nes2=ne/2;
	Fw=fft(vf);
	Fw=fftshift(Fw);
%	for n=1:ne,
%   		Fw(n)=exp(i*(n-nes2+1)*pi)*Fw(n);
%	end
        temp=[1:ne];
        Fw=(exp(i*temp*pi)*exp(-i*(nes2-1)*pi)).*Fw;

else 	'tf : Le vecteur des données doit être un vecteur ligne'
end