function [vf]=tfourinv(vFw)


%tfi	Transformée de Fourier Inverse discrète.
% 	tfi(vFw) est la transformée de Fourier inverse discrète de 
%	la fonction F(w) dont (ne) echantillons sont rangés dans le 
%	vecteur ligne vFw. La valeur F(0) doit être dans 
%	l'echantillon (ne/2 + 1).
%
%  	Voir aussi tf, IFFT, FFT2, IFFT2, FFTSHIFT.



if (size(vFw,1)==1)

	ne=size(vFw,2);

	i=sqrt(-1);
	nes2=ne/2;
	for n=1:ne,
	   	vFw(n)=exp(-i*(n-nes2+1)*pi)*vFw(n);
	end
	vFw=fftshift(vFw);
	vf=ifft(vFw);
else 	'tfi : Le vecteur des données doit être un vecteur ligne'
end