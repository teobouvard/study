function passeBasGaussien(img)
close all;

if img == 0
im = imread('imtest01.png');
else
im = imread('imtest02.png');
end

[l,c,~]=size(im);

map=([0:255]'/255)*[1 1 1];

IM = fftshift(fft2(im));%on passe en frequentiel
IMaff = (log(abs(IM)+1));%création d'une version affichable de la représentation fréquentielle

gaussMatrix = zeros(l,c); %création du filtre
IMF = zeros(l,c); %création de la représentation fréquentielle filtrée

for n = 1:0.3:5
    k = -power(10,-n)
    for u = 1:l
        for v = 1:c
            gaussMatrix(u,v)=exp(k*((u-l/2+1)^2+(v-c/2+1)^2));
            IMF(u,v)=IM(u,v)*gaussMatrix(u,v);
        end
    end
    
    IMFaff = (log(abs(IMF)+1));

    IMF = fftshift(IMF);%reorganisation representation frequentielle
    imf = ifft2(IMF);%on passe en spatial

    subplot(2,2,1);
    image(im);
    colormap(map);

    subplot(2,2,2);
    imagesc(IMaff);

    subplot(2,2,3);
    image(abs(imf));
    colormap(map);

    subplot(2,2,4);
    imagesc(IMFaff);
    
    waitforbuttonpress;

end
end

