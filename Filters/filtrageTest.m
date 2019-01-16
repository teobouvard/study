function filtrageTest(k)

im = imread('Stephane_Bres2.jpg');
[l,c,m]=size(im);
im = 0.3*im(:,:,1) + 0.59*im(:,:,2) + 0.11*im(:,:,3);

IM = fftshift(fft2(im));%on passe en frequentiel
IMaff = (log(abs(IM)+1));%reorganisation representation frequentielle

gaussMatrix = zeros(l,c);
IMF = zeros(l,c);

for u = 1:l
    for v = 1:c
        gaussMatrix(u,v)=exp(-k*((u-l/2+1)^2+(v-c/2+1)^2));
        IMF(u,v)=IM(u,v)*gaussMatrix(u,v);
    end
end


IMF = fftshift(IMF);%reorganisation representation frequentielle
imf = ifft2(IMF);%on passe en spatial
(max(abs(imag(imf))))
mapbw=([0:255]'/255)*[1 1 1];
figure(1)
%imagesc(log(abs(fftshift(IMBW))))
image((abs(imf)));
colormap(mapbw);