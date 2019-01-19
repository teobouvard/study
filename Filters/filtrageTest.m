function filtrageTest(k)

    im = rgb2gray(imread('Stephane_Bres.jpg'));
    [l,c,~]=size(im);

    %creation d'un mapping couleur linéaire sur chaque flux RGB
    map=([0:255]'/255)*[1 1 1];

    %calcul du spectre frequentiel 
    spectre_freq = fftshift(fft2(im));

    %initialisation du filtre
    gaussMatrix = zeros(l,c);

    for n = 0.1:0.1:5

        k = -power(10,-n);

        for u = 1:l
            for v = 1:c
                gaussMatrix(u,v)=exp(k*((u-l/2+1)^2+(v-c/2+1)^2));
            end
        end

        %filtrage fréquentiel
        spectre_filtre = spectre_freq .* gaussMatrix;

        %version affichable
        spectre_filtre_aff = (log(abs(spectre_filtre)+1));

        %reorganisation representation frequentielle
        spectre_filtre = fftshift(spectre_filtre);

        %calcul de l'image filtrée par transformée de fourier inverse
        im_filtre = ifft2(spectre_filtre);

        %affichage
        subplot(1,2,1);
        image(abs(im_filtre));
        pbaspect([1 1 1]);
        colormap(map);

        subplot(1,2,2);
        imagesc(spectre_filtre_aff);
        pbaspect([1 1 1]);

        waitforbuttonpress;
    end

    close all;
end

