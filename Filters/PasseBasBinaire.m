function PasseBasBinaire(img)

    close all;

    %lecture des images test
    if img == 0
    im = imread('imtest01.png');
    else
    im = imread('imtest02.png');
    end

    [l,c,~]=size(im);

    %creation d'un mapping couleur linéaire sur chaque flux RGB
    map=([0:255]'/255)*[1 1 1];

    %calcul du spectre frequentiel 
    spectre_freq = fftshift(fft2(im));

    %création d'une version affichable de la représentation fréquentielle
    spectre_freq_aff = (log(abs(spectre_freq))+1);

    %initialisation du filtre
    binaryFilter = zeros(l,c);
    
    figure('units','normalized','outerposition',[0 0 1 1]);

    for r = 1:2:180
        
        ratio = r/180
        
        for u = 1:l
            for v = 1:c
                if (((u-l/2+1)^2+(v-c/2+1)^2) < r^2)
                    binaryFilter(u,v) = 1;
                end
            end
        end

        %filtrage fréquentiel
        spectre_filtre = spectre_freq .* binaryFilter;

        %version affichable
        spectre_filtre_aff = (log(abs(spectre_filtre)+1));

        %reorganisation representation frequentielle
        spectre_filtre = fftshift(spectre_filtre);

        %calcul de l'image filtrée par transformée de fourier inverse
        im_filtre = ifft2(spectre_filtre);

        %affichage
        subplot(2,2,1);
        image(im);
        colormap(map);

        subplot(2,2,2);
        imagesc(spectre_freq_aff);

        subplot(2,2,3);
        image(abs(im_filtre));
        colormap(map);

        subplot(2,2,4);
        imagesc(spectre_filtre_aff);

        waitforbuttonpress;

    end

    close all;
end
