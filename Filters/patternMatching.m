function [] = patternMatching2

warning('off', 'Images:initSize:adjustingMag');
close all;

%initialisation du seuil de corrélation 
threshold = 0.95;

%lecture de l'image de jeu et transformation en version fréquentielle
tableau = imread('puissance6.png');
[l_tableau,c_tableau,~] = size(tableau);
tableau_frequency = fft2(tableau);


%lecture des patterns issus de l'image du jeu
%les patterns sont retournés afin d'utiliser le résultat 
%de la convolution comme étant un calcul de corrélation

pattern1h = rgb2gray(imread('pattern1h.png'));
pattern1h = rot90(pattern1h,2);
pattern1h_frequency = fft2(pattern1h,l_tableau,c_tableau);

pattern1v = rgb2gray(imread('pattern1v.png'));
pattern1v = rot90(pattern1v,2);
pattern1v_frequency = fft2(pattern1v,l_tableau,c_tableau);

pattern2h = rgb2gray(imread('pattern2h.png'));
pattern2h = rot90(pattern2h,2);
pattern2h_frequency = fft2(pattern2h,l_tableau,c_tableau);

pattern2v = rgb2gray(imread('pattern2v.png'));
pattern2v = rot90(pattern2v,2);
pattern2v_frequency = fft2(pattern2v,l_tableau,c_tableau);


%calculs de convolution en version fréquentielle
%puis reconversion en version spatiale
matching_frequency_1h = pattern1h_frequency .* tableau_frequency;
matching_1h = ifft2(matching_frequency_1h);

matching_frequency_1v = pattern1v_frequency .* tableau_frequency;
matching_1v = ifft2(matching_frequency_1v);

matching_frequency_2h = pattern2h_frequency .* tableau_frequency;
matching_2h = ifft2(matching_frequency_2h);

matching_frequency_2v = pattern2v_frequency .* tableau_frequency;
matching_2v = ifft2(matching_frequency_2v);

%extraction des maxima de corrélation 
matching_max_1h = max(max(matching_1h));
matching_max_1v = max(max(matching_1v));
matching_max_2h = max(max(matching_2h));
matching_max_2v = max(max(matching_2v));

%matrices binaires correspondant aux points supérieurs au seuil de corrélation
D1_h = matching_1h > threshold*matching_max_1h;
D1_v = matching_1v > threshold*matching_max_1v;
D1 = D1_h + D1_v;
compteur1 = nnz(D1_h)+nnz(D1_v);

D2_h = matching_2h > threshold*matching_max_2h;
D2_v = matching_2v > threshold*matching_max_2v;
D2 = D2_h + D2_v;
compteur2 = nnz(D2_h)+nnz(D2_v);

%éléments structurels repérant les corrélations
disks = strel('disk',4);
squares = strel('disk',4);
mask1 = imdilate(D1,disks);
mask2 = imdilate(D2, squares);

%fusion des différentes couches
tableau1 = imfuse(mask1,tableau);
tableau2 = imfuse(mask2,tableau);
tableau = imfuse(tableau1,tableau2);

%écriture du fichier image
imwrite(tableau,'correlation.png');

%affichage
message = sprintf('Compteur 1 = %d\nCompteur 2 = %d', compteur1,compteur2);

figure 
imshow(tableau) ;
text(1050,512,message, 'FontSize', 20, 'FontWeight', 'bold');
