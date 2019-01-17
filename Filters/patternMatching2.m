function [compteur1] = patternMatching

compteur1 = 0;
compteur2 = 0;
offset = [-50 2];


map=([0:255]'/255)*[1 1 1];

tableau = imread('puissance6.png');
[l_tableau,c_tableau,~] = size(tableau);
tableau_frequency = fft2(tableau);

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


matching_frequency = pattern1h_frequency .* tableau_frequency;
matching = ifft2(matching_frequency);


matching_max = max(max(matching));
threshold = 0.95*matching_max; % Use a threshold that's a little less than max.
D = matching > threshold;
compteur1 = nnz(D);

disks = translate(strel('disk',5),offset);

E = imdilate(D,disks);

figure
E = imfuse(E,tableau);
imshow(E) % Display pixels with values over the threshold.


%surf(abs(matching))
