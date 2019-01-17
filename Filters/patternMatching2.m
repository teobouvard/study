function [compteur1, compteur2] = patternMatching2()
warning('off', 'Images:initSize:adjustingMag');
close all;

compteur1 = 0;
compteur2 = 0;
threshold = 0.95;

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


matching_frequency_1h = pattern1h_frequency .* tableau_frequency;
matching_1h = ifft2(matching_frequency_1h);

matching_frequency_1v = pattern1v_frequency .* tableau_frequency;
matching_1v = ifft2(matching_frequency_1v);

matching_frequency_2h = pattern2h_frequency .* tableau_frequency;
matching_2h = ifft2(matching_frequency_2h);

matching_frequency_2v = pattern2v_frequency .* tableau_frequency;
matching_2v = ifft2(matching_frequency_2v);


matching_max_1h = max(max(matching_1h));
matching_max_1v = max(max(matching_1v));
matching_max_2h = max(max(matching_2h));
matching_max_2v = max(max(matching_2v));


D1_h = matching_1h > threshold*matching_max_1h; % Use a threshold that's a little less than max.
D1_v = matching_1v > threshold*matching_max_1v;
D1 = D1_h + D1_v;
D1 = D1 > 0.5;
compteur1 = nnz(D1);

D2_h = matching_2h > threshold*matching_max_2h; % Use a threshold that's a little less than max.
D2_v = matching_2v > threshold*matching_max_2v;
D2 = D2_h + D2_v;
D2 = D2 > 0.5;
compteur2 = nnz(D2);

disks = strel('disk',5);
mask1 = imdilate(D1,disks);


figure

tableau = imfuse(mask1,tableau);
imshow(tableau) % Display pixels with values over the threshold.


