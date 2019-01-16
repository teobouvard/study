function [matching_frequency] = patternMatching

compteur1 = 0;
compteur2 = 0;

map=([0:255]'/255)*[1 1 1];

tableau = imread('puissance6.png');
%tableau = 0.3*tableau(:,:,1) + 0.59*tableau(:,:,2) + 0.11*tableau(:,:,3);
[l_tableau,c_tableau,~] = size(tableau);
tableau_frequency = fft2(tableau);

pattern1h = imread('pattern1h.png');
pattern1h = 0.3*pattern1h(:,:,1) + 0.59*pattern1h(:,:,2) + 0.11*pattern1h(:,:,3);
[l_pattern1h,c_pattern1h,~] = size(pattern1h);
pattern1h = fliplr(flipud(pattern1h));
pattern1h_frequency = fft2(pattern1h,l_tableau,c_tableau);

pattern1v = imread('pattern1v.png');
[l_pattern1v,c_pattern1v,~] = size(pattern1v);

pattern2h = imread('pattern2h.png');
[l_pattern2h,c_pattern2h,~] = size(pattern2h);

pattern2v = imread('pattern2v.png');
[l_pattern2v,c_pattern2v,~] = size(pattern2v);


matching_frequency = pattern1h_frequency .* tableau_frequency;
matching = ifft2(matching_frequency);


matching_max = max(max(matching));
thresh = 0.95*matching_max; % Use a threshold that's a little less than max.
D = matching > thresh;
se = strel('disk',5);
E = imdilate(D,se);
figure
imshow(E) % Display pixels with values over the threshold.


%surf(abs(matching))