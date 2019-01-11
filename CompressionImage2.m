function [] = CompressionImage2(qualite)

warning('off', 'Images:initSize:adjustingMag');

%lecture de l'image et conversion en classe double pour l'utiliation de SVD
inputImage = imread('imageMystere2.jpg');
inputImage = im2double(inputImage);

%décomposition de l'image en 3 matrices RGB
inputImageRed = inputImage(:,:,1);
inputImageGreen = inputImage(:,:,2);
inputImageBlue = inputImage(:,:,3);

%décomposition en valeur singulières de chaque couche, en supprimant les
%zéros de la matrice diagonale ('econ')
[URed, DRed, VRed] = svd(inputImageRed,'econ');
[UGreen, DGreen, VGreen] = svd(inputImageGreen,'econ');
[UBlue, DBlue, VBlue] = svd(inputImageBlue,'econ');

%mise à zéro des valeurs singulières de rang supérieur à la qualité
%demandée
NbValSing = size(DRed);


DRed(floor(qualite*NbValSing):end,:) = 0;
DRed(:,floor(qualite*NbValSing):end) = 0;

DGreen(floor(qualite*NbValSing)+1:end,:) = 0;
DGreen(:,floor(qualite*NbValSing)+1:end) = 0;

DBlue(floor(qualite*NbValSing)+1:end,:) = 0;
DBlue(:,floor(qualite*NbValSing)+1:end) = 0;

%reconstitution de chaque couche
outputImageRed = URed*DRed*VRed';
outputImageGreen = UGreen*DGreen*VGreen';
outputImageBlue = UBlue*DBlue*VBlue';

%reconsitution de l'image de sortie par concaténation des RGB
outputImage = cat(3, outputImageRed, outputImageGreen, outputImageBlue);

%affichage de l'image compressée
imshow(outputImage);





end

