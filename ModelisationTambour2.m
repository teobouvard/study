function [] = ModelisationTambour2(valPropreChoisie)

altitudeFixe = 1;

%maillage de la surface du tambour
surfaceT = zeros(15,40);
[m,n] = size(surfaceT);

%initialisation du laplacien (m*n = 600)
laplacien = -4*eye(m*n);

laplacien(1,1) = altitudeFixe; %point fixe en haut à gauche

for i = 15*m+1:m:n*m %semi-bord haut
    laplacien(i,i) = altitudeFixe;
end
for i = m*(n-1):m*n %bord droit
    laplacien(i,i) = altitudeFixe;
end
for i = m:m:m*(n-1) %bord bas
    laplacien(i,i) = altitudeFixe;
end
for i = (10*m)+6:(10*m)+10 %barre fixe 1
    laplacien(i,i) = altitudeFixe;
end
for i = (25*m)+6:(25*m)+10 %barre fixe 2
    laplacien(i,i) = altitudeFixe;
end

%discrétisation du laplacien

%point normaux
for i = m+1:m*(n-1)
    if (laplacien(i,i) == -4)
        laplacien(i,i-1) = 1;
        laplacien(i,i+1) = 1;
        laplacien(i,i+m) = 1;
        laplacien(i,i-m) = 1;
    end
end

%bord haut non contraint
for i = m+1:m:13*(m+1)
   laplacien(i,i) = -3;
   laplacien(i,i-1) = 0;
end

%bord gauche non contraint
for i = 1:m
    if (laplacien(i,i) == -4)
        laplacien(i,i) = -3;
        laplacien(i,i-1) = 1;
        laplacien(i,i+1) = 1;
        laplacien(i,i+m) = 1;
    end
end

invLaplacien = inv(laplacien);

[ valeursPropres , vecteursPropres ] = Deflation(invLaplacien, valPropreChoisie);

%attribution des valeurs propres réelles
for i = 1:valPropreChoisie
    valeursPropres(i,1) = (-1/valeursPropres(i,1));
end

%on attribue à chaque maille la composante du vecteur propre associé
vectorT = vecteursPropres(:,valPropreChoisie);

surfaceT = reshape(vectorT,m,n);

%affichage de la surface du tambour
surf(surfaceT);

end

