function [ valeursPropres , vecteursPropres ] = Deflation(A, iterations)

[m,n] = size(A);

valeursPropres = zeros(1,1);
vecteursPropres = zeros(n,1);

for i = 1:iterations                                                   
    %calcul des valeurs et vecteurs propres
    [lambda,vecteurV,vecteurU] = EigenValues(A);
    
    valeursPropres(i,1) = lambda;
    vecteursPropres(:,i) = vecteurV;
    N = vecteurV*vecteurU;
    D = vecteurU*vecteurV;
    B = lambda*(N/D);
    
    %déflation de la plus grande valeur propre
    A = A - B;
end

end

