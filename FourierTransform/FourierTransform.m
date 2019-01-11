function [] = FourierTransform(i,type)

    a = - 5;
    b = 5;
    N = 16384;
    fe = N/(b-a);
    te = 1/fe;

    xt = linspace(a,b-te,N);    % N intervalles -> il faut s'arrêter à b-T
    xf = linspace(-fe/2,fe/2-1/(b-a),N);    %idem

    pulse = 2*pi*(1/10);   %oméga = 2*PI*f
    deltaT = 0;         %décalage temporel du Dirac
    deltaF = 5;         %décalage fréquentiel pour aliasing

    f = zeros(1,N);

    switch(i)
        case 0
            f = ones(1,N);  %constante
        case 1
            for n=1:N
                f(1,n)= cos(pulse*((n-1)*te + a));  %cosinus
            end
        case 2
            for n=1:N
                f(1,n)= sin(pulse*((n-1)*te + a));  %sinus
            end
        case 3
            f(1,(N/2+1)-deltaT)=1;  %dirac en deltaT
        case 4
            for n=1:N
                f(1,n)= exp(i*(pulse*((n-1)*te + a)));  %exponentielle complexe
            end
        case 5
            for n=floor(N/2-0.1*N):floor(N/2+0.1*N)+1   %rectangle(0.1)
                f(1,n)= 1;
            end
        case 6
            for n=1:N
                f(1,n)= exp(-pi*((n-1)*te + a)^2);  %exponentielle décroissante
            end
        case 7
            act = -0.4;
            for o=1:3
                for n=floor(N/2+act*N):floor(N/2+(act+0.2)*N)+1 %créneau
                    f(1,n)= 1;
                end
                act = act + 0.3;
            end
        case 8
            for n=1:N
                f(1,n)= sin(pulse*((n-1)*te + a)) + sin((pulse+deltaF)*((n-1)*te + a)) + sin((pulse+2*deltaF)*((n-1)*te + a)) + sin((pulse+3*deltaF)*((n-1)*te + a));
            end
    end


    F = tfour(f);
    figure('units','normalized','outerposition',[0 0 1 1]);

    subplot(6,2,[1,2]);
    plot(xt,f);
    title('Echantillonage temporel')

    %Partie Réelle/Imaginaire
    if (type==0)
        subplot(3,2,3);
        plot(xf,real(F));
        title('Partie réelle de la transformée de Fourier')

        subplot(3,2,4);
        plot(xf,imag(F));
        title('Partie imaginaire de la transformée de Fourier')

    end

    %Amplitude/Phase
    if (type==1)
        subplot(3,2,3);
        plot(xf,abs(F));
        title('Module de la transformée de Fourier')

        subplot(3,2,4);
        plot(xf,mod(angle(F),pi)); %modulo pi pour n'avoir qu'une seule valeur ?
        title('Argument de la transformée de Fourier')
    end


    subplot(3,2,[5,6]);
    plot(xt,real(tfourinv(F)));
    title('Transformée inverse de Fourier')

end

