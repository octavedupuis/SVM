% Séparateurs à Vaste Marge pour la Classification
% Cas linéaire, non linéaire, marges "souples"

clear all
close all
global ikernel  % type de noyau choisi.     1: produit scalaire dans Rp (linéaire)
                                                         % 2: noyau gaussien
                                                            % Vous pouvez
                                                            % en ajouter
                                                            % d'autres

 load 'mensur' X lab     % chargement des données
 ikernel=2;             % choix du type de noyau
 marge_souple= true;  % choix marge souple ou pas
 X = X(1:2,:) ;
 C_souple= 20;          % coef souplesse de la marge souple 
% bornes pour le dessin 2D
xmin=min(X(1,:));
ymin=min(X(2,:));
xmax=max(X(1,:));
ymax=max(X(2,:));

na=length(X); % nombre de points (d'apprentissage)

% dessin des points dans R^2
subplot(1,2,1)
for i=1:na
    if lab(i)==1
        plot(X(1,i),X(2,i),'o','linewidth',2)
        hold on
    else
        plot(X(1,i),X(2,i),'x','linewidth',2,'markersize',12)
        hold on
    end
end
axis([xmin xmax ymin ymax])
grid
axis equal
hold on
pause

% Coeur du Programme - Méthode d'UZAWA
% assemblage matrice de la forme quadratique du problème dual



% gradient à pas constant pour problème dual
% On vous laisse quelques valeurs pour les paramettres reglables... à vous de voir
alph0=0.5*ones(na,1); % point de départ (0, ou autre): réglable
pasgrad=5e-3;         % pas du gradient : parametre réglable
u=ones(na,1);         % vecteur de 1
crit_arret=1;         % initialisation critere d'arret
npas=0;               % comptage du nombre de pas
npas_max=100000;      % garde-fou convergence : on arrete si le nombre d'iterations est trop grand
epsi=1e-5;            % seuil convergence
A = Compute_A(X,lab); % On assemble la matrice A
% boucle de gradient projeté
% A vous de jouer!
alph = alph0 ;
alph_hat = 0 ;
vit_conv = [] ;
critere = 2*epsi ;
while(npas <= npas_max & critere > epsi)
    alph0 = alph ;
    alph_hat = alph + pasgrad*gradH(alph, A, u) ; 
    alph = alph_hat - (lab'*alph_hat)*lab/(norm(lab)*norm(lab)) ;
    alph = max(0, alph) ;
    if marge_souple==true
        alph = min(alph,C_souple) ;
    end
    npas = npas + 1 ;
    vit_conv(npas) = norm(alph0-alph) ; 
    critere = norm(alph0-alph) ;
end

% recherche des points supports
epsi=1e-5;
Points_support=find(alph>epsi)'; % voir help find
if isempty(Points_support) == 1
    fprintf("Impossible de réaliser la SVM \n") ;
end

if marge_souple == true
    Points_support = find(alph>epsi & abs(C_souple - alph)>epsi) ;
end    

moy_b = zeros(length(Points_support), 1) ;
for i=1:length(Points_support)
    moy_b(i) = 1/lab(Points_support(i)) - prodw(X,lab,alph,X(:,Points_support(i))) ;
end
% on ne se sert plus directement de w, mais de son produit scalaire 
% avec un vecteur, defini par un noyeau : fonction prodw

% Calcul de b (on le calcule pour chaque pt support, puis on moyenne)
% chaque pt support devrait fournit le même b (mais aux erreurs
% numériques près, c'est pouquoi on moyenne)


b=mean(moy_b);   %  on moyenne les b des points supports
grid


% Fin du coeur du programme


%calcul et tracé des isovaleurs
xp=xmin:0.2:xmax;   % création d'une grille pour les besoins de contour
yp=ymin:0.2:ymax;
npx=length(xp);
npy=length(yp);
for i=1:npx
    for j=1:npy
        ps=prodw(X,lab,alph,[xp(i),yp(j)]'); % calcul de <w,x> + b sur une grille
        V(i,j)=ps + b ;  % on n'a pas besoin explicitement de w, mais de son 
                         % produit scalaire avec tout vecteur qu'on
                         % encapsule dans prodw (utilisation noyau si cas non
                         % linéaire
    end
end
hold on
contour(xp,yp,V',[-1 0 1],'linewidth',2,'color','r')
axis([xmin xmax ymin ymax])
title(' Suport Vector Machine')
grid

subplot(1,2,2)
plot(log(vit_conv),'linewidth',2)
xlabel(' nombre itérations')
ylabel(' log (ecart)')
title('comportement convergence')
grid

function NP = new_point(x)
if prodw(X,lab,alph,x) + b > 0
    NP = 1 ;
end
if prodw(X,lab,alph,x) + b <= 0
    NP = 0 ;
end
end






