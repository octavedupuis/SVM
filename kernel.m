function kern=kernel(u,v)
%  fonction renvoyant la valeur du noyau K(u,v)
% u et v sont deux vecteurs (colonnes) de Rn
global ikernel % pass� en global pour alleger

if ikernel==1
    kern = u'*v;
elseif ikernel==2  % noyau Gaussien
    kern = exp(-0.001*norm(u-v)^2);
elseif ikernel==3
    kern = exp(-0.5*norm(u-v));
end

