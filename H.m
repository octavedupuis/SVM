function H=H(alpha,A,u)
%  fonction renvoyant la valeur du lagrangien H
H = -1/2*alpha'*A*alpha + u'*alpha ;
end

