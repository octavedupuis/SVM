function A=Compute_A(X,lab)
%  fonction qui assemble A
A = zeros(length(lab),length(lab)) ;
for i=1:length(lab)
    for j=1:length(lab)
        A(i,j) = lab(i)*lab(j)*kernel(X(:,i),X(:,j)) ;
    end
end
end
