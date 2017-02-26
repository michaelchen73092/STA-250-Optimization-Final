function d = fast_conjugate_gradient_solve(datax, b)
%Using Conjugate gradient to solve Aw=b
%w : input weight(x) (No need wait to kill)
%d: optimal solution that we want to compute
%b: input that need to calculate before input
%datax: to faster compute A  = X^T * X + I
sizex = size(datax,2);
d = zeros(sizex,1);
r = b; %r0 = b - X*d but initial d = 0 so r = b;
p = r;
k=0;
while true
    FastComp1 = datax * p; 
    FastComp2 = datax.' * FastComp1 + p; % result of A * p
    rk_square = r.' * r;
    alpha_k = rk_square / (p.' * FastComp2);
    d = d + alpha_k * p;
    r = r - alpha_k * FastComp2;
    if norm(r,2)/norm(b,2) <= 1e-3
        break
    end
    belta = (r.' * r) / rk_square;
    p = r + belta * p;
    k = k+1;
end
end