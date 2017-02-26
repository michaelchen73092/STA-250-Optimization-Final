%Gradient Descent is applied to ridge regression problem
load('E2006_matlab.mat');
lamda = 1;
dimnum = size(X,2); %x dimension number

% Initialize values
%(f)
w = zeros(dimnum, 1);
numIters = 200;
CostHistorye = zeros(numIters, 1);
eta = zeros(numIters,1);
for iter=1:numIters
    g1 = X * w - y;
    g = (X).'* g1 + w;
    eta_e=1;
    f_xsubg = fvalue(X, w-eta_e*g, y);
    f_x = fvalue(X, w, y);
    while f_xsubg - f_x > -0.001*eta_e*(g).'*g 
        eta_e = 0.5*eta_e;
        f_xsubg = fvalue(X, w-eta_e*g, y);
    end
    eta(iter, 1) = eta_e;
    w = w - eta_e*g;
    CostHistorye(iter,1) = fvalue(X, w, y);
end
countNonZero = 0;
for i=1:dimnum
    if w(i,1) ~= 0
       countNonZero = countNonZero+ 1;
    end    
end

err = zeros(180,1);
for i = 1 : 180
   err(i,1) = log(CostHistorye(i,1)- CostHistorye(numIters,1));
end    
figure;
plot(err);
title('Gradient descent for ridge')
xlabel('x');
ylabel('log(f(x^k)-f(x*))');
