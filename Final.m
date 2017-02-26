%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Load this part for every problem
load('E2006_matlab.mat');
lamda = 1;
dimnum = size(X,2); %x dimension number
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Problem 2 Proximal Gradient for Lasso
%(a)
eta =[2^-14, 2^-16, 2^-18, 2^-20, 2^-22];
countNonZero = zeros(5,1);
CostHis = zeros(5,50);
for etanum=1:5
    w = zeros(dimnum,1);
    compare = eta(etanum)*lamda;
    %Proximal Gradient Method
    for i=1:50
        %calculate dg
        dg1 = X * w - y;
        dg = X.' * dg1;
        wbar = w - eta(etanum) * dg;
        %soft Threshold
        w = sign(wbar).*max(abs(wbar)-compare,0);
        CostHis(etanum, i) = 0.5*norm(X * w - y,2)^2 + lamda*norm(w,1);
    end % end Proximal Gradient Method for 50 iterations
    
    for i=1:dimnum
        if w(i,1) ~= 0
            countNonZero(etanum) = countNonZero(etanum)+ 1;
        end    
    end
end % end different eta

%(b)
err = zeros(5,10);
for j=1:5
    for i = 1 : 10
        err(j,i) = CostHis(j,i) - CostHis(j,50);
    end
end
figure
hold all
axis([0 10 0 1e5]); 
plot(err(1,:),'r')
plot(err(2,:),'g')
plot(err(3,:),'k')
plot(err(4,:),'b')
plot(err(5,:),'c')

title('Problem 2 Proximal Gradient for Lasso')
xlabel('iteration');
ylabel('f(w^)-f(w*)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Promblem 3. Coordinate Descent for Lasso
wCoordi = zeros(dimnum,1);
iterNum = 50;
CostHisCoordi = zeros(1,iterNum);
h = zeros(1,dimnum);
for j=1:dimnum
     h(1,j) = sum(X(:,j).^2);
     h(1,j) = h(1,j) + 1e-50; 
end

for k=1:iterNum
    p = randperm(dimnum);
    %precompute ri
    r = X * wCoordi -y;
    
    for s =1 : dimnum
        i=p(s);
        %calculate coordinate update rule
        delta = X(:,i).' * r;
        delta = -delta / h(1,i);
        wbar(i) = wCoordi(i) + delta;
        wCoordi(i) = sign(wbar(i)).*max(abs(wbar(i))-lamda / h(1,i),0);
        if wbar(i) > lamda / h(1,i)
            r = r + (delta - lamda / h(1,i)) * X(:,i);
        elseif wbar(i) < - lamda / h(1,i)
            r = r + (delta + lamda / h(1,i)) * X(:,i);
        end    
    end    
    CostHisCoordi(1,k) = 0.5*norm(X * wCoordi - y,2)^2 + lamda*norm(wCoordi,1);
end

errCoordi = zeros(1,iterNum - 1);
for i = 1 : iterNum - 1
    errCoordi(1,i) = CostHisCoordi(1,i);
end

figure
hold all
plot(errCoordi,'r')
title('Problem 3 Coordinate Descent for Lasso')
xlabel('iteration');
ylabel('f(w^k)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Promblem 4. ADMM for Lasso
%(c) Calculate Ax=b
wADMM1 = zeros(dimnum,1);
wADMM2 = zeros(dimnum,1);
u = zeros(dimnum,1);
b = X.' * y - u + wADMM2;
wADMM1 = fast_conjugate_gradient_solve(X, b);

%(d) ADMM for Lasso
eta = [0.01,0.1,1,10,100];
iter = 50;
CostHistADMM = zeros(5,iter);
for etanum=1:5
    wADMM1 = zeros(dimnum,1);
    wADMM2 = zeros(dimnum,1);
    u = zeros(dimnum,1);
    for k =1: iter    
        %w1^k update
        b = X.' * y - u + wADMM2; %u^k-1 and w2^k-1
        wADMM1_k = fast_conjugate_gradient_solve(X, b); %record new w1^k
        %w2^k update
        inner = wADMM1 + u;
        wADMM2 = sign(inner).*max(abs(inner)-lamda,0);
        %u^k update
        u = u + eta(etanum) * (wADMM1_k - wADMM2);
        wADMM1 = wADMM1_k;
        CostHistADMM(etanum, k) = 0.5*norm(X * wADMM1 - y,2)^2 + lamda*norm(wADMM2,1) + 0.5*norm(wADMM1-wADMM2,2)^2;
    end
end 

errADMM = zeros(5,20);
for j=1:5
    for i = 1 : 20
        errADMM(j,i) = CostHistADMM(j,i);
    end
end
figure
hold all
axis([1 20 0 1e5]); 
plot(errADMM(1,:),'r')
plot(errADMM(2,:),'g')
plot(errADMM(3,:),'y')
plot(errADMM(4,:),'b')
plot(errADMM(5,:),'c')

title('Problem 4 ADMM for Lasso')
xlabel('iteration');
ylabel('f(w^k)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Problem 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for (1) and (2)
load('E2006_matlab.mat');
eta4 = 1; %for (1) and (2) 
iter2 = 450;
iter3 = 30;
iter4 = 350;
eta = 2^-18;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for (3) and (4)
%load('realsim_matlab.mat');
%eta4 = 0.1;  %for (3) and (4) 
%iter2 = 2000;
%iter3 = 60;
%iter4 = 65;
%eta = 2^-14;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization 
lamda = 1;
dimnum = size(X,2); %x dimension number
%Problem 2
Time_CostHisProx = zeros(1,iter);
w_2_recorder = zeros(dimnum,iter2);
time_2 = zeros(1,iter2);
w = zeros(dimnum,1);
compare = eta*lamda;
%Proximal Gradient Method
tic
for i=1:iter2
    %calculate dg
    dg1 = X * w - y;
    dg = X.' * dg1;
    wbar = w - eta * dg;
    %soft Threshold
    w = sign(wbar).*max(abs(wbar)-compare,0);    
    w_2_recorder(:,i) = w;
    toc;
    time_2(i) = toc;
end % end Proximal Gradient Method for 50 iterations
for i=1:iter2
    Time_CostHisProx(1, i) = 0.5*norm(X * w_2_recorder(:,i) - y,2)^2 + lamda*norm(w_2_recorder(:,i),1);
end
varlist ={'w', 'compare','dg1','wbar','i','dg'};
clear (varlist{:})
%Problem 3
wCoordi = zeros(dimnum,1);
Time_CostHisCoordi = zeros(1,iter3);
h = zeros(1,dimnum);
w_3_recorder = zeros(dimnum,iter3);
time_3 = zeros(1,iter3);
for j=1:dimnum
     h(1,j) = sum(X(:,j).^2);
     h(1,j) = h(1,j) + 1e-50; 
end
tic
for k=1:iter3
    p = randperm(dimnum);
    %precompute ri
    r = X * wCoordi -y;
    for s =1 : dimnum
        i=p(s);
        %calculate coordinate update rule
        delta = X(:,i).' * r;
        delta = -delta / h(1,i);
        wbar(i) = wCoordi(i) + delta;
        wCoordi(i) = sign(wbar(i)).*max(abs(wbar(i))-lamda / h(1,i),0);
        if wbar(i) > lamda / h(1,i)
            r = r + (delta - lamda / h(1,i)) * X(:,i);
        elseif wbar(i) < - lamda / h(1,i)
            r = r + (delta + lamda / h(1,i)) * X(:,i);
        end          
    end
    w_3_recorder(:,k) = wCoordi;
    toc;
    time_3(k) = toc;    
end
for k=1:iter3
    Time_CostHisCoordi(1,k) = 0.5*norm(X *  w_3_recorder(:,k) - y,2)^2 + lamda*norm( w_3_recorder(:,k),1);
end
varlist ={'wCoordi', 'h','p','r','delta'};
clear (varlist{:})
%Problem 4
Time_CostHistADMM = zeros(1,iter4);
w_4_recorder = zeros(dimnum,iter4);
w_4_recorder2 = zeros(dimnum,iter4);
time_4 = zeros(1,iter4);
wADMM1 = zeros(dimnum,1);
wADMM2 = zeros(dimnum,1);
u = zeros(dimnum,1);
tic
for k =1: iter4    
        %w1^k update
        b = X.' * y - u + wADMM2; %u^k-1 and w2^k-1
        wADMM1_k = fast_conjugate_gradient_solve(X, b); %record new w1^k
        %w2^k update
        inner = wADMM1 + u;
        wADMM2 = sign(inner).*max(abs(inner)-lamda,0);
        %u^k update
        u = u + eta4 * (wADMM1_k - wADMM2);
        wADMM1 = wADMM1_k;   
    w_4_recorder(:,k) = wADMM1;
    w_4_recorder2(:,k) = wADMM2;
    toc;
    time_4(k) = toc;
end
for k=1:iter4
    Time_CostHistADMM(1, k) = 0.5*norm(X * w_4_recorder(:,k) - y,2)^2 + lamda*norm(w_4_recorder2(:,k),1);% + 0.5*norm(w_4_recorder(:,k)-w_4_recorder2(:,k),2)^2;
end
varlist ={'w_4_recorder2','wADMM1','wADMM2','u','inner'};
clear (varlist{:})

figure
hold all
axis([0 30 0 3e4]); 
plot(time_2, Time_CostHisProx,'r','LineWidth',2.5)
plot(time_3, Time_CostHisCoordi,'g','LineWidth',2)
plot(time_4, Time_CostHistADMM,'b','LineWidth',2)
title('Problem 5 - Figure 1: Time vs objective function value')
%title('Problem 5 - Figure 3: Time vs objective function value for real-sim')
xlabel('time(second)');
ylabel('f(w^k)')

%(2) Time vs objective function value
predict_Pro = zeros(1, iter2);
predict_Coordi = zeros(1, iter3);
predict_ADMM = zeros(1, iter4);
ntest = size(Xt,1);

for k=1:iter2
    predict_Pro(1,k) = sum((Xt * w_2_recorder(1:150358,k) - yt).^2) / ntest;
end
for k=1:iter3
    predict_Coordi(1,k) = sum((Xt * w_3_recorder(1:150358,k) - yt).^2) / ntest;
end
for k=1:iter4
    predict_ADMM(1,k) = sum((Xt * w_4_recorder(1:150358,k) - yt).^2) / ntest;
end
figure
hold all
axis([0 30 0 2.5]); 
plot(time_2(1:iter2), predict_Pro,'r','LineWidth',3)
plot(time_3(1:iter3), predict_Coordi,'g','LineWidth',2)
plot(time_4(1:iter4), predict_ADMM,'b','LineWidth',2)
title('Problem 5 - Figure 2: Time vs prediction accuracy')
%title('Problem 5 - Figure 4: Time vs prediction accuracy for real-sim')
xlabel('time(second)');
ylabel('prediction accuracy')

%(4) Time vs objective function value
predict_Pro = zeros(1, iter2);
predict_Coordi = zeros(1, iter3);
predict_ADMM = zeros(1, iter4);
ntest = size(Xt,1);

for k=1:iter2
    predict_Pro(1,k) = sum((Xt * w_2_recorder(:,k) - yt).^2) / ntest;
end
for k=1:iter3
    predict_Coordi(1,k) = sum((Xt * w_3_recorder(:,k) - yt).^2) / ntest;
end
for k=1:iter4
    predict_ADMM(1,k) = sum((Xt * w_4_recorder(:,k) - yt).^2) / ntest;
end
figure
hold all
axis([0 30 0 1.5]); 
plot(time_2(1:iter2), predict_Pro,'r','LineWidth',3)
plot(time_3(1:iter3), predict_Coordi,'g','LineWidth',2)
plot(time_4(1:iter4), predict_ADMM,'b','LineWidth',2)
title('Problem 5 - Figure 4: Time vs prediction accuracy for real-sim')
xlabel('time(second)');
ylabel('prediction accuracy')
