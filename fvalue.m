function f = fvalue(datax, theta, datay) 
%FVALUE Summary of this function goes here
%   Detailed explanation goes here
 f = 0.5 * norm(datax * theta - datay,2)^2 + norm(theta,1); %for Lasso
 %f = 0.5 * norm(datax * theta - datay,2)^2 + 0.5 * norm(theta,2)^2; %for
 %GradientforRidge
 %f = 0.5*(datax*theta - datay).'*(datax*theta - datay) + 0.5*(theta).'*theta;
end

