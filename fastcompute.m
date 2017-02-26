function result = fastcompute(datax, datay, w, v)
    % compute result = A*v
    step1 = datax * v;
    step2 = D * step1;
    step3 = datax.' * step2;
    for i = 1 : dimnum 
        step3(i) = step3(i) + v(i); 
    end
    result = step3;
end