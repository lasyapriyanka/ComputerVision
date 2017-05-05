function svm(tr, ts)

load 'gs.mat';
%svmmodel = fit

for i=1:8
    keys = find(train_gs~=i);
    keys1 = find(test_gs~=i);
    new_gs1 = test_gs;
    new_gs = train_gs;
    new_gs(keys)=0;
    new_gs1(keys1)=0;
    %optimset('svmtrain');
    options = optimset('Display', 'off', 'MaxIter',100000);
    %optnew = optimset(options);
    tic;
    svstruct = svmtrain(tr, new_gs','kktviolationlevel', 0.6,'options',options,'tolkkt',0.01);
    tm_tr(i) = toc;
    tic;
    group = svmclassify(svstruct,ts);
    tm_ts(i) = toc;
    acc(i) =sum(group'==new_gs1)/800;
    C{i} = confusionmat(new_gs1,group');
    
  
    
end
acc_result = mean(acc)
train_time = mean(tm_tr)
classify_time = mean(tm_ts)

%svstruct = svmtrain(tr, train_gs');
end


