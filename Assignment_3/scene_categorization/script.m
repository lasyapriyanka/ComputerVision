%%
%A
clear all
clc
load('gs.mat');
K = 1;
bins = 10;
datasetPath = 'train';
imgDir = dir(fullfile(datasetPath, '*.jpg'));
numImg = length(imgDir);
h_feat = zeros(numImg,bins*3);
h_feat_lab = zeros(numImg,1);
for i=1:numImg
    imgName = imgDir(i).name;
    y = strtok(imgName,'.');
    h_feat_lab(i) = train_gs(str2num(y));
    im = imread(fullfile(datasetPath, imgName));
    z1 = imhist(im(:,:,1),bins);
    z1 = double(z1)./sum(z1,1);
    z2 = imhist(im(:,:,2),bins);
    z2 = double(z2)./sum(z2,1);
    z3 = imhist(im(:,:,3),bins);
    z3 = double(z3)./sum(z3,1);
    z = [z1;z2;z3];
    h_feat(i,:) = z';
end
cm = zeros(8,8);
datasetPath = 'test';
imgDir = dir(fullfile(datasetPath, '*.jpg'));
numImgtest = length(imgDir);
for i=1:numImgtest
    temp = zeros(8,1);
    imgName = imgDir(i).name;
    y = strtok(imgName,'.');
    lb = test_gs(str2num(y));
    im = imread(fullfile(datasetPath, imgName));
    z1 = imhist(im(:,:,1),bins);
    z1 = double(z1)./sum(z1,1);
    z2 = imhist(im(:,:,2),bins);
    z2 = double(z2)./sum(z2,1);
    z3 = imhist(im(:,:,3),bins);
    z3 = double(z3)./sum(z3,1);
    z = [z1;z2;z3];
    [D,I] = pdist2(h_feat,z','euclidean','Smallest',K);
    for j=1:length(I)
        temp(h_feat_lab(I(j))) = temp(h_feat_lab(I(j))) + 1; 
    end
    [maxi,lab] = max(temp);
    cm(lb,lab) = cm(lb,lab) + 1; 
end
cat_accuracy = sum(diag(cm))/sum(cm(:));



%%
%B
clear all
clc
K = 100;
kn = 20;
np = 6;
cm = zeros(8,8);
load('sift_desc.mat');
load('gs.mat');
bag_feat = zeros(1888,K);
sift_feat = zeros(128,1888*np);
wor = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    ord = randperm(size(im,2));
    sift_feat(:,np*(i-1)+1:np*(i)) = im(:,ord(1:np));
end
ord1 = randperm(size(sift_feat,2));
wor(:,:) = sift_feat(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor',sift_feat','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat(:,z);
        temp1= sum(temp')/size(temp,2);
        wor(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)/sum(temp,2);
    bag_feat(i,:) = temp;
end
for i=1:800
    temps = zeros(8,1);
    im = double(test_D{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)/sum(temp,2);
    lb = test_gs(i);
    [D,I] = pdist2(bag_feat,temp,'euclidean','Smallest',kn);
    for j=1:length(I)
        temps(train_gs(I(j))) = temps(train_gs(I(j))) + 1; 
    end
    [maxi,lab] = max(temps);
    cm(lb,lab) = cm(lb,lab) + 1; 
end
cat_accuracy = sum(diag(cm))/sum(cm(:));

%%
%C
clear all
clc
K = 100;
np = 6;
cm = zeros(8,8);
load('sift_desc.mat');
load('gs.mat');
bag_feat = zeros(1888,K);
sift_feat = zeros(128,1888*np);
wor = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    ord = randperm(size(im,2));
    sift_feat(:,np*(i-1)+1:np*(i)) = im(:,ord(1:np));
end
ord1 = randperm(size(sift_feat,2));
wor(:,:) = sift_feat(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor',sift_feat','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat(:,z);
        temp1= sum(temp')/size(temp,2);
        wor(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)/sum(temp,2);
    bag_feat(i,:) = temp;
end
%%
svmn1 = fitcsvm(bag_feat,double(train_gs==1)');
svmn2 = fitcsvm(bag_feat,double(train_gs==2)');
svmn3 = fitcsvm(bag_feat,double(train_gs==3)');
svmn4 = fitcsvm(bag_feat,double(train_gs==4)');
svmn5 = fitcsvm(bag_feat,double(train_gs==5)');
svmn6 = fitcsvm(bag_feat,double(train_gs==6)');
svmn7 = fitcsvm(bag_feat,double(train_gs==7)');
svmn8 = fitcsvm(bag_feat,double(train_gs==8)');
%%
opt.MaxIter = Inf;
svm1 = svmtrain(bag_feat,double(train_gs==1)','options',opt);
svm2 = svmtrain(bag_feat,double(train_gs==2)','options',opt);
svm3 = svmtrain(bag_feat,double(train_gs==3)','options',opt);
svm4 = svmtrain(bag_feat,double(train_gs==4)','options',opt);
svm5 = svmtrain(bag_feat,double(train_gs==5)','options',opt);
svm6 = svmtrain(bag_feat,double(train_gs==6)','options',opt);
svm7 = svmtrain(bag_feat,double(train_gs==7)','options',opt);
svm8 = svmtrain(bag_feat,double(train_gs==8)','options',opt);
%%
cm = zeros(8,8);
for i=1:800
    temps = zeros(8,2);
    label = zeros(8,1);
    im = double(test_D{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)/sum(temp,2);
    lb = test_gs(i);
%     temps(1) = svmclassify(svm1,temp);
%     temps(2) = svmclassify(svm2,temp);
%     temps(3) = svmclassify(svm3,temp);
%     temps(4) = svmclassify(svm4,temp);
%     temps(5) = svmclassify(svm5,temp);
%     temps(6) = svmclassify(svm6,temp);
%     temps(7) = svmclassify(svm7,temp);
%     temps(8) = svmclassify(svm8,temp);
    [label(1),temps(1,:)] = predict(svmn1,temp);
    [label(2),temps(2,:)] = predict(svmn2,temp);
    [label(3),temps(3,:)] = predict(svmn3,temp);
    [label(4),temps(4,:)] = predict(svmn4,temp);
    [label(5),temps(5,:)] = predict(svmn5,temp);
    [label(6),temps(6,:)] = predict(svmn6,temp);
    [label(7),temps(7,:)] = predict(svmn7,temp);
    [label(8),temps(8,:)] = predict(svmn8,temp);
    [maxi,lab] = max(temps(:,2));
    cm(lb,lab) = cm(lb,lab) + 1; 
end
cat_accuracy = sum(diag(cm))/sum(cm(:));





%%
%D
clear all
clc
K = 100;
np = 6;
cm = zeros(8,8);
load('sift_desc.mat');
load('gs.mat');
bag_feat = zeros(1888,K);
sift_feat = zeros(128,1888*np);
wor = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    ord = randperm(size(im,2));
    sift_feat(:,np*(i-1)+1:np*(i)) = im(:,ord(1:np));
end
ord1 = randperm(size(sift_feat,2));
wor(:,:) = sift_feat(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor',sift_feat','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat(:,z);
        temp1= sum(temp')./size(temp,2);
        wor(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    bag_feat(i,:) = temp;
end

K = 100;
np = 3;
bag_feat1 = zeros(1888,K);
sift_feat1 = zeros(128,1888*np);
wor1 = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    ord = randperm(length(idx1));
    sift_feat1(:,np*(i-1)+1:np*(i)) = im(:,idx1(ord(1:np)));
end
ord1 = randperm(size(sift_feat1,2));
wor1(:,:) = sift_feat1(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor1',sift_feat1','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat1(:,z);
        temp1= sum(temp')./size(temp,2);
        wor1(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor1',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    bag_feat1(i,:) = temp;
end

K = 100;
np = 3;
bag_feat2 = zeros(1888,K);
sift_feat2 = zeros(128,1888*np);
wor2 = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    ord = randperm(length(idx1));
    sift_feat2(:,np*(i-1)+1:np*(i)) = im(:,idx1(ord(1:np)));
end
ord1 = randperm(size(sift_feat2,2));
wor2(:,:) = sift_feat2(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor2',sift_feat2','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat2(:,z);
        temp1= sum(temp')./size(temp,2);
        wor2(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor2',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    bag_feat2(i,:) = temp;
end

K = 100;
np = 3;
bag_feat3 = zeros(1888,K);
sift_feat3 = zeros(128,1888*np);
wor3 = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    ord = randperm(length(idx1));
    sift_feat3(:,np*(i-1)+1:np*(i)) = im(:,idx1(ord(1:np)));
end
ord1 = randperm(size(sift_feat3,2));
wor3(:,:) = sift_feat3(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor3',sift_feat3','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat3(:,z);
        temp1= sum(temp')./size(temp,2);
        wor3(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor3',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    bag_feat3(i,:) = temp;
end

K = 100;
np = 3;
bag_feat4 = zeros(1888,K);
sift_feat4 = zeros(128,1888*np);
wor4 = zeros(128,K);
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    ord = randperm(length(idx1));
    sift_feat4(:,np*(i-1)+1:np*(i)) = im(:,idx1(ord(1:np)));
end
ord1 = randperm(size(sift_feat4,2));
wor4(:,:) = sift_feat4(:,ord1(1:K));
nerr = 0;
lerr = -2;
while abs(nerr-lerr)>0
    [D,I] = pdist2(wor4',sift_feat4','euclidean','Smallest',1);
    lerr = nerr;
    nerr = sum(D(:));
    for i=1:K
        z = find(I==i);
        temp = sift_feat4(:,z);
        temp1= sum(temp')./size(temp,2);
        wor4(:,i) = temp1';
    end
end
for i=1:1888
    im = double(train_D{i});
    im_coord = double(train_F{i});
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor4',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    bag_feat4(i,:) = temp;
end

%%
bag_feat_final = [bag_feat bag_feat1 bag_feat2 bag_feat3 bag_feat4];
%bag_feat_final = cat(1, bag_feat, bag_feat1, bag_feat2, bag_feat3, bag_feat4);  

%%
opt.MaxIter = Inf;
svm1 = svmtrain(bag_feat_final,double(train_gs==1)','options',opt);
svm2 = svmtrain(bag_feat_final,double(train_gs==2)','options',opt);
svm3 = svmtrain(bag_feat_final,double(train_gs==3)','options',opt);
svm4 = svmtrain(bag_feat_final,double(train_gs==4)','options',opt);
svm5 = svmtrain(bag_feat_final,double(train_gs==5)','options',opt);
svm6 = svmtrain(bag_feat_final,double(train_gs==6)','options',opt);
svm7 = svmtrain(bag_feat_final,double(train_gs==7)','options',opt);
svm8 = svmtrain(bag_feat_final,double(train_gs==8)','options',opt);

%%

svmn1 = fitcsvm(bag_feat_final,double(train_gs==1)');
svmn2 = fitcsvm(bag_feat_final,double(train_gs==2)');
svmn3 = fitcsvm(bag_feat_final,double(train_gs==3)');
svmn4 = fitcsvm(bag_feat_final,double(train_gs==4)');
svmn5 = fitcsvm(bag_feat_final,double(train_gs==5)');
svmn6 = fitcsvm(bag_feat_final,double(train_gs==6)');
svmn7 = fitcsvm(bag_feat_final,double(train_gs==7)');
svmn8 = fitcsvm(bag_feat_final,double(train_gs==8)');
% svmna1 = svmn1.Trained{1};
% svmna2 = svmn2.Trained{1};
% svmna3 = svmn3.Trained{1};
% svmna4 = svmn4.Trained{1};
% svmna5 = svmn5.Trained{1};
% svmna6 = svmn6.Trained{1};
% svmna7 = svmn7.Trained{1};
% svmna8 = svmn8.Trained{1};
svmna1 = compact(svmn1);
svmna2 = compact(svmn2);
svmna3 = compact(svmn3);
svmna4 = compact(svmn4);
svmna5 = compact(svmn5);
svmna6 = compact(svmn6);
svmna7 = compact(svmn7);
svmna8 = compact(svmn8);

%%

for i=1:800
    temps = zeros(8,2);
    label = zeros(8,1);
    im = double(test_D{i});
    im_coord = double(test_F{i});
    [D,I] = pdist2(wor',im','euclidean','Smallest',1);
    temp = hist(I,K);
    temp = double(temp)./sum(temp,2);
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor1',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp1 = double(temp)./sum(temp,2);
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>0)&(im_coord(2,:)'<=128));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor2',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp2 = double(temp)./sum(temp,2);
    idx = double((im_coord(1,:)'>0)&(im_coord(1,:)'<=128)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor3',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp3 = double(temp)./sum(temp,2);
    idx = double((im_coord(1,:)'>128)&(im_coord(1,:)'<=256)&(im_coord(2,:)'>128)&(im_coord(2,:)'<=256));
    idx1 = find(idx==1);
    im1 = im(:,idx1);
    [D,I] = pdist2(wor1',im1','euclidean','Smallest',1);
    temp = hist(I,K);
    temp4 = double(temp)./sum(temp,2);
    temp_final = [temp temp1 temp2 temp3 temp4];  
    lb = test_gs(i);
%     temps(1) = svmclassify(svm1,temp_final);
%     temps(2) = svmclassify(svm2,temp_final);
%     temps(3) = svmclassify(svm3,temp_final);
%     temps(4) = svmclassify(svm4,temp_final);
%     temps(5) = svmclassify(svm5,temp_final);
%     temps(6) = svmclassify(svm6,temp_final);
%     temps(7) = svmclassify(svm7,temp_final);
%     temps(8) = svmclassify(svm8,temp_final);
    [label(1),temps(1,:)] = predict(svmna1,temp_final);
    [label(2),temps(2,:)] = predict(svmna2,temp_final);
    [label(3),temps(3,:)] = predict(svmna3,temp_final);
    [label(4),temps(4,:)] = predict(svmna4,temp_final);
    [label(5),temps(5,:)] = predict(svmna5,temp_final);
    [label(6),temps(6,:)] = predict(svmna6,temp_final);
    [label(7),temps(7,:)] = predict(svmna7,temp_final);
    [label(8),temps(8,:)] = predict(svmna8,temp_final);
    [maxi,lab] = max(label.*temps(:,2));
    cm(lb,lab) = cm(lb,lab) + 1; 
end
cat_accuracy = sum(diag(cm))/sum(cm(:));



%%
