load 'sift_desc.mat';
load 'gs.mat';
srcFiles = dir('train/*.jpg');
traininglnth=length(srcFiles);
k=65;
for i=1:traininglnth
    
    vecdict{i}= train_D{i}(:,50:55);
end
matvecdict=cell2mat(vecdict);
vector = (reshape(matvecdict, [128, traininglnth*6]))';
size(vector,1);
%ind = randperm(size(vector,1));
ind = randperm(9440);

clct = vector(ind(1:k), :);
% size(clct)
newclust=kmean(vector,clct,k);
%clustsize=size(newclust)
for i=1:1888
    
    train_hist(i,1:k) = colhistbow(train_D{i},newclust,k);
end

for j=1:800
    test_hist(j,1:k) = colhistbow(test_D{j},newclust,k);
end
svm(train_hist, test_hist);

%pyramid(newclust,train_hist, test_hist);