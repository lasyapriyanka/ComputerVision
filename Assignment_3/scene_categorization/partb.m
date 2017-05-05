path = '/home/ground/down_camera/scene_categorization';
load('sift_desc.mat');
load('gs.mat');
vocabsize = 500;
dframe = [];
dscript = [];
srcFiles = dir('train/*.jpg');
traininglnth=length(srcFiles);
srcFiles1 = dir('test/*.jpg');
testlnth=length(srcFiles1);
sct=[];
npimg = ceil(10000/traininglnth);
for i=1:traininglnth
    des = train_D{i};
    r = randperm(size(des,2));
    df = des(:,r);
    sct = [sct df(:,1:npimg)];
end
size(sct);
[clust,p] = kmeans(vocabsize,double(sct'));
%size(clusters)
% for m=1:1888
%      trainfeat(m,1:300) = colhistbow(train_D{m},newclust);
% end
% for b=1:800
%     testfeat(b,1:300) = colhistbow(test_D{b},newclust);
% end
% for m=1:1888
%     
%     hist_tr(m,1:300) = colhistbow(train_D{m},newclust);
% end
% 
% for b=1:800
%     hist_ts(b,1:300) = colhistbow(test_D{b},newclust);
% end
% % 
% % size(hist_tr)
% % size(hist_ts)
% 
% for i=1:800
%     mindist = pdist2(hist_ts(i,:),hist_tr);
%     for j=1:14
%         [V, I(j)]=min(mindist);
%         mindist(I(j))=max(mindist);
%         I(j) = train_gs(1,I(j));
%     end
%     val = mode(I);
%     result_gs(1,i)= val;
% end
% acc =sum(result_gs==test_gs)/800;


for i=1:traininglnth
    
    desc = double(train_D{i});
    des = double(desc'); 
    dessize = size(des,1);
    clustsize = size(clust,1);
    cNorm = sum(clust.^2, 2);
    xNorm = sum((des').^2, 1);
    aMat = repmat(cNorm,1, dessize) + repmat(xNorm, clustsize, 1) - 2*clust*des';
    [minValue, idx] = min(aMat,[], 1);
    trainfeat(i,:) = histc(idx,1:clustsize);
    trainfeat(i,:) = trainfeat(i,:)/sum(trainfeat(i,:));
end

for i=1:testlnth
    desc = double(test_D{i});
    des = double(desc'); 
    clustsize = size(clust,1);
    dessize = size(des,1);
    cNorm = sum(clust.^2, 2);
    xNorm = sum((des').^2, 1);
    aMat = repmat(cNorm,1, dessize) + repmat(xNorm, clustsize, 1) - 2*clust*des';
    [minValue, IDX] = min(aMat,[], 1);
    testfeat(i,:) = histc(IDX,1:clustsize);
    testfeat(i,:) = testfeat(i, :)/sum(testfeat(i,:));
    
end
labels = zeros(testlnth,1);

for i=1:testlnth
    vect = testfeat(i,:);
    dist = sum(abs(trainfeat - repmat(vect,[traininglnth 1])),2);
    
    %dist = sqrt(sum((trainfeat - repmat(vect,[imgnum 1])).^2,2));
    [~,id] = min(dist);
    
    labels(i) = train_gs(id);
end

accuracy = numel(find(labels == test_gs'))/testlnth
C = confusionmat(test_gs,labels)
    


  