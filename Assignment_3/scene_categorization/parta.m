% img=imread('1.jpg');
% figure(1);imshow(img);
bins=10;
load 'gs.mat';
srcFiles = dir('train/*.jpg');
traininglnth=length(srcFiles);
for i = 1 : traininglnth
    im  = im2single(imread(fullfile('train', [num2str(i) '.jpg'])));
%     r=im(:,:,1);
%     b=im(:,:,2);
%     g=im(:,:,3);
    r_train(i, :)  = colhist(im(:,:,1),bins);
    b_train(i, :) = colhist(im(:,:,2),bins);
    g_train(i, :)  = colhist(im(:,:,3),bins);
    tfeat(i,:)= [r_train(i, :)' ; b_train(i, :)' ; g_train(i, :)']';

end
%size(tfeat)
srcFiles1 = dir('test/*.jpg');
testlnth=length(srcFiles1);
for i=1:testlnth
    im1 = im2single(imread(fullfile('test', [num2str(i) '.jpg'])));

   
    r_test(i, :)  = colhist(im1(:,:,1),bins);
    g_test(i,:) = colhist(im1(:,:,2),bins);
    b_test(i,:)  = colhist(im1(:,:,3),bins);
    
    testfeat(i,:)= [r_test(i, :)' ; g_test(i, :)' ; b_test(i, :)']';

end
%size(testfeat)
for i=1:testlnth
    nearestdist = pdist2(testfeat(i,:),tfeat);
    [val,ind(1)]=min(nearestdist);
    nearestdist(ind(1))=max(nearestdist);
    ind(1)=train_gs(1,ind(1));
    [val,ind(2)]=min(nearestdist);
    nearestdist(ind(2))=max(nearestdist);
    ind(2)=train_gs(1,ind(2));
    [val,ind(3)]=min(nearestdist);
    nearestdist(ind(3))=max(nearestdist);
    ind(3)=train_gs(1,ind(3));
    [val,ind(4)]=min(nearestdist);
    nearestdist(ind(4))=max(nearestdist);
    ind(4)=train_gs(1,ind(4));
  
    valueofind = mode(ind);
    result_gs(1,i)= valueofind;
end

accuracy =sum(result_gs==test_gs)/testlnth
C= confusionmat(test_gs, result_gs);

