function runPyramidFeatures(clct,hist_tr, hist_ts);


load 'sift_desc.mat';
load 'gs.mat';

folder ='train';
folder1 = 'test';
bins = 12;
q1=1;
q2=1;
q3=1;
q4=1;


%Read Images  and compute the color histogram
for i = 1: 1888
    %im  = im2single(imread(fullfile(folder, [num2str(i) '.jpg'])));
    %[r,c] = size(im);
    for j=1:size(train_F{i},2)
        x = train_F{i};
        if x(1,j)<= 128
            if x(2,j)<= 128
                f1 = train_D{i}';
                fq{q1} = f1(j,:)';
                q1= q1+1;
                %size(fq{1})
                
            else
                f2 = train_D{i}';
                sq{q2} = f2(j,:)';
                q2= q2+1;
            end
        else
            if x(2,j)<= 128
                f3 = train_D{i}';
                tq{q3} = f3(j,:)';
                q3= q3+1;
                
            else
                f4 = train_D{i}';
                frq{q4} = f4(j,:)';
                q4= q4+1;            
            end
        end
    end
    if(~isempty(fq))
        invec = cell2mat(fq);
        invec = invec';
        [r1,c1] = size(invec);
        size(invec)
        size(clct)
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        fh = (hist(key_clust,300))/300;
    else
        fh = zeros(1,300);
    end
    
        
    %%%%%%%%%
    if(~isempty(sq))
        invec = cell2mat(sq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        sh = (hist(key_clust,300))/300;
    else
        sh = zeros(1,300);
    end
    %%%%%%%%%%%5
    if(~isempty(tq))
        invec = cell2mat(fq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        th = (hist(key_clust,300))/300;
    else
        th = zeros(1,300);
    end
    %%%%%%%%%%%%%%%%%%%%%5
    if(~isempty(frq))
        invec = cell2mat(frq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        frh = (hist(key_clust,300))/300;
    else
        frh = zeros(1,300);
    end
        
    fnl(i,:) = [fh sh th frh hist_tr(i,:)];
%     fq = {};
%     sq = {};
%     tq = {};
%     frq = {};
    
end

size(fnl,2)
 
q1=1;
q2=1;
q3=1;
q4=1;
%%%%%%%%%%%%%%%%%%5
for i = 1: 800
    %im  = im2single(imread(fullfile(folder, [num2str(i) '.jpg'])));
    %[r,c] = size(im);
    for j=1:size(test_F{i},2)
        x1 = test_F{i};
        if x1(1,j)<= 128
            if x1(2,j)<= 128
                fs1 = test_D{i}';
                fsq{q1} = fs1(j,:)';
                q1= q1+1;
                %size(fq{1})
                
            else
                fs2 = test_D{i}';
                ssq{q2} = fs2(j,:)';
                q2= q2+1;
            end
        else
            if x1(2,j)<= 128
                fs3 = test_D{i}';
                tsq{q3} = fs3(j,:)';
                q3= q3+1;
                
            else
                fs4 = test_D{i}';
                fsrq{q4} = fs4(j,:)';
                q4= q4+1;            
            end
        end
    end
    if(~isempty(fsq))
        invec = cell2mat(fsq);
        invec = invec';
        [r1,c1] = size(invec);
        size(invec)
        size(clct)
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        fsh = (hist(key_clust,300))/300;
    else
        fsh = zeros(1,300);
    end
    
        
    %%%%%%%%%
    if(~isempty(ssq))
        invec = cell2mat(ssq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        ssh = (hist(key_clust,300))/300;
    else
        ssh = zeros(1,300);
    end
    %%%%%%%%%%%5
    if(~isempty(tsq))
        invec = cell2mat(fsq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        tsh = (hist(key_clust,300))/300;
    else
        tsh = zeros(1,300);
    end
    %%%%%%%%%%%%%%%%%%%%%5
    if(~isempty(fsrq))
        invec = cell2mat(fsrq);
        invec = invec';
        [r1,c1] = size(invec);
        dist = pdist2(invec,clct);
        for h=1:r1
            [val, I] = min(dist(h,:));
            key_clust(h) = I;
        end
        
        fsrh = (hist(key_clust,300))/300;
    else
        fsrh = zeros(1,300);
    end
        
    fn2(i,:) = [fsh ssh tsh fsrh hist_ts(i,:)];
%     fq = {};
%     sq = {};
%     tq = {};
%     frq = {};
    
end


                
        
classify_svm(fnl, fn2);
