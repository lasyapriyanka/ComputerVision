function [kclust] = kmean(vecs,clct,k)

dist1 = 1000;
for iter =1:50
    dist = pdist2(vecs, clct);
    if(max(max(abs(dist1-dist)))==0)
        break;
    end
    dist1 = dist;
    %size(dist)
    
    for p=1:11328
        [val, I1] = min(dist(p,:));
        key_clust(p) = I1;
        
    end
    %key_clust
    %clct(1,:)
    %size(key_clust)
    %for h=1:300
    for h=1:k
        
        group = find(key_clust==h);
        
        %size(group)
        clct(h,:)=mean(vecs(group(:),:));
        
        
    end
end
kclust=clct;
end

