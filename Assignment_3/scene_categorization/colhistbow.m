function histvec = colhistbow(invec, clct,numk)

%invec = cell2mat(invec);
invec = invec';
[r,c] = size(invec);
dist = pdist2(invec,clct);
for i=1:r
        [val, I] = min(dist(i,:));
        key_clust(i) = I;
end

histvec = hist(key_clust,numk);
histvec = histvec/numk;
end

