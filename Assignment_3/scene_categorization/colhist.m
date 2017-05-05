function [ fhist ] = colhist( img,bins)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[h,w,p]=size(img);
n=h*w;
% % r=reshape(img(:,:,1),n,1);
% % b=reshape(img(:,:,2),n,1);
% % g=reshape(img(:,:,3),n,1);
% r_hist=hist(double(r),bins);
% b_hist=hist(double(b),bins);
% g_hist=hist(double(g),bins);
% %Normalize the histogram
% r_hist=r_hist-min(r_hist);
% b_hist=b_hist-min(b_hist);
% g_hist=g_hist-min(g_hist);
% %Divide by Max
% r1=max(r_hist);
% b1=max(b_hist);
% g1=max(g_hist);
% %normalized value
% r_hist=r_hist/r1;
% b_hist=b_hist/b1;
% g_hist=g_hist/g1;
% %check the Sizes
% % size(r_hist)
% % size(b_hist)
% % size(g_hist)
% fhist=[r_hist';b_hist';g_hist']';
%size(hist);
data = reshape(img,n, 1);
histogram = hist(double(data),bins);
histogram = histogram - min(histogram);
fhist = histogram ./ max(histogram);
% [r, c] = size(img);
% 
% data = reshape(img, r*c, 1);
% 
% histogram = hist(double(data), bins);
% 
% histogram = histogram - min(histogram);
% fhist = histogram ./ max(histogram);
% %histogram = histogram.^2;
end

