close all;
clear all;
I=imread('test1.jpg');
I=rgb2gray(I);
ID=im2double(I);
figure;imshow(ID);
%%padding
IG=zeros(size(ID,1)+2,size(ID,2)+2);
for i=drange(1:size(ID,1))
    for j=drange(1:size(ID,2))
        IG(i+1,j+1)=ID(i,j);
    end
end
%%R matrix
R=ID;
%%constant k
k=0.04;
%%computing R
for i=drange(1:size(ID,1)-2)
    for j=drange(1:size(ID,2)-2)
        %%taking window
        W=IG(i:i+2,j:j+2);
        Ix=W;
        Iy=W;
        %%computing gradients
        Ix(1,1)=W(1,1);
        Ix(1,2)=W(1,2)-W(1,1);
        Ix(1,3)=W(1,3)-W(1,2);
        Ix(2,1)=W(2,1);
        Ix(2,2)=W(2,2)-W(2,1);
        Ix(2,3)=W(2,3)-W(2,2);
        Ix(3,1)=W(3,1);
        Ix(3,2)=W(3,2)-W(3,1);
        Ix(3,3)=W(3,3)-W(3,2);
        Iy(1,1)=W(1,1);
        Iy(1,2)=W(1,2);
        Iy(1,3)=W(1,3);
        Iy(2,1)=W(2,1)-W(1,1);
        Iy(2,2)=W(2,2)-W(2,1);
        Iy(2,3)=W(2,3)-W(1,3);
        Iy(3,1)=W(3,1)-W(2,1);
        Iy(3,2)=W(3,2)-W(2,2);
        Iy(3,3)=W(3,3)-W(2,3);
        %%fing A,B,c
        A=sum(sum(Ix.*Ix));
        B=sum(sum(Ix.*Iy));
        C=sum(sum(Iy.*Iy));
        %%defining Harris or homography matrix
        H=[A B;B C];
        %%finding R at each point
        R(i,j)=det(H)-k*(trace(H)^2);
    end
end
%%plotting histogram of R
figure;hist(R);
%%taking input of threshold
T=input('Threshold = ');
%%thresholding
threshold=find(R<T);
R(threshold)=0;
%%non maxima supression
R=R>imdilate(R, [1 1 1; 1 0 1; 1 1 1]);
figure;imshow(R);