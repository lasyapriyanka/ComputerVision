clear all
im=imread('lenna.pgm');
im=im2double(im);
%smoothening using gaussian
gfilter= [0 0 1 0 0;
       0 1 2 1 0;
       1 2 -16 2 1;
       0 1 2 1 0;
       0 0 1 0 0];
s=conv2(im,gfilter);
%zero crossings
[r,c]=size(s);
zc=zeros([r,c]);

for i=2:r-1
    for j=2:c-1
        if (s(i,j)>0)
             if (s(i,j+1)>=0 && s(i,j-1)<0) || (s(i,j+1)<0 && s(i,j-1)>=0)
                   zc(i,j)= s(i,j+1);
                        
            elseif (s(i+1,j)>=0 && s(i-1,j)<0) || (s(i+1,j)<0 && s(i-1,j)>=0)
                    zc(i,j)= s(i,j+1);
            elseif (s(i+1,j+1)>=0 && s(i-1,j-1)<0) || (s(i+1,j+1)<0 && s(i-1,j-1)>=0)
                  zc(i,j)= s(i,j+1);
            elseif (s(i-1,j+1)>=0 && s(i+1,j-1)<0) || (s(i-1,j+1)<0 && s(i+1,j-1)>=0)
                  zc(i,j)=s(i,j+1);
            end
                        
        end
            
    end
end
out=im2uint8(zc);
% threshold
th= out>105;

figure;
  subplot(2,2,1);imshow(im);title('Origional image');
  subplot(2,2,2);imshow(s);title('Smoothened image');
  subplot(2,2,3);imshow(out);title('Output image');
  subplot(2,2,4);imshow(th);title('Output image with threshold');

  % final result
   figure, imshow(th);

   


