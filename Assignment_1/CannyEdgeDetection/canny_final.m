clc
clear all
close all
I = imread('1.jpg');
out = cannyEdge(I);
figure, imshow(out);
