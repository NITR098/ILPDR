% Copyright (c) 2014, E. Khodapanah Aghdam
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       

%************************************************************************** 
 
% Name:                ILPRT
% Authors:             E.Khodapanah Aghdam
% Date:                May 23, 2014
% Description:         Iranian LPR(License plate recognition) from
%                      thumbnail


%**************************************************************************

%Usage:
% This program recognizes the characters - containing numbers, alphabet and
% symbols - of Iranian license plates. Note that it's restricted just for 
% personal, police, taxi, government, agriculture, and general vehicle
% plates.


%**************************************************************************

clear % Clears all variables in Workspace
close all % Closes all figures (Not those opend with imtool, etc)
clc       % Clears Command Window

im=imread('I:\Tabriz Univesity\License Card Recognition/P013.jpg'); % Reads the desired thumbnail to detect
cha1=0; % Preaalocation of variable to detect colors
cha2=0; % Preaalocation of variable to detect colors
tre=0;  % Preallocation of variable to treshhold in im2bw
imx=imresize(im, [160 724]); % Resizing image to desired size 160*724
figure(1), imshow(imx)  % To show image
title('Resized original thumbnail license plate')
%pause;


%% Color detection(Yellow or White)

pla_col=cell(1,2);
figure

title('Sample White(top) and Yellow(down) Iranian plate without characters')
for i=1:1:2
    pla_col{1,i}=imread(fullfile('I:\Tabriz Univesity\License Card Recognition','C',[num2str(i) '.png']));
    subplot(2,1,i)
    imshow(pla_col{1,i})
    %pause;
end

for i=1:1:2
    cor_col(1,i)=corr2(pla_col{1,i}(:,:,3),imx(:,:,3));
end
cromo=find(max(cor_col)==cor_col);

if cromo==2
    cha2=1;
end


%% Making Black-White

im1=rgb2gray(imx);
%figure(2), imshow(im1)
%pause;
im1_bw=im2bw(im1);
figure(3), imshow(im1_bw)
%pause;


%% Color Detection(Green or Red)

p_c=sum(sum(im1_bw(:)));
[m,n]=size(im1_bw);
mn=(m*n)/2;
if p_c<mn
    imx=imcomplement(imx);
    im1=rgb2gray(imx);
    im1_bw=imcomplement(im1_bw);
    cha1=1;
end
figure(1000), imshow(mat2gray(im1_bw))
%pause;

cha1;
cha2;


%% Treshold

if cha1==0 && cha2==0 %White
    tre=0.8;
elseif cha1==0 && cha2==1 %Yellow
    tre=0.2;
elseif cha1==1 %Green (or Red)
    tre=0.3;
end


%% Edge detection and labeling

im2=edge(im1_bw,'sobel');
figure(4), imshow(im2)
%pause;
im3=imdilate(im2, strel('diamond',1));
figure(5), imshow(im3)
%pause;
im4=imfill(im3, 'holes');
figure(6), imshow(im4)
%pause;
sta=regionprops(bwlabel(im4),'Area','Extent','BoundingBox','Centroid','Orientation');


%% Plate Crop

depth=-1; % Intializing
for i=1:length([sta.Area]) 
    if sta(i).BoundingBox(2)>=depth && sta(i).Extent >= 0.82 && sta(i).Area>100000 
    depth=sta(i).BoundingBox(2); 
    end
end

% Finding the components which are at the depth "depth": 
r=[]; 
for i=1:length([sta.Area]) 
    if sta(i).BoundingBox(2)==depth && sta(i).Extent>= 0.82 && sta(i).Area>6500 
    r=[r sta(i).Area]; 
    end 
end
if(isempty(r)) 
    index=(find([sta.Area] == max([sta.Area]))); 
    else 
    % Otherwise, taking the candidate with maximum area from the 
    % filtered candidates: 
    index=(find([sta.Area] == max(r))); 
end

% Finding Plate coordinates
x1=floor(sta(index).BoundingBox(1));
x2=ceil(sta(index).BoundingBox(3));
y1=ceil(sta(index).BoundingBox(2));
y2=ceil(sta(index).BoundingBox(4));

pla_rgb=imcrop(imx(:,:,:),[x1,y1,x2,y2]); 
pla_gray=imcrop(im1(:,:),[x1,y1,x2,y2]); 

figure(7), imshow(pla_rgb);
%pause;
figure(8), imshow(pla_gray);
%pause;


%% Plate enhancment

tre;

pla_gray_adj=imadjust(pla_gray, stretchlim(pla_gray), [0 1]); % Specify lower and upper limits that can be used for contrast stretching image(J =imadjust(I,[low_in; high_in],[low_out; high_out])) 
pla_gray_double=im2double(pla_gray_adj); 
pla_bw=im2bw(pla_gray_double,tre);
figure(9), imshow(pla_bw)
%pause;


%% Border removing

pla_bw=imdilate(pla_bw,strel('line',8,90));
figure(3000), imshow(pla_bw)

pla_bw_dil_fil=imfill(pla_bw,'holes');
figure(11), imshow(pla_bw_dil_fil)

pla_enh1=xor(pla_bw_dil_fil , pla_bw); 
figure(12), imshow(pla_enh1)

pla_enh2=imdilate(pla_enh1,strel('line',8,90));
figure(4000), imshow(pla_enh2)


%% First filter_Part I

pla_op=bwareaopen(pla_enh2,1000);
figure(13), imshow(pla_op)
%pause;


%% Rotation

if abs(sta(index).Orientation)>=2 % The orientation is the angle between the horizontal line and the major axis of ellipse=angle 
    pla_bw_rot=imrotate(pla_op,-sta(index).Orientation);
    
    %figure(14), imshow(pla_bw_rot)
else
    pla_bw_rot=pla_op;
end

figure(15), imshow(pla_bw_rot)
%pause

%% First filter_Part II

pla_bw_rot(:,1:75)=0;
pla_bw_rot(:,720:724)=0;
pla_bw_rot(1:10,:)=0;
pla_bw_rot(150:160,:)=0;
figure, imshow(pla_bw_rot)
%pause;
pla_bw_rot=bwareaopen(pla_bw_rot,1000);
figure(13), imshow(pla_op)
%pause;


%% Second Filter

rawImage=1-pla_bw_rot;
maxValue = double(max(rawImage(:)));                            % Find the maximum pixel value
N = 600;                                                        % Threshold number of white pixels
boxIndex = sum(rawImage) < N*maxValue;                          % Find columns with fewer white pixels
boxImage = rawImage;                                            % Initialize the box image
boxImage(:,boxIndex) = 0;                                       % Set the indexed columns to 0 (black)
dilatedIndex = conv(double(boxIndex),ones(1,3),'same') > 0;     % Dilate the index
dilatedImage = rawImage;                                        % Initialize the dilated box image
dilatedImage(:,dilatedIndex) = 1;                               % Set the indexed columns to 0 (black)
figure(16), imshow(dilatedImage)
%pause;
pla_bw_roto=xor(dilatedImage,rawImage);
figure(17), imshow(pla_bw_roto)
%pause;

%% Third filter
 
sta2=regionprops(bwlabel(pla_bw_roto),'Area','Extent','BoundingBox','Centroid','Orientation','Image');
[L,cn]=bwlabel(pla_bw_roto);
cn;

% Omitting objects more than 8
while cn>9
    sta2=regionprops(bwlabel(pla_bw_roto),'Area','Extent','BoundingBox','Centroid','Orientation','Image');

    [L,cn]=bwlabel(pla_bw_roto);
    cn;
    sta2_Area=[sta2.Area];

    pp=min(sta2_Area);
    qq=find(sta2_Area==pp);
    [m,n]=size(L);
    for i=1:1:m
        for j=1:1:n
            if L(i,j)==qq
                L(i,j)=1;
            else
                L(i,j)=0;
            end
        end
    end


    iml=double(L).*double(pla_bw_roto);
    pla_bw_roto=xor(iml,pla_bw_roto);
    %figure(18), imshow(pla_bw_rot)
    %pause;
end


%% Character Crop

se=strel('disk',1);
pla_bw_xxx=imclose(pla_bw_roto,se);
figure(19), imshow(pla_bw_xxx)


pla_cell=cell(1,8); 
sta2=regionprops(bwlabel(pla_bw_xxx),'Area','Extent','BoundingBox','Centroid','Orientation','Image');

figure(20)
for i=1:1:8
    pla_ch=sta2(i).Image;
    res_pla_ch=imresize(pla_ch,[64 32]);
    subplot(1,8,i)
    imshow(res_pla_ch)
    pla_cell{1,i}=res_pla_ch; 
    
    % Using this part for making database or updating it
    imwrite(res_pla_ch, fullfile('I:\Tabriz Univesity\License Card Recognition','W',[num2str(i) '.png']));
end


%% Database

pla_data=cell(1,21);
for i=1:1:21
        pla_data{1,i}=imread(fullfile('I:\Tabriz Univesity\License Card Recognition','R',[num2str(i) '.png'])); 
end

%% Detection

cor=zeros(1,21);

for j=1:1:8
    for i=1:1:21
        cor(1,i)=corr2(pla_data{1,i},pla_cell{1,j});
    end
    mx=find(max(cor)==cor);

    if mx==1
        disp('1')%one
    elseif mx==2
        disp('2')%Two
    elseif mx==3
        disp('3')%Three
    elseif mx==4
        disp('4')%Four
    elseif mx==5
        disp('5')%Five
    elseif mx==6
        disp('6')%Six
    elseif mx==7
        disp('7')%Seven
    elseif mx==8
        disp('8')%Eight
    elseif mx==9
        disp('9')%Nine
    elseif mx==10 && cha1==0 && cha2==0 %White
        disp('B')%Ba
    elseif mx==10 && cha1==0 && cha2==1 %Yellow
        disp('T')%Ta
    elseif mx==10 && cha1==1 %Green
        disp('P')%Pe 
    elseif mx==11
        disp('N')%Nun
    elseif mx==12
        disp('J')%Jim
    elseif mx==13
        disp('S')%Sin
    elseif mx==14
        disp('C')%Sad
    elseif mx==15
        disp('V')%Vav
    elseif mx==16
        disp('L')%Lam
    elseif mx==17
        disp('I')%Ain
    elseif mx==18
        disp('K')%Kaf
    elseif mx==19
        disp('A')%Alef
    elseif mx==20
        disp('Q')%Qaf
    elseif mx==21
        disp('Dis')%Disabled
    end
end
