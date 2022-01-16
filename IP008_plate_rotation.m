%**************************************************************************
% © By: E. Khodapanah Aghdam, 2014
% Usage: License Plate Detection and Recognition of Iranian Personal Plates
%**************************************************************************

clear     % Clears all Variables of the Workspace
close all % Closes all figures
clc       % Clears the Command Window

im=imread('I:/Tabriz Univesity/License Card Recognition/P022.jpg'); % Reads the sample image
imx=imresize(im, [640 NaN]); % Resizes the image to desired size
figure, imshow(imx)
%pause;
im1=rgb2gray(imx); % Changes the image from rgb(color image) to gray
%figure, imshow(im1)
%pause;
im1_bw=im2bw(im1); % Changes the image to black-white
figure, imshow(im1_bw)
%pause;
im2=edge(im1,'sobel'); % Finds the edges by derivation
figure, imshow(im2)
%pause;
im3=imdilate(im2, strel('diamond',1)); % Expands the edges
figure, imshow(im3)
%pause;
im4=imfill(im3, 'holes'); % Fills the holes
figure, imshow(im4)
%pause;
sta=regionprops(bwlabel(im4),'Area','Extent','BoundingBox','Centroid','Orientation'); % Finds the black-white image properties



%% Plate Crop

depth=-1; % Initializing
for i=1:length([sta.Area]) 
    if sta(i).BoundingBox(2)>=depth && sta(i).Extent >= 0.80 && sta(i).Area>10000 
    depth=sta(i).BoundingBox(2); 
    end
end 

% Finding the components which are at the depth "depth": 
r=[]; 
for i=1:length([sta.Area]) 
    if sta(i).BoundingBox(2)==depth && sta(i).Extent>= 0.80 && sta(i).Area>10000
    r=[r sta(i).Area]
    end 
end
if(isempty(r)) 
index=(find([sta.Area] == max([sta.Area])));
else 
% otherwise, taking the candidate with maximum area from the 
% filtered candidates: 
index=(find([sta.Area] == max(r))); 
end

x1=floor(sta(index).BoundingBox(1));
x2=ceil(sta(index).BoundingBox(3));
y1=ceil(sta(index).BoundingBox(2)); 
y2=ceil(sta(index).BoundingBox(4));

pla_rgb=imcrop(imx(:,:,:),[x1,y1,x2,y2]); 
pla_gray=imcrop(im1(:,:),[x1,y1,x2,y2]); 

figure, imshow(pla_rgb);
%pause;
figure, imshow(pla_gray);
%pause;



%% plate enhancment

pla_gray_adj=imadjust(pla_gray, stretchlim(pla_gray), [0 1]); % specify lower and upper limits that can be used for contrast stretching image(J =imadjust(I,[low_in; high_in],[low_out; high_out])) 
pla_gray_double=im2double(pla_gray_adj); 
pla_bw=im2bw(pla_gray_double);%im2bw(I, level) converts the grayscale image I to a binary image 
figure, imshow(pla_bw) 
%pause;



%% Border removing

pla_bw_dil=imdilate(pla_bw,strel('line',1,0)); 
figure, imshow(pla_bw_dil)
%pause;

pla_bw_dil_fil=imfill(pla_bw_dil,'holes');
figure, imshow(pla_bw_dil_fil)
%pause;

pla_enh=xor(pla_bw_dil_fil , pla_bw_dil); 
figure, imshow(pla_enh)
%pause;
  

   
%% First filter_Part I

pla_op=bwareaopen(pla_enh,100);
figure, imshow(pla_op)
%pause;



%% Rotation

if abs(sta(index).Orientation)>=2 %The orientation is the angle between the horizontal line and the major axis of ellipse=angle 
    pla_bw_rot=imrotate(pla_op,-sta(index).Orientation);
    
    %figure, imshow(pla_bw)
    %pause;

else
    pla_bw_rot=pla_op;
end

figure, imshow(pla_bw_rot)



%% First filter_Part II

pla_bw_roto=bwareaopen(pla_bw_rot, 100);
figure, imshow(pla_bw_roto)
%pause;



%% Second Filter

rawImage=1-pla_bw_roto;
maxValue = double(max(rawImage(:)));     % Find the maximum pixel value
N = 300;                                  % Threshold number of white pixels
boxIndex = sum(rawImage) < N*maxValue;   %# Find columns with fewer white pixels
boxImage = rawImage;                     %# Initialize the box image
boxImage(:,boxIndex) = 0;                %# Set the indexed columns to 0 (black)
dilatedIndex = conv(double(boxIndex),ones(1,5),'same') > 0;  %# Dilate the index
dilatedImage = rawImage;                 %# Initialize the dilated box image
dilatedImage(:,dilatedIndex) = 1;        %# Set the indexed columns to 0 (black)
figure, imshow(dilatedImage)
%pause;
pla_bw_roto=xor(dilatedImage,rawImage);
figure, imshow(pla_bw_roto)
%pause;



 %% Third filter
 
sta2=regionprops(bwlabel(pla_bw_roto),'Area');
[L,cn]=bwlabel(pla_bw_roto);
cn;
sta2_Area=[sta2.Area];
pp=min(sta2_Area);
qq=find(sta2_Area==pp);
[t,s]=size(qq);
loop=0;
while cn>=9

    if s~=1 % It's here because of same size of noises.
        qq=qq(1,1);
    end
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
    figure, imshow(pla_bw_rot)
    loop=loop+1;
    cn;
    if loop>=30
        error('Internal error: Loop exceeds the limitation. Check While loop.')
    end
 
    sta2=regionprops(bwlabel(pla_bw_roto),'Area');

    [L,cn]=bwlabel(pla_bw_roto);
    sta2_Area=[sta2.Area];

    pp=min(sta2_Area);
    qq=find(sta2_Area==pp);
    [t,s]=size(qq);
end
figure, imshow(pla_bw_rot)
%pause;



%% Character Crop

se=strel('disk',1);
pla_bw_xxx=imclose(pla_bw_roto,se);
figure(2222), imshow(pla_bw_xxx)
%pause;

pla_cell=cell(1,8); 
sta2=regionprops(bwlabel(pla_bw_xxx),'Area','Extent','BoundingBox','Centroid','Orientation','Image');

figure
for i=1:1:8
    pla_ch=sta2(i).Image;
    res_pla_ch=imresize(pla_ch,[64 32]);
    subplot(1,8,i)
    imshow(res_pla_ch)
    %pause;
    pla_cell{1,i}=res_pla_ch;
    
    % This is onlyfor extracting characters to create the data base.
    imwrite(res_pla_ch, fullfile('I:\Tabriz Univesity\License Card Recognition','W',[num2str(i) '.png'])); 
end
    
%% Database

pla_data=cell(1,22);
for i=1:1:22
        pla_data{1,i}=imread(fullfile('I:\Tabriz Univesity\License Card Recognition','R',[num2str(i) '.png'])); 
end


%% Character Recognition

cor=zeros(1,22);

for j=1:1:8
    for i=1:1:22
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
    elseif mx==10
        disp('B')%Ba
    elseif mx==11
        disp('N')%Nun
    elseif mx==12
        disp('J')%Jim
    elseif mx==13
        disp('S')%Sin
    elseif mx==14
        disp('Z')%Sad
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
        disp('D')%Dal
    elseif mx==22
        disp('Dis')%Disabled
    end
end
