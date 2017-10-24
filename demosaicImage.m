function output = demosaicImage(im, method)
% DEMOSAICIMAGE computes the color image from mosaiced input
%   OUTPUT = DEMOSAICIMAGE(IM, METHOD) computes a demosaiced OUTPUT from
%   the input IM. The choice of the interpolation METHOD can be 
%   'baseline', 'nn', 'linear', 'adagrad'. 
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji
%

switch lower(method)
    case 'baseline'
        output = demosaicBaseline(im);
    case 'nn'
        output = demosaicNN(im);         % Implement this
    case 'linear'
        output = demosaicLinear(im);     % Implement this
    case 'adagrad'
        output = demosaicAdagrad(im);    % Implement this
    case 'transformation'
        output = demosaicTransformation(im); % Extra Implementation
    case 'log_transformation'
        output = demosaicLogTransformation(im); % Extra Implementation
end

%--------------------------------------------------------------------------
%                          Baseline demosacing algorithm. 
%                          The algorithm replaces missing values with the
%                          mean of each color channel.
%--------------------------------------------------------------------------
function mosim = demosaicBaseline(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
redValues = im(1:2:imageHeight, 1:2:imageWidth);
meanValue = mean(mean(redValues));
mosim(:,:,1) = meanValue;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
blueValues = im(2:2:imageHeight, 2:2:imageWidth);
meanValue = mean(mean(blueValues));
mosim(:,:,3) = meanValue;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
greenValues = mosim(mask > 0);
meanValue = mean(greenValues);
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = meanValue;
mosim(:,:,2) = greenChannel;

%--------------------------------------------------------------------------
%                           Nearest neighbour algorithm
%--------------------------------------------------------------------------
function mosim = demosaicNN(im)
%mosim = demosaicBaseline(im);

[imageWidth,imageHeight]=size(im);

%creating masks for the bayer pattern
bayer_red = repmat([1 0; 0 0], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_blue = repmat([0 0; 0 1], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_green = repmat([0 1; 1 0], ceil(imageWidth/2),ceil(imageHeight/2));

%truncating the extra pixels at the edges
if(mod(imageWidth,2))==1
   bayer_red(size(bayer_red,1),:)=[];
   bayer_blue(size(bayer_blue,1),:)=[];
   bayer_green(size(bayer_green,1),:)=[];
end
if(mod(imageHeight,2)==1)
   bayer_red(:,size(bayer_red,2))=[];
   bayer_blue(:,size(bayer_blue,2))=[];
   bayer_green(:,size(bayer_green,2))=[];
end
    
%extracting the red, green and blue components of the image using the mask

red_image = im.*bayer_red;
blue_image = im.*bayer_blue;
green_image = im.*bayer_green;

%deducing the green pixels at missing points
green = green_image+imfilter(green_image,[0 1]);

%deducing the red pixels at missing points
redValue=im(1:2:imageWidth,1:2:imageHeight);
meanRed = mean(mean(redValue));
%red@blue
red_1 = imfilter(red_image, [0 0;0 1], meanRed);
%red@green
red_2 = imfilter(red_image, [0 1;1 0], meanRed);
%combine
red = red_image + red_1 +red_2;

%deducing the blue pixels at missing points
blueValue=im(1:2:imageWidth,1:2:imageHeight);
meanBlue = mean(mean(blueValue));
%blue@red
blue_1 = imfilter(blue_image, [0 0;0 1], meanBlue);
%blue@green
blue_2 = imfilter(blue_image, [0 1;1 0], meanBlue);
%combine
blue = blue_image + blue_1 +blue_2;

mosim(:,:,1) = red;
mosim(:,:,2) = green;
mosim(:,:,3) = blue;
%--------------------------------------------------------------------------
%                           Linear interpolation
%--------------------------------------------------------------------------
function mosim = demosaicLinear(im)
% mosim = demosaicBaseline(im);
% mosim = repmat(im, [1 1 3]);
%
[imageWidth,imageHeight]=size(im);

%creating masks for the bayer pattern
bayer_red = repmat([1 0; 0 0], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_blue = repmat([0 0; 0 1], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_green = repmat([0 1; 1 0], ceil(imageWidth/2),ceil(imageHeight/2));

%truncating the extra pixels at the edges
if(mod(imageWidth,2))==1
   bayer_red(size(bayer_red,1),:)=[];
   bayer_blue(size(bayer_blue,1),:)=[];
   bayer_green(size(bayer_green,1),:)=[];
end
if(mod(imageHeight,2)==1)
   bayer_red(:,size(bayer_red,2))=[];
   bayer_blue(:,size(bayer_blue,2))=[];
   bayer_green(:,size(bayer_green,2))=[];
end
    
%extracting the red, green and blue components of the image using the mask
red_image = im.*bayer_red;
blue_image = im.*bayer_blue;
green_image = im.*bayer_green;

%deducing the green pixels at missing points
green = green_image + imfilter(green_image, [0 1 0;1 0 1; 0 1 0]/4);

%deducing the red pixels at missing points
red_1 = imfilter(red_image, [1 0 1;0 0 0;1 0 1]/4);
red_2 = imfilter(red_image, [0 1 0;1 0 1;0 1 0]/2);
red = red_image+red_1+red_2;

%deducing the blue pixels at missing points
blue_1 = imfilter(blue_image, [1 0 1;0 0 0;1 0 1]/4);
blue_2 = imfilter(blue_image, [0 1 0;1 0 1;0 1 0]/2);
blue = blue_image+blue_1+blue_2;

mosim(:,:,1) = red;
mosim(:,:,2) = green;
mosim(:,:,3) = blue;
%--------------------------------------------------------------------------

%                           Adaptive gradient
%--------------------------------------------------------------------------
function mosim = demosaicAdagrad(im)
 mosim = demosaicBaseline(im);
%
[imageWidth,imageHeight]=size(im);

%creating masks for the bayer pattern
bayer_red = repmat([1 0; 0 0], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_blue = repmat([0 0; 0 1], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_green = repmat([0 1; 1 0], ceil(imageWidth/2),ceil(imageHeight/2));

%truncating the extra pixels at the edges
if(mod(imageWidth,2))==1
   bayer_red(size(bayer_red,1),:)=[];
   bayer_blue(size(bayer_blue,1),:)=[];
   bayer_green(size(bayer_green,1),:)=[];
end
if(mod(imageHeight,2)==1)
   bayer_red(:,size(bayer_red,2))=[];
   bayer_blue(:,size(bayer_blue,2))=[];
   bayer_green(:,size(bayer_green,2))=[];
end
    
%extracting the red, green and blue components of the image using the mask
red_image = im.*bayer_red;
blue_image = im.*bayer_blue;
green_image = im.*bayer_green;

%deducing the green pixels at the missing points
green = green_image + imfilter(green_image, [0 1 0; 1 0 1; 0 1 0]);
for x = 3:2:(imageWidth-2)
    for y = 3:2:(imageHeight-2)
        horizontal_gradient = abs((red_image(x,y-2)+red_image(x,y+2))/2 - red_image(x,y));
        vertical_gradient = abs((red_image(x-2,y)+red_image(x+2,y))/2 - red_image(x,y));
        if(horizontal_gradient<vertical_gradient)
            mosim(x,y,2)=(green(x,y-1)+green(x,y+1))/2;
        elseif (horizontal_gradient>vertical_gradient)
             mosim(x,y,2)=(green(x-1,y)+green(x+1,y))/2;
        else
            mosim(x,y,2)=(green(x-1,y)+green(x+1,y)+green(x,y-1)+green(x,y+1))/4;
        end       
    end
end
for x = 4:2:(imageWidth-2)
    for y = 4:2:(imageHeight-2)
        horizontal_gradient = abs((blue_image(x,y-2)+blue_image(x,y+2))/2 - blue_image(x,y));
        vertical_gradient = abs((blue_image(x-2,y)+blue_image(x+2,y))/2 - blue_image(x,y));
        if(horizontal_gradient<vertical_gradient)
            mosim(x,y,2)=(green(x,y-1)+green(x,y+1))/2;
        elseif (horizontal_gradient>vertical_gradient)
             mosim(x,y,2)=(green(x-1,y)+green(x+1,y))/2;
        else
            mosim(x,y,2)=(green(x-1,y)+green(x+1,y)+green(x,y-1)+green(x,y+1))/4;
        end           
    end
end

%deducing the blue pixels at missing points
blue_1 = imfilter(blue_image, [1 0 1;0 0 0;1 0 1]/4);
blue_2 = imfilter(blue_image, [0 1 0;1 0 1;0 1 0]/2);
blue = blue_image+blue_1+blue_2;

%deducing the red pixels at missing points
red_1 = imfilter(red_image, [1 0 1;0 0 0;1 0 1]/4);
red_2 = imfilter(red_image, [0 1 0;1 0 1;0 1 0]/2);
red = red_image+red_1+red_2;

mosim(:,:,1) = red;
mosim(:,:,3) = blue;
%--------------------------------------------------------------------------

%                           Transformed Color Spaces - Linear
%--------------------------------------------------------------------------
function mosim = demosaicTransformation(im)
 %mosim = demosaicBaseline(im);
 
[imageWidth,imageHeight]=size(im);

%creating masks for the bayer pattern
bayer_red = repmat([1 0; 0 0], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_blue = repmat([0 0; 0 1], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_green = repmat([0 1; 1 0], ceil(imageWidth/2),ceil(imageHeight/2));

%truncating the extra pixels at the edges
if(mod(imageWidth,2))==1
   bayer_red(size(bayer_red,1),:)=[];
   bayer_blue(size(bayer_blue,1),:)=[];
   bayer_green(size(bayer_green,1),:)=[];
end
if(mod(imageHeight,2)==1)
   bayer_red(:,size(bayer_red,2))=[];
   bayer_blue(:,size(bayer_blue,2))=[];
   bayer_green(:,size(bayer_green,2))=[];
end
    
%extracting the red, green and blue components of the image using the mask

red_image = im.*bayer_red;
blue_image = im.*bayer_blue;
green_image = im.*bayer_green;

%deducing the green pixels at missing points
green = green_image + imfilter(green_image, [0 1 0;1 0 1; 0 1 0]/4);

%checking for pixel values to be zero and correcting it
min_green = min(min(green(green~=0)));
green(green==0) = min_green;

%transforming the images
red_image = red_image./green;
red_image = red_image.*bayer_red;

blue_image = blue_image./green;
blue_image = blue_image.*bayer_blue;

%deducing the red pixels at missing points
red_1 = imfilter(red_image, [1 0 1;0 0 0;1 0 1]/4);
red_2 = imfilter(red_image, [0 1 0;1 0 1;0 1 0]/2);
red = red_image+red_1+red_2;

%deducing the blue pixels at missing points
blue_1 = imfilter(blue_image, [1 0 1;0 0 0;1 0 1]/4);
blue_2 = imfilter(blue_image, [0 1 0;1 0 1;0 1 0]/2);
blue = blue_image+blue_1+blue_2;

%applying inverse transformation
mosim(:,:,1) = red.*green;
mosim(:,:,2) = green;
mosim(:,:,3) = blue.*green;
%--------------------------------------------------------------------------

%                           Transformed Color Spaces - Logarithmic
%--------------------------------------------------------------------------
function mosim = demosaicLogTransformation(im)
 %mosim = demosaicBaseline(im);
 
[imageWidth,imageHeight]=size(im);

%creating masks for the bayer pattern
bayer_red = repmat([1 0; 0 0], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_blue = repmat([0 0; 0 1], ceil(imageWidth/2),ceil(imageHeight/2));
bayer_green = repmat([0 1; 1 0], ceil(imageWidth/2),ceil(imageHeight/2));

%truncating the extra pixels at the edges
if(mod(imageWidth,2))==1
   bayer_red(size(bayer_red,1),:)=[];
   bayer_blue(size(bayer_blue,1),:)=[];
   bayer_green(size(bayer_green,1),:)=[];
end
if(mod(imageHeight,2)==1)
   bayer_red(:,size(bayer_red,2))=[];
   bayer_blue(:,size(bayer_blue,2))=[];
   bayer_green(:,size(bayer_green,2))=[];
end
    
%extracting the red, green and blue components of the image using the mask

green_image = im.*bayer_green;
blue_image = im.*bayer_blue;
red_image = im.*bayer_red;

%deducing the green pixels at missing points
green = green_image + imfilter(green_image, [0 1 0;1 0 1; 0 1 0]/4);

%checking for pixel values to be zero and correcting it
min_green = min(min(green(green~=0)));
green(green==0) = min_green;

min_blue = min(min(blue_image(blue_image~=0)));
blue_image(blue_image==0) = min_blue;

min_red = min(min(red_image(red_image~=0)));
red_image(red_image==0) = min_red;

%transforming the images
red_image = log(red_image./green);
red_image = red_image.*bayer_red;

blue_image = log(blue_image./green);
blue_image = blue_image.*bayer_blue;

%deducing the red pixels at missing points
red_1 = imfilter(red_image, [1 0 1;0 0 0;1 0 1]/4);
red_2 = imfilter(red_image, [0 1 0;1 0 1;0 1 0]/2);
red = red_image+red_1+red_2;

%deducing the blue pixels at missing points
blue_1 = imfilter(blue_image, [1 0 1;0 0 0;1 0 1]/4);
blue_2 = imfilter(blue_image, [0 1 0;1 0 1;0 1 0]/2);
blue = blue_image+blue_1+blue_2;

%applying the inverse transformation
mosim(:,:,1) = exp(red).*green;
mosim(:,:,2) = green;
mosim(:,:,3) = exp(blue).*green;
