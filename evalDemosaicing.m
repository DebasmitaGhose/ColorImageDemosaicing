% Entry code for evaluating demosaicing algorithms
% The code loops over all images and methods, computes the error and
% displays them in a table.
% 
%
% This code is part of:
%
%   CMPSCI 670: Computer Vision
%   University of Massachusetts, Amherst
%   Instructor: Subhransu Maji
%
% Path to your data directory
dataDir = fullfile('C:','Users','User','Documents','Umass Amherst', 'Semester 1', 'COMPSCI 670 - Computer Vision','Mini Projects','2','p2_data','data','demosaic');

% Path to your output directory
outDir = fullfile('C:','Users','User','Documents','Umass Amherst', 'Semester 1', 'COMPSCI 670 - Computer Vision','Mini Projects','2','p2_code','output','demosaic');
if ~exist(outDir, 'file')
    mkdir(outDir);
end

% List of images
imageNames = {'balloon.jpeg',	'cat.jpg',	'ip.jpg','puppy.jpg','squirrel.jpg', ...
              'pencils.jpg',	'house.png', 'light.png', 'sails.png', 'tree.jpeg'};
numImages = length(imageNames);

% List of methods you have to implement          
methods = {'baseline', 'nn', 'linear', 'adagrad', 'transformation','log_transformation'};
numMethods = length(methods);

% Global variables
display = false;
error = zeros(numImages, numMethods);

% Loop over methods and print results
fprintf([repmat('-',[1 100]),'\n']); 
fprintf('# \t image \t\t baseline \t nn \t\t linear \t adagrad \t linear_trans \t log_trans\n'); 
fprintf([repmat('-',[1 100]),'\n']); 
for i = 1:numImages,
    fprintf('%i \t %s ', i, imageNames{i});
    for j = 1:numMethods, 
        thisImage = fullfile(dataDir, imageNames{i});
        thisMethod = methods{j};
        [error(i,j), colorIm] = runDemosaicing(thisImage, thisMethod, display);
        fprintf('\t %f ', error(i,j)); 
        
        % Write the output
        outfileName = fullfile(outDir, [imageNames{i}(1:end-5) '-' thisMethod '-dmsc.jpg']);
        imwrite(colorIm, outfileName);
        
    end
    fprintf('\n');
end

% Compute average errors
fprintf([repmat('-',[1 100]),'\n']); 
fprintf(' \t %s ', 'average');
for j = 1:numMethods, 
        fprintf('\t %f ', mean(error(:,j)));     
end
fprintf('\n');
fprintf([repmat('-',[1 100]),'\n']); 
