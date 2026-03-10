% Main entrypoint: ensure the dataset is available, let the user pick an
% image, then load and display it. Extend or pipe the loaded image into
% `process_image.m` for further processing.

clear; clc; close all;

% Resolve dataset location relative to this file (../dataset).
projectRoot = fileparts(fileparts(mfilename('fullpath')));
datasetDir  = fullfile(projectRoot, 'dataset');

% Add `image_processing/` to the project path so that we can call functions inside of it.
pipelineDir = fullfile(fileparts(mfilename('fullpath')), 'image_processing');
addpath(pipelineDir);

% -------------------------------------------------------------------------
% 1) Ensure dataset is present (download + unzip if missing)
% -------------------------------------------------------------------------
if ~exist(datasetDir, 'dir')
    fprintf('Dataset folder not found. Downloading dataset...\n');
    url     = 'https://www.zemris.fer.hr/projects/LicensePlates/english/baza_slika.zip';
    zipPath = fullfile(projectRoot, 'dataset.zip');
    
    % Download zip
    
    fprintf('  -> Saving to %s\n', zipPath);
    websave(zipPath, url);
    
    % Extract and clean up
    if ~exist(datasetDir, 'dir')
        mkdir(datasetDir);
    end
    fprintf('  -> Extracting into %s\n', datasetDir);
    unzip(zipPath, datasetDir);
    delete(zipPath);
    
    fprintf('Dataset downloaded and extracted.\n');
else
    fprintf('Dataset already present at %s\n', datasetDir);
end

% -------------------------------------------------------------------------
% 2) Let the user pick an image from the dataset folder
% -------------------------------------------------------------------------
[filename, pathname] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif)'}, ...
    'Select an image to process', ...
    datasetDir);

if isequal(filename, 0)
    fprintf('No file selected. Exiting.\n');
    return;
end

imgPath = fullfile(pathname, filename);
fprintf('Selected image: %s\n', imgPath);

% -------------------------------------------------------------------------
% 3) Load and show the chosen image
% -------------------------------------------------------------------------
img = imread(imgPath);

figure('Name', 'Selected Image', 'NumberTitle', 'off');
imshow(img);
title(sprintf('Selected image: %s', filename), 'Interpreter', 'none');

% -------------------------------------------------------------------------
% 4) Hook into processing pipeline (edit process_image.m to add steps)
% -------------------------------------------------------------------------
processedImg = process_image(img);

% If processing changes the image, show it.
if ~isequal(processedImg, img)
    figure('Name', 'Processed Image', 'NumberTitle', 'off');
    imshow(processedImg);
    title('Processed output');
end
