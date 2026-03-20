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
% 0) Let the user choose an image processing pipeline version
% -------------------------------------------------------------------------
availablePipelines = discover_pipeline_versions(pipelineDir);

if isempty(availablePipelines)
    error('main:noPipelinesFound', ...
        'No pipeline versions matching *_v*.m were found in %s', pipelineDir);
end

defaultPipelineIdx = find(strcmp(availablePipelines, 'oliver_v4'), 1);
if isempty(defaultPipelineIdx)
    defaultPipelineIdx = 1;
end

[selectedIdx, selectionConfirmed] = listdlg( ...
    'PromptString', 'Select an image processing script version:', ...
    'SelectionMode', 'single', ...
    'ListString', availablePipelines, ...
    'InitialValue', defaultPipelineIdx, ...
    'ListSize', [220 160], ...
    'Name', 'Select Processing Version');

if ~selectionConfirmed
    fprintf('No processing version selected. Exiting.\n');
    return;
end

selectedPipeline = availablePipelines{selectedIdx};
fprintf('Selected pipeline: %s\n', selectedPipeline);

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
processedImg = process_image(img, selectedPipeline);

% If processing changes the image, show a side-by-side montage (original left, processed right).
if ~isequal(processedImg, img)
    originalRGB  = ensureRGB(img);
    processedRGB = ensureRGB(processedImg);
    figure('Name', 'Original vs Processed', 'NumberTitle', 'off');
    montage({originalRGB, processedRGB}, 'Size', [1 2]);
    title('Original (left) and processed (right)');
end

function rgbImg = ensureRGB(inImg)
% Convert grayscale or binary images to 3-channel RGB for consistent montage display.
    if ndims(inImg) == 2 || size(inImg, 3) == 1
        rgbImg = cat(3, inImg, inImg, inImg);
    else
        rgbImg = inImg;
    end
end

function pipelineNames = discover_pipeline_versions(pipelineDir)
% Discover top-level versioned pipeline scripts, e.g. oliver_v4.m.
    pipelineFiles = dir(fullfile(pipelineDir, '*_v*.m'));
    pipelineNames = {};

    for idx = 1:numel(pipelineFiles)
        [~, pipelineName] = fileparts(pipelineFiles(idx).name);
        if ~isempty(regexp(pipelineName, '^[A-Za-z]\w*_v\d+$', 'once'))
            pipelineNames{end + 1} = pipelineName; %#ok<AGROW>
        end
    end

    pipelineNames = sort(pipelineNames);
end
