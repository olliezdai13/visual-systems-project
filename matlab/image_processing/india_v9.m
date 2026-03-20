function processedImg = india_v9(img)
% Boilerplate wrapper for the provided india_v9 licence plate reader script.

    validateattributes(img, {'uint8', 'uint16', 'single', 'double'}, ...
        {'nonempty', 'size', [NaN NaN 3]}, mfilename, 'img');

    tempImagePath = [tempname, '.png'];
    imwrite(img, tempImagePath);
    cleanupObj = onCleanup(@() cleanup_temp_file(tempImagePath)); %#ok<NASGU>

    processedImg = read_license_plate(tempImagePath);
end

function processedImg = read_license_plate(imagePath)
%% =========================================================
%  read_license_plate.m  –  Single-file licence plate reader
%  Usage: press F5, or type run('read_license_plate.m')
% =========================================================
close all; clc;

%imagePath = "C:\Users\44793\Documents\VS Project\visual-systems-project\dataset\040603\P1010003.jpg";
%[file,path]  = uigetfile('*.jpg'); %user dialog to find csv pressure file
% = imread([path, file]); %read in cvs file 

%% STAGE 1 – Load & preprocess
[img, imgGray, imgEq] = load_and_preprocess(imagePath);

%% STAGE 2 – Detect plate region
plateCrop = detect_plate_region(img, imgGray, imgEq);

if isempty(plateCrop)
    error('Could not locate a licence plate in this image.');
end

%% STAGE 3 – Binarise the cropped plate
BW_plate = binarise_plate(plateCrop);

%% STAGE 4 – Segment character band + OCR
segment_and_ocr(BW_plate, plateCrop);

    processedImg = plateCrop;
end

%% =========================================================
%  HELPER FUNCTIONS
%% =========================================================

% ---------------------------------------------------------
function [img, imgGray, imgEq] = load_and_preprocess(imagePath)
% Load image, convert to greyscale, enhance contrast (CLAHE)

    img     = imread(imagePath);
    imgGray = rgb2gray(img);
    imgEq   = adapthisteq(imgGray, 'NumTiles', [8 8], 'ClipLimit', 0.02);

    figure('Name','Stage 1 - Preprocess');
    montage({img, imgGray, imgEq}, ...
        'Size', [1 3], 'BorderSize', [4 4], ...
        'BackgroundColor', [0.15 0.15 0.15]);
    title('Stage 1: Original  |  Greyscale  |  CLAHE','Color','w','FontSize',12);
    set(gcf,'Color',[0.15 0.15 0.15]);
    fprintf('[Stage 1] Loaded: %d x %d px\n', size(img,2), size(img,1));
end


% ---------------------------------------------------------
function plateCrop = detect_plate_region(img, imgGray, imgEq)
% Try Hough, yellow mask, white mask — pick best aspect ratio

    [h, w] = size(imgGray);

    %% 1. Sobel + Hough
    BW_edges = edge(imgEq, 'sobel');
    [H, theta, rho] = hough(BW_edges);
    P     = houghpeaks(H, 9, 'Threshold', ceil(0.25 * max(H(:))));
    lines = houghlines(BW_edges, theta, rho, P, 'FillGap', 25, 'MinLength', 60);

    plateCropHough = [];
    if ~isempty(lines)
        horLines = lines(abs([lines.theta]) < 25);
        if length(horLines) >= 2
            allPts = [[horLines.point1]; [horLines.point2]];
            xMin = max(1, min(allPts(:,1)) - 15);
            xMax = min(w, max(allPts(:,1)) + 15);
            yMin = max(1, min(allPts(:,2)) - 15);
            yMax = min(h, max(allPts(:,2)) + 15);
            plateCropHough = imcrop(img, [xMin, yMin, xMax-xMin, yMax-yMin]);
        end
    end

    %% 2. Yellow mask (French / UK rear plates)
    imgHSV = rgb2hsv(img);
    H_ch = imgHSV(:,:,1);  S_ch = imgHSV(:,:,2);  V_ch = imgHSV(:,:,3);

    yellowMask = (H_ch >= 0.10 & H_ch <= 0.20) & (S_ch > 0.45) & (V_ch > 0.45);
    yellowMask = imclose(yellowMask, strel('rectangle', [5 25]));
    yellowMask = imfill(yellowMask, 'holes');
    yellowMask = imopen(yellowMask, strel('rectangle', [3 10]));
    plateCropYellow = blob_crop(img, yellowMask, 2.0, 800);

    %% 3. White mask (UK front / European front plates)
    whiteMask = (S_ch < 0.20) & (V_ch > 0.80);
    whiteMask = imclose(whiteMask, strel('rectangle', [5 25]));
    whiteMask = imfill(whiteMask, 'holes');
    whiteMask = imopen(whiteMask, strel('rectangle', [3 10]));
    plateCropWhite = blob_crop(img, whiteMask, 2.0, 1000);

    %% Montage - masks
    figure('Name','Stage 2 - Colour Masks');
    montage({repmat(uint8(BW_edges)*255,[1 1 3]), ...
             repmat(uint8(yellowMask)*255,[1 1 3]), ...
             repmat(uint8(whiteMask)*255,[1 1 3])}, ...
        'Size',[1 3],'BorderSize',[4 4],'BackgroundColor',[0.1 0.1 0.1]);
    title('Stage 2: Sobel Edges  |  Yellow Mask  |  White Mask','Color','w','FontSize',12);
    set(gcf,'Color',[0.1 0.1 0.1]);

    %% Pick best candidate (closest aspect ratio to 4.5:1)
    candidates = {plateCropHough, plateCropYellow, plateCropWhite};
    names      = {'Hough', 'Yellow mask', 'White mask'};
    scores     = cellfun(@aspect_score, candidates);
    [bestScore, bestIdx] = min(scores);

    if bestScore == Inf
        plateCrop = [];
        warning('No valid plate candidate found.');
        return;
    end

    plateCrop = candidates{bestIdx};
    fprintf('[Stage 2] Winner: %s  (score=%.2f)\n', names{bestIdx}, bestScore);

    validCrops = candidates(~cellfun(@isempty, candidates));
    if ~isempty(validCrops)
        figure('Name','Stage 2 - Plate Candidates');
        montage(validCrops,'BorderSize',[4 4],'BackgroundColor',[0.1 0.1 0.1]);
        title(sprintf('Stage 2: Candidates  (chosen: %s)', names{bestIdx}), ...
              'Color','w','FontSize',12);
        set(gcf,'Color',[0.1 0.1 0.1]);
    end
end


% ---------------------------------------------------------
function BW_plate = binarise_plate(plateCrop)
% Otsu threshold -> remove noise blobs -> close gaps

    plateGray = rgb2gray(plateCrop);
    T         = graythresh(plateGray);
    BW_otsu   = imbinarize(plateGray, T);
    BW_clean  = bwareaopen(BW_otsu, 80);
    BW_clean  = ~bwareaopen(~BW_clean, 60);
    BW_plate  = imclose(BW_clean, strel('rectangle',[2 2]));

    fprintf('[Stage 3] Otsu threshold = %.3f\n', T);

    figure('Name','Stage 3 - Binarise');
    montage({plateGray, BW_otsu, BW_clean, BW_plate}, ...
        'Size',[1 4],'BorderSize',[4 4],'BackgroundColor',[0.1 0.1 0.1]);
    title(sprintf('Stage 3: Grey  |  Otsu T=%.2f  |  Noise removed  |  Closed', T), ...
          'Color','w','FontSize',11);
    set(gcf,'Color',[0.1 0.1 0.1]);
end


% ---------------------------------------------------------
function segment_and_ocr(BW_plate, plateCrop)
% SEGMENT_AND_OCR
%   1. Row projection  -> find the character band (tallest bright run)
%   2. Sharpen + upscale the band for better OCR accuracy
%   3. Run ocr() on the whole band as a single line
%   4. Clean result to plate format and display

    [~, plateW] = size(BW_plate);

    %% ── Row projection: find tallest continuous band ─────────────────────
    whitePerRow = sum(BW_plate, 2);
    rowThresh   = max(whitePerRow) * 0.25;
    rowRegions  = whitePerRow > rowThresh;

    rowStarts = find(diff([0;        rowRegions]) ==  1);
    rowEnds   = find(diff([rowRegions; 0])        == -1);

    if isempty(rowStarts)
        warning('No character rows found - check binarisation.');
        return;
    end

    [~, bestRow] = max(rowEnds - rowStarts);
    upperRow     = rowStarts(bestRow);
    lowerRow     = rowEnds(bestRow);

    BW_band   = BW_plate(upperRow:lowerRow, :);
    colorBand = plateCrop(upperRow:lowerRow, :, :);

    %% ── Column projection (for display only) ─────────────────────────────
    whitePerCol = sum(BW_band, 1);
    colThresh   = max(whitePerCol) * 0.10;

    %% ── Projection plots ─────────────────────────────────────────────────
    figure('Name','Stage 4 - Projections');
    subplot(2,1,1);
    plot(1:length(whitePerRow), whitePerRow, 'b', 'LineWidth', 1.5); hold on;
    yline(rowThresh, 'r--', 'Threshold');
    xline(upperRow, 'g-', 'Top'); xline(lowerRow, 'm-', 'Bottom');
    xlabel('Row (top to bottom)'); ylabel('White pixels');
    title('Row Projection - character band between green/magenta lines');
    grid on; axis tight;

    subplot(2,1,2);
    plot(1:plateW, whitePerCol, 'b', 'LineWidth', 1.5); hold on;
    yline(colThresh, 'r--', 'Threshold');
    xlabel('Column (left to right)'); ylabel('White pixels');
    title('Column Projection of Character Band');
    grid on; axis tight;

    %% ── Connected components (for bounding box display only) ─────────────
    CC    = bwconncomp(~BW_band);   % invert: find dark character blobs
    stats = regionprops(CC, 'BoundingBox', 'Area');

    bandH    = size(BW_band, 1);
    bandW    = size(BW_band, 2);
    bandArea = bandH * bandW;

    charBlobs = [];
    for i = 1:length(stats)
        bb = stats(i).BoundingBox;
        ar = bb(4) / max(bb(3), 1);
        if stats(i).Area > 0.005 * bandArea && ...
           stats(i).Area < 0.25  * bandArea && ...
           ar > 0.8 && bb(4) > 0.40*bandH  && ...
           bb(3) < 0.30*bandW
            charBlobs(end+1) = i; %#ok<AGROW>
        end
    end

    %% ── Annotated band ───────────────────────────────────────────────────
    figure('Name','Stage 4 - Character Band');
    imshow(colorBand); hold on;
    for i = 1:length(charBlobs)
        bb = stats(charBlobs(i)).BoundingBox;
        rectangle('Position', bb, 'EdgeColor','r','LineWidth',1.5);
    end
    title(sprintf('Stage 4: Character Band  (%d blobs detected)', length(charBlobs)));
    hold off;

    %% ── Prepare binary band for OCR ──────────────────────────────────────
    % Upscale 3x — OCR engines work better on larger text
    bw = imresize(BW_band, 3.0, 'nearest');

    % Convert logical -> uint8 BEFORE imsharpen (it cannot accept logical)
    bw = im2uint8(bw);

    % Sharpen to crisp up character edges after resize
    bw = imsharpen(bw, 'Amount', 1.5, 'Radius', 1.0, 'Threshold', 0);

    % Montage: what we feed to OCR
    figure('Name','Stage 4 - OCR Input');
    imshow(bw);
    title('Stage 4: Band fed to OCR (3x upscaled + sharpened)');

    %% ── OCR ──────────────────────────────────────────────────────────────
    fprintf('\n[Stage 4] Running OCR...\n');
    try
        res = ocr(bw, ...
            'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', ...
            'LayoutAnalysis', 'line');
        raw = res.Text;

        disp('Raw OCR output:');
        disp(raw);

        % ── Clean to plate format ─────────────────────────────────────────
        % Remove anything that isn't A-Z or 0-9
        cleaned = upper(regexprep(raw, '[^A-Z0-9]', ''));

        % Common OCR confusions on number plates:
        %   O/0/Q are ambiguous — keep as-is (plate has both letters and digits)
        %   I/1/L are ambiguous — keep as-is for same reason
        % Only normalise if you know the plate format (e.g. all-letter suffix)
        % Uncomment lines below if needed for your specific plate type:
        %cleaned = regexprep(cleaned, '[0OQ]', 'O');
        %cleaned = regexprep(cleaned, '[1IL]', 'I');
        %cleaned = regexprep(cleaned, '[5S]',  'S');

        % ── Print result ──────────────────────────────────────────────────
        if length(cleaned) >= 7
            fprintf('\n==========================================\n');
            fprintf('  DETECTED PLATE:  ** %s **\n', cleaned(1:7));
            fprintf('==========================================\n');
        elseif length(cleaned) >= 5
            fprintf('\nPartial plate detected: %s\n', cleaned);
        else
            disp('No convincing plate text detected.');
            disp(' ');
            disp('Troubleshooting tips:');
            disp('  - Lower Otsu threshold manually in binarise_plate()');
            disp('  - Increase upscale factor above 3.0');
            disp('  - Increase sharpen Amount to 2.0');
            disp('  - Check Stage 3 montage: characters should be solid black on white');
        end

    catch ME
        fprintf('OCR failed: %s\n', ME.message);
        disp('Make sure the Computer Vision Toolbox is installed (run: ver)');
    end

    %% ── Final result figure ──────────────────────────────────────────────
    figure('Name','RESULT');
    imshow(plateCrop);
    if exist('cleaned','var') && length(cleaned) >= 5
        title(sprintf('Detected plate: %s', cleaned), ...
              'FontSize', 20, 'FontWeight', 'bold');
    else
        title('Plate crop (OCR result unclear)', 'FontSize', 14);
    end

    disp(' ');
    disp('If result is empty or wrong:');
    disp('  - Try increasing upscale factor (change 3.0 to 4.0)');
    disp('  - Lower sensitivity to 0.95-0.97 if too noisy');
    disp('  - Increase sharpen Amount/Radius');
end


% ---------------------------------------------------------
function plateCrop = blob_crop(img, mask, minAspect, minArea)
    plateCrop = [];
    stats = regionprops(mask, 'BoundingBox', 'Area');
    if isempty(stats); return; end

    validIdx = [];
    for i = 1:length(stats)
        bb = stats(i).BoundingBox;
        if (bb(3)/max(bb(4),1)) >= minAspect && stats(i).Area >= minArea
            validIdx(end+1) = i; %#ok<AGROW>
        end
    end
    if isempty(validIdx); return; end

    [~, best] = max([stats(validIdx).Area]);
    bb = stats(validIdx(best)).BoundingBox;

    [h, w, ~] = size(img);
    pad   = 6;
    bb(1) = max(1,   bb(1) - pad);
    bb(2) = max(1,   bb(2) - pad);
    bb(3) = min(w - bb(1), bb(3) + 2*pad);
    bb(4) = min(h - bb(2), bb(4) + 2*pad);
    plateCrop = imcrop(img, bb);
end


% ---------------------------------------------------------
function score = aspect_score(cropImg)
    if isempty(cropImg); score = Inf; return; end
    [ch, cw, ~] = size(cropImg);
    if ch == 0;          score = Inf; return; end
    score = abs((cw/ch) - 4.5);
end

function cleanup_temp_file(tempImagePath)
% Remove the temporary image file written by the wrapper.
    if exist(tempImagePath, 'file')
        delete(tempImagePath);
    end
end
