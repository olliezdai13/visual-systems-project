function processedImg = lynton_v3(img)
% Wrapper for the provided UK number plate reader script using the selected image.

    validateattributes(img, {'uint8', 'uint16', 'double', 'single'}, ...
        {'nonempty', 'size', [NaN NaN 3]}, mfilename, 'img');

    setappdata(0, 'lynton_v3_img', img);
    cleanupObj = onCleanup(@() rmappdata(0, 'lynton_v3_img')); %#ok<NASGU>

    processedImg = run_lynton_v3();
end

function processedImg = run_lynton_v3()
% =============================================================================
% UK Number Plate Reader - dual yellow (rear) + white (front) detection
% =============================================================================

clear; clc; close all;
%% 1. Load image
img = getappdata(0, 'lynton_v3_img');
processedImg = img;

if size(img,3) ~= 3
    error('Image must be RGB (3 channels)');
end

figure('Name','1 - Original','NumberTitle','off');
imshow(img); title('Original image');

%% 2. Convert color spaces
gray = rgb2gray(img);
hsv  = rgb2hsv(img);

%% 3. Dual color detection: yellow (rear) + white (front)

% ── Yellow branch (rear plate) ───────────────────────────────────────
yellowMask = (hsv(:,:,1) >= 0.08 & hsv(:,:,1) <= 0.22) & ...
             (hsv(:,:,2) >= 0.65) & ...
             (hsv(:,:,3) >= 0.35);

yellowMask = imopen(yellowMask, strel('disk', 4));
yellowMask = imclose(yellowMask, strel('rectangle',[6 25]));
yellowMask = imfill(yellowMask, 'holes');
yellowMask = bwareaopen(yellowMask, 100);
yellowMask = imdilate(yellowMask, strel('rectangle',[3 12]));

% ── White branch (front plate) ───────────────────────────────────────
whiteMask = (hsv(:,:,2) <= 0.08) & ...           % allow some tint
            (hsv(:,:,3) >= 0.8);                % bright but not extreme


whiteMask = imopen(whiteMask, strel('disk', 4));
whiteMask = imclose(whiteMask, strel('rectangle',[10 40]));
whiteMask = imfill(whiteMask, 'holes');
whiteMask = bwareaopen(whiteMask, 10000);
whiteMask = imdilate(whiteMask, strel('rectangle',[3 12]));


%% 4. Morphology cleanup – keep largest region only
% Yellow
yellowMask = bwareafilt(yellowMask, 1, 'largest');
% White
whiteMask = bwareafilt(whiteMask, 1, 'largest');

fprintf('After cleanup:\n');

%%
fprintf(' Yellow pixels: %.2f%%\n', 100 * nnz(yellowMask)/numel(yellowMask));
fprintf(' White pixels:  %.2f%%\n', 100 * nnz(whiteMask)/numel(whiteMask));

%% 5. Visualize masks (safe – no crash if no selection yet)

figure('Name', '5 – Yellow vs White Masks', 'NumberTitle', 'off', ...
       'Position', [150 150 1400 800]);

subplot(2,3,1); imshow(yellowMask); title('Yellow Mask');
subplot(2,3,2); imshow(whiteMask);  title('White Mask');

subplot(2,3,4); imshow(labeloverlay(img, yellowMask, 'Transparency',0.65,'Colormap',[1 0.5 0]));
title('Yellow overlaid');
subplot(2,3,5); imshow(labeloverlay(img, whiteMask, 'Transparency',0.65,'Colormap',[0 0.8 1]));
title('White overlaid');


%% 6. Find largest plate-like candidate (yellow or white)
stats = [];
roi = [];
orientation = 0;
selectedColor = 'none';

fprintf('Candidate search debug:\n');

% Try yellow first
if exist('yellowMask', 'var') && any(yellowMask(:))
    statsY = regionprops(yellowMask, 'BoundingBox', 'Area', 'Eccentricity', 'Extent', 'Orientation');
    if ~isempty(statsY)
        areasY = [statsY.Area];
        maxY = max(areasY);
        fprintf('  Yellow: %d regions, largest = %d px\n', numel(statsY), maxY);
        if maxY >= 800
            [~, idxY] = max(areasY);
            stats = statsY;
            roi = round(statsY(idxY).BoundingBox);
            orientation = statsY(idxY).Orientation;
            selectedColor = 'yellow';
        else
            fprintf('  Yellow skipped – largest too small (%d px)\n', maxY);
        end
    else
        fprintf('  Yellow: no regions\n');
    end
else
    fprintf('  Yellow mask empty or missing\n');
end

% Fallback to white
if isempty(stats) && exist('whiteMask', 'var') && any(whiteMask(:))
    statsW = regionprops(whiteMask, 'BoundingBox', 'Area', 'Eccentricity', 'Extent', 'Orientation');
    if ~isempty(statsW)
        areasW = [statsW.Area];
        maxW = max(areasW);
        fprintf('  White: %d regions, largest = %d px\n', numel(statsW), maxW);
        if maxW >= 800
            [~, idxW] = max(areasW);
            stats = statsW;
            roi = round(statsW(idxW).BoundingBox);
            orientation = statsW(idxW).Orientation;
            selectedColor = 'white';
        else
            fprintf('  White skipped – largest too small (%d px)\n', maxW);
        end
    else
        fprintf('  White: no regions\n');
    end
else
    fprintf('  White mask empty or missing\n');
end

if isempty(stats)
    warning('No valid region in yellow or white mask.');
    disp('Try manual crop fallback below.');
    figure('Name', 'No Detection', 'NumberTitle', 'off');
    imshow(img);
    title('No plate detected – try manual crop', 'Color', 'red');
    return;
end

fprintf('SELECTED: %s plate | area = %d px\n', upper(selectedColor), stats.Area);

roi = [roi(1)-20, roi(2)-20, roi(3)+40, roi(4)+40];
roi(1) = max(1, roi(1));
roi(2) = max(1, roi(2));
roi(3) = min(size(img,2)-roi(1)+1, roi(3));
roi(4) = min(size(img,1)-roi(2)+1, roi(4));


%% 7. Crop and deskew
plate_rgb  = imcrop(img, roi);
plate_gray = imcrop(gray, roi);

if abs(orientation) > 0.8
    plate_gray = imrotate(plate_gray, -orientation, 'bilinear', 'crop');
    plate_rgb  = imrotate(plate_rgb, -orientation, 'bilinear', 'crop');
    fprintf('Deskewed by %.1f degrees.\n', -orientation);
end

% Ensure the cropped image has a white background
plate_gray(plate_gray == 0) = 255;  % Set black pixels to white


figure('Name','Cropped & Deskewed','NumberTitle','off');
imshow(plate_rgb); title('Selected Crop');

%% 8. Preprocess – remove small white blobs at top/edges

% Boost contrast of the image 'plate_gray'
conplate_gray = imadjust(plate_gray);
% increase brightness of conplate_gray by a given value 
% Increase brightness of conplate_gray by a given value
conplate_gray = conplate_gray + 100;  % Adjust brightness level as needed

bw = imbinarize(plate_gray, "adaptive", "sensitivity", 0.99);
bw = imdilate(bw, strel('disk', 9));
bw = imerode(bw, strel('disk', 3));
bw = imclose(bw, strel('disk', 10));
bw = imcomplement(bw);
bw = bwareaopen(bw, 5000);

figure('Name','4 - Preprocessed','NumberTitle','off');
montage({plate_gray, bw}, 'Size',[1 2], 'BackgroundColor','w', 'BorderSize',10);
title('Grayscale | Binary inverted + small top blob removal');
%% OCR


try
    res = ocr(bw, ...
        'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', ...
        'LayoutAnalysis', 'line');
    
    raw = res.Text;
    disp('Raw OCR output:');
    disp(raw);
    
    % Clean UK plate style
    cleaned = upper(regexprep(raw, '[^A-Z0-9]', ''));
    cleaned = regexprep(cleaned, '[0OQ]', 'O');
    cleaned = regexprep(cleaned, '[1IL]', 'I');
    cleaned = regexprep(cleaned, '[5S]', 'S');
    
    if length(cleaned) >= 7
        fprintf('\nFinal cleaned plate: ** %s **\n', cleaned(1:7));
    elseif length(cleaned) >= 5
        fprintf('\nPartial plate: %s\n', cleaned);
    else
        disp('No convincing plate text detected.');
    end
    
catch ME
    fprintf('OCR failed: %s\n', ME.message);
end

% Show montage of original image and final result
try
    % Determine title text
    if exist('cleaned','var') && ~isempty(cleaned)
        if length(cleaned) >= 7
            plateText = cleaned(1:7);
            titleStr = sprintf('Final cleaned plate: ** %s **', plateText);
        else
            titleStr = sprintf('Final cleaned plate (partial): %s', cleaned);
        end
    else
        titleStr = 'Final cleaned plate: (none)';
    end
catch
    titleStr = 'Final cleaned plate: (none)';
end

% Prepare images for montage: original RGB 'img' and binary 'bw' as RGB
if size(img,3) == 1
    img_rgb = repmat(img, [1 1 3]);
else
    img_rgb = img;
end
bw_rgb = repmat(uint8(~bw) * 255, [1 1 3]); % show binary inverted as white-on-black for visibility

figure('Name','Final Comparison','NumberTitle','off');
montage({img_rgb, bw_rgb}, 'Size', [1 2], 'BackgroundColor', 'w', 'BorderSize', 10);
title(titleStr, 'Interpreter', 'none');

processedImg = bw_rgb;
end
