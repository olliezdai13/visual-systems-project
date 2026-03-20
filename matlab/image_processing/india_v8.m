function processedImg = india_v8(img)
% Boilerplate controller for India pipeline v8.

    processedImg = run_india_v8_script_block(img);
end

function processedImg = run_india_v8_script_block(img)
% Wrapped legacy script block.
    testName = stage_selected_image(img);
    licensePlate = imread(testName);
    whos licensePlate
    imshow(licensePlate)
    licenseplateGrey = rgb2gray(licensePlate);
    max(licenseplateGrey(:))

    % Sobel edge detection
    BW_edges = edge(licenseplateGrey, 'sobel');

    % Hough Transform
    [H, theta, rho] = hough(BW_edges);
    P = houghpeaks(H, 6, 'Threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(BW_edges, theta, rho, P, 'FillGap', 20, 'MinLength', 80);

    % Get image dimensions
    [h, w] = size(licenseplateGrey);

    % Visualise detected lines
    figure;
    imshow(licensePlate); hold on; title('Detected Hough Lines');

    horIdx   = abs([lines.theta]) < 20;
    horLines = lines(horIdx);

    vertX = [];
    vertY = [];
    for k = 1:length(lines)
        angle = lines(k).theta;
        xy = [lines(k).point1; lines(k).point2];
        if (angle == -90) || (angle == 90)
            vertX = [vertX, lines(k).point1(1), lines(k).point2(1)];
            vertY = [vertY, lines(k).point1(2), lines(k).point2(2)];
            plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'blue');
        else
            plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');
        end
    end

    %% Crop using horizontal lines
    if length(horLines) >= 2
        allX = [[horLines.point1]; [horLines.point2]];
        allY = [[horLines.point1]; [horLines.point2]];
        allX = allX(:,1);
        allY = allY(:,2);
        xMin = max(1,   min(allX) - 10);
        xMax = min(w,   max(allX) + 10);
        yMin = max(1,   min(allY) - 10);
        yMax = min(h,   max(allY) + 10);
        plateCrop = imcrop(licensePlate, [xMin, yMin, xMax - xMin, yMax - yMin]);
        figure; imshow(plateCrop); title('Cropped Plate (Hough)');
    else
        warning('Not enough horizontal lines found.');
        plateCrop = [];
    end

    %% ── YELLOW mask (French rear / UK rear plates) ──────────────────────────
    imgHSV = rgb2hsv(licensePlate);
    H_ch = imgHSV(:,:,1);
    S_ch = imgHSV(:,:,2);
    V_ch = imgHSV(:,:,3);

    yellowMask = (H_ch > 0.10 & H_ch < 0.20) & ...
                 (S_ch > 0.50)                & ...
                 (V_ch > 0.50);
    yellowMask = imclose(yellowMask, strel('rect', [5 20]));
    yellowMask = imfill(yellowMask, 'holes');

    statsY = regionprops(yellowMask, 'BoundingBox', 'Area');
    plateCropYellow = [];
    if ~isempty(statsY)
        [~, idxY]      = max([statsY.Area]);
        bbY            = statsY(idxY).BoundingBox;
        plateCropYellow = imcrop(licensePlate, bbY);
        figure; imshow(plateCropYellow); title('Cropped Plate (yellow mask)');
    end

    %% ── WHITE mask (UK front plates / most European front plates) ───────────
    % White = high Value, low Saturation, any Hue
    whiteMask = (S_ch < 0.20) & ...   % low saturation  → not colourful
                (V_ch > 0.80);         % high brightness → white not grey

    % Morphological cleanup — close small gaps, fill holes
    whiteMask = imclose(whiteMask, strel('rect', [5 20]));
    whiteMask = imfill(whiteMask, 'holes');

    % Remove blobs that are too small or have the wrong aspect ratio for a plate
    statsW = regionprops(whiteMask, 'BoundingBox', 'Area', 'Extent');
    plateCropWhite = [];

    if ~isempty(statsW)
        % Filter candidates: must be wide (aspect ratio > 2) and reasonably large
        validIdx = [];
        for i = 1:length(statsW)
            bb_i    = statsW(i).BoundingBox;
            aspect  = bb_i(3) / bb_i(4);          % width / height
            area    = statsW(i).Area;
            if aspect > 2.0 && area > 1000
                validIdx(end+1) = i;               %#ok<AGROW>
            end
        end

        if ~isempty(validIdx)
            % Pick the largest valid white region
            validAreas      = [statsW(validIdx).Area];
            [~, bestLocal]  = max(validAreas);
            bestIdx         = validIdx(bestLocal);
            bbW             = statsW(bestIdx).BoundingBox;
            plateCropWhite  = imcrop(licensePlate, bbW);
            figure; imshow(plateCropWhite); title('Cropped Plate (white mask)');
        else
            warning('White mask: no region passed the aspect-ratio filter.');
        end
    end

    %% ── Pick best result ─────────────────────────────────────────────────────
    % Score each candidate by aspect ratio closeness to ideal plate (~4.5:1)
    IDEAL_ASPECT = 4.5;

    function score = aspectScore(cropImg)
        if isempty(cropImg)
            score = Inf;
            return;
        end
        [ch, cw, ~] = size(cropImg);
        score = abs((cw / ch) - 4.5);
    end

    candidates  = {plateCrop, plateCropYellow, plateCropWhite};
    candNames   = {'Hough', 'Yellow mask', 'White mask'};
    scores      = cellfun(@aspectScore, candidates);

    [~, bestCand] = min(scores);
    finalPlate    = candidates{bestCand};

    figure;
    imshow(finalPlate);
    title(['Final Plate — source: ' candNames{bestCand}]);

    processedImg = finalPlate;

    figure('Name', 'india_v8 - Script Montage', 'NumberTitle', 'off');
    montage(build_montage_cells(licensePlate, plateCrop, plateCropYellow, plateCropWhite, finalPlate), ...
        'Size', [1 5], 'BackgroundColor', 'white', 'BorderSize', 8);
    title('Original | Hough crop | Yellow crop | White crop | Final plate');
end

function stagedPath = stage_selected_image(img)
% Stage the selected image for the wrapped script.
    stagedPath = fullfile(tempdir, 'india_v8_selected_image.png');
    imwrite(img, stagedPath);
end

function cells = build_montage_cells(licensePlate, plateCrop, plateCropYellow, plateCropWhite, finalPlate)
% Build montage inputs with safe RGB placeholders.
    cells = { ...
        ensure_uint8_rgb(licensePlate), ...
        crop_or_placeholder(plateCrop, licensePlate), ...
        crop_or_placeholder(plateCropYellow, licensePlate), ...
        crop_or_placeholder(plateCropWhite, licensePlate), ...
        crop_or_placeholder(finalPlate, licensePlate)};
end

function rgbImg = crop_or_placeholder(cropImg, referenceImg)
% Return a crop if present, otherwise a blank placeholder sized to the input.
    if isempty(cropImg)
        rgbImg = zeros(size(referenceImg), 'like', referenceImg);
    else
        rgbImg = ensure_uint8_rgb(cropImg);
    end
end

function rgbImg = ensure_uint8_rgb(img)
% Normalize image type/channels for montage display.
    if isa(img, 'uint8')
        rgbImg = img;
    else
        rgbImg = im2uint8(img);
    end

    if ndims(rgbImg) == 2 || size(rgbImg, 3) == 1
        rgbImg = cat(3, rgbImg, rgbImg, rgbImg);
    end
end
