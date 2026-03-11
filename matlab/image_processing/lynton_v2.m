function processedImg = lynton_v2(img)
% Dual yellow/white UK plate detector with helper-block montages.
% Returns the binarized, cleaned plate image (or empty if no plate).

    validateattributes(img, {'uint8','uint16','double','single'}, {'nonempty','size',[NaN NaN 3]}, mfilename, 'img');

    [yellowMask, whiteMask] = extract_color_masks_block(img);
    [roi, orientation, chosenColor] = select_plate_candidate_block(yellowMask, whiteMask, img);
    [plateRGB, plateGray] = crop_and_deskew_block(img, roi, orientation, chosenColor);
    processedImg = preprocess_plate_block(plateRGB, plateGray, chosenColor);

    ocr_debug_block(processedImg, plateRGB);
end

function [yellowMask, whiteMask] = extract_color_masks_block(img)
% HSV gating for yellow (rear) and white (front) plates with morphology cleanup.
    hsvImg = rgb2hsv(img);
    hue = hsvImg(:, :, 1);
    sat = hsvImg(:, :, 2);
    val = hsvImg(:, :, 3);

    yellowMask = (hue >= 0.08 & hue <= 0.22) & (sat >= 0.65) & (val >= 0.35);
    yellowMask = imopen(yellowMask, strel('disk', 4));
    yellowMask = imclose(yellowMask, strel('rectangle', [6 25]));
    yellowMask = imfill(yellowMask, 'holes');
    yellowMask = imdilate(yellowMask, strel('rectangle', [3 12]));

    whiteMask = (sat <= 0.08) & (val >= 0.80);
    whiteMask = imopen(whiteMask, strel('disk', 4));
    whiteMask = imclose(whiteMask, strel('rectangle', [10 40]));
    whiteMask = imfill(whiteMask, 'holes');
    whiteMask = imdilate(whiteMask, strel('rectangle', [3 12]));

    % Keep only the largest blob in each mask to suppress noise.
    yellowMask = keep_largest_component(yellowMask);
    whiteMask = keep_largest_component(whiteMask);

    % Build montage frames (convert masks to RGB for consistent sizing).
    yellowRGB = repmat(uint8(yellowMask) * 255, [1 1 3]);
    whiteRGB = repmat(uint8(whiteMask) * 255, [1 1 3]);
    combinedRGB = repmat(uint8(yellowMask | whiteMask) * 255, [1 1 3]);

    figure('Name', 'lynton_v2 - Color Masks', 'NumberTitle', 'off');
    montage({img, yellowRGB, whiteRGB, combinedRGB}, 'Size', [1 4], 'BackgroundColor', 'white', 'BorderSize', 8);
    title('Original | Yellow mask | White mask | Combined');
end

function [roi, orientation, chosenColor] = select_plate_candidate_block(yellowMask, whiteMask, img)
% Choose the better plate candidate between yellow and white masks using area.
    [roiYellow, oriYellow, areaYellow] = largest_component_props(yellowMask);
    [roiWhite, oriWhite, areaWhite] = largest_component_props(whiteMask);

    minArea = 800;
    if areaYellow >= max(areaWhite, minArea)
        roi = roiYellow;
        orientation = oriYellow;
        chosenColor = 'yellow';
    elseif areaWhite >= minArea
        roi = roiWhite;
        orientation = oriWhite;
        chosenColor = 'white';
    else
        roi = [];
        orientation = 0;
        chosenColor = 'none';
    end

    % Overlay visualizations for montage.
    overlayBase = im2double(img);
    yellowOverlay = overlay_mask(overlayBase, yellowMask, [1 0.55 0]);
    whiteOverlay  = overlay_mask(overlayBase, whiteMask,  [0 0.75 1]);
    choiceOverlay = overlayBase;
    if ~isempty(roi)
        choiceOverlay = draw_box(choiceOverlay, roi, [0 1 0]);
    end

    figure('Name', 'lynton_v2 - Candidate Selection', 'NumberTitle', 'off');
    montage({im2uint8(overlayBase), im2uint8(yellowOverlay), im2uint8(whiteOverlay), im2uint8(choiceOverlay)}, ...
        'Size', [1 4], 'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Base | Yellow overlay | White overlay | Choice (%s)', chosenColor));
end

function [plateRGB, plateGray] = crop_and_deskew_block(img, roi, orientation, chosenColor)
% Crop around the selected ROI and optionally deskew using orientation.
    plateRGB = [];
    plateGray = [];

    if isempty(roi)
        figure('Name', 'lynton_v2 - Crop', 'NumberTitle', 'off');
        montage({img}, 'Size', [1 1], 'BackgroundColor', 'white', 'BorderSize', 8);
        title('No plate detected');
        return;
    end

    pad = 20;
    x1 = max(1, round(roi(1) - pad));
    y1 = max(1, round(roi(2) - pad));
    x2 = min(size(img, 2), round(roi(1) + roi(3) + pad));
    y2 = min(size(img, 1), round(roi(2) + roi(4) + pad));

    cols = x1:x2;
    rows = y1:y2;
    plateRGB = img(rows, cols, :);
    plateGray = rgb2gray(plateRGB);

    if abs(orientation) > 0.8
        plateRGB = imrotate(plateRGB, -orientation, 'bilinear', 'crop');
        plateGray = imrotate(plateGray, -orientation, 'bilinear', 'crop');
    end

    boxOverlay = draw_box(im2double(img), [x1, y1, x2 - x1 + 1, y2 - y1 + 1], [0 1 0]);

    figure('Name', 'lynton_v2 - Crop', 'NumberTitle', 'off');
    montage({im2uint8(img), im2uint8(boxOverlay), im2uint8(plateRGB)}, 'Size', [1 3], 'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Original | ROI (%s) | Cropped', chosenColor));
end

function processedImg = preprocess_plate_block(plateRGB, plateGray, chosenColor)
% Adaptive binarization + morphology cleanup for OCR-friendly plate image.
    if isempty(plateGray)
        processedImg = [];
        figure('Name', 'lynton_v2 - Preprocess', 'NumberTitle', 'off');
        montage({uint8(zeros(100, 200))}, 'Size', [1 1]);
        title('No crop available');
        return;
    end

    % Brighten and contrast stretch to tame harsh shadows.
    stretched = imadjust(plateGray, stretchlim(plateGray, [0.02 0.98]), []);

    bw = imbinarize(stretched, 'adaptive', 'Sensitivity', 0.99);
    bw = imdilate(bw, strel('disk', 9));
    bw = imerode(bw, strel('disk', 3));
    bw = imclose(bw, strel('disk', 10));
    bw = imcomplement(bw);
    processedImg = bwareaopen(bw, 5000);

    % Visualize each stage.
    stretchedRGB = repmat(im2uint8(stretched), [1 1 3]);
    bwRGB = repmat(uint8(processedImg) * 255, [1 1 3]);
    figure('Name', 'lynton_v2 - Preprocess', 'NumberTitle', 'off');
    montage({im2uint8(plateRGB), stretchedRGB, bwRGB}, 'Size', [1 3], 'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Cropped RGB | Contrast stretch | Binary cleaned (%s)', chosenColor));
end

function ocr_debug_block(bwPlate, plateRGB)
% Optional OCR attempt with a quick montage.
    if isempty(bwPlate)
        return;
    end

    try
        res = ocr(bwPlate, 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', 'LayoutAnalysis', 'line');
        raw = strtrim(res.Text);
        cleaned = upper(regexprep(raw, '[^A-Z0-9]', ''));
        cleaned = regexprep(cleaned, '[0OQ]', 'O');
        cleaned = regexprep(cleaned, '[1IL]', 'I');
        cleaned = regexprep(cleaned, '[5S]', 'S');
    catch ME
        raw = sprintf('OCR failed: %s', ME.message);
        cleaned = '';
    end

    bwRGB = repmat(uint8(bwPlate) * 255, [1 1 3]);

    figure('Name', 'lynton_v2 - OCR', 'NumberTitle', 'off');
    montage({im2uint8(plateRGB), bwRGB}, 'Size', [1 2], 'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Plate RGB | Binary for OCR | Text: %s', cleaned));
end

function maskOut = keep_largest_component(maskIn)
% Keep only the largest connected component; return empty mask if none.
    maskOut = false(size(maskIn));
    if ~any(maskIn(:))
        return;
    end
    conn = bwconncomp(maskIn);
    stats = regionprops(conn, 'Area');
    [~, idx] = max([stats.Area]);
    maskOut(conn.PixelIdxList{idx}) = true;
end

function [roi, orientation, area] = largest_component_props(mask)
% Return bounding box + orientation for the largest component in mask.
    roi = [];
    orientation = 0;
    area = 0;
    if ~any(mask(:))
        return;
    end
    conn = bwconncomp(mask);
    props = regionprops(conn, 'BoundingBox', 'Area', 'Orientation');
    [area, idx] = max([props.Area]);
    roi = props(idx).BoundingBox;
    orientation = props(idx).Orientation;
end

function overlay = overlay_mask(baseImg, mask, color)
% Simple alpha overlay of a binary mask onto an RGB image.
    overlay = baseImg;
    if ~any(mask(:))
        return;
    end
    alpha = 0.35;
    for c = 1:3
        overlay(:, :, c) = min(1, overlay(:, :, c) + alpha * color(c) * double(mask));
    end
end

function imgOut = draw_box(imgIn, roi, color)
% Draw rectangle edges into an RGB image without toolbox helpers.
    imgOut = imgIn;
    if isempty(roi)
        return;
    end
    x = round(roi(1));
    y = round(roi(2));
    w = round(roi(3));
    h = round(roi(4));

    cols = max(1, x):min(size(imgOut, 2), x + w);
    rows = max(1, y):min(size(imgOut, 1), y + h);

    % Top and bottom lines
    imgOut(rows(1), cols, 1) = color(1);
    imgOut(rows(1), cols, 2) = color(2);
    imgOut(rows(1), cols, 3) = color(3);
    imgOut(rows(end), cols, 1) = color(1);
    imgOut(rows(end), cols, 2) = color(2);
    imgOut(rows(end), cols, 3) = color(3);

    % Left and right lines
    imgOut(rows, cols(1), 1) = color(1);
    imgOut(rows, cols(1), 2) = color(2);
    imgOut(rows, cols(1), 3) = color(3);
    imgOut(rows, cols(end), 1) = color(1);
    imgOut(rows, cols(end), 2) = color(2);
    imgOut(rows, cols(end), 3) = color(3);
end
