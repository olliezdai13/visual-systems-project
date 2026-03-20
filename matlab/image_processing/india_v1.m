function processedImg = india_v1(img)
% Hough-first plate crop with yellow-mask fallback for India pipeline v1.
% Returns the best cropped RGB candidate, or the original image if none found.

    validateattributes(img, {'uint8','uint16','double','single'}, {'nonempty','size',[NaN NaN 3]}, mfilename, 'img');

    [grayImg, edgeMask, lines, horizontalLines, verticalLines] = detect_hough_lines_block(img);
    houghCrop = crop_from_horizontal_lines_block(img, grayImg, horizontalLines);
    fallbackCrop = yellow_mask_fallback_block(img);
    processedImg = choose_best_crop_block(img, houghCrop, fallbackCrop, verticalLines);
end

function [grayImg, edgeMask, lines, horizontalLines, verticalLines] = detect_hough_lines_block(img)
% Convert to grayscale, detect Sobel edges, extract Hough lines, and show line classes.
    grayImg = rgb2gray(img);
    edgeMask = edge(grayImg, 'sobel');

    [H, theta, rho] = hough(edgeMask);
    peakThreshold = ceil(0.3 * max(H(:)));
    peaks = houghpeaks(H, 6, 'Threshold', peakThreshold);
    lines = houghlines(edgeMask, theta, rho, peaks, 'FillGap', 20, 'MinLength', 80);

    if isempty(lines)
        horizontalLines = lines;
        verticalLines = lines;
    else
        lineAngles = [lines.theta];
        horizontalLines = lines(abs(lineAngles) < 20);
        verticalLines = lines(abs(abs(lineAngles) - 90) <= 1);
    end

    rawGray = repmat(grayImg, [1 1 3]);
    edgeRGB = repmat(uint8(edgeMask) * 255, [1 1 3]);
    overlay = im2double(img);
    overlay = draw_line_set(overlay, lines, [1 0 0]);
    overlay = draw_line_set(overlay, verticalLines, [0 0 1]);

    figure('Name', 'india_v1 - Hough Detection', 'NumberTitle', 'off');
    montage({im2uint8(rawGray), edgeRGB, im2uint8(overlay)}, 'Size', [1 3], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Grayscale | Sobel edges | Hough lines (red = other, blue = vertical)');
end

function plateCrop = crop_from_horizontal_lines_block(img, grayImg, horizontalLines)
% Crop a candidate plate region using the horizontal Hough line envelope.
    [imgHeight, imgWidth] = size(grayImg);
    plateCrop = [];
    cropOverlay = im2double(img);

    if numel(horizontalLines) >= 2
        allPoints = [[horizontalLines.point1]; [horizontalLines.point2]];
        xCoords = allPoints(:, 1);
        yCoords = allPoints(:, 2);

        xMin = max(1, min(xCoords) - 10);
        xMax = min(imgWidth, max(xCoords) + 10);
        yMin = max(1, min(yCoords) - 10);
        yMax = min(imgHeight, max(yCoords) + 10);

        plateCrop = crop_by_bounds(img, xMin, xMax, yMin, yMax);
        cropOverlay = draw_box(cropOverlay, [xMin, yMin, xMax - xMin + 1, yMax - yMin + 1], [0 1 0]);
    end

    cropPreview = montage_placeholder(img, plateCrop);
    figure('Name', 'india_v1 - Hough Crop', 'NumberTitle', 'off');
    montage({im2uint8(cropOverlay), cropPreview}, 'Size', [1 2], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Horizontal-line crop box | Cropped licence plate region');
end

function plateCrop = yellow_mask_fallback_block(img)
% Build a yellow HSV mask, clean it morphologically, and crop the largest blob.
    hsvImg = rgb2hsv(img);
    hue = hsvImg(:, :, 1);
    sat = hsvImg(:, :, 2);
    val = hsvImg(:, :, 3);

    yellowMask = (hue > 0.10 & hue < 0.20) & (sat > 0.50) & (val > 0.50);
    yellowMask = close_with_primitives(yellowMask, [5 20]);
    yellowMask = imfill(yellowMask, 'holes');

    [bbox, hasRegion] = largest_component_bbox(yellowMask);
    if hasRegion
        xMin = bbox(1);
        yMin = bbox(2);
        xMax = bbox(1) + bbox(3) - 1;
        yMax = bbox(2) + bbox(4) - 1;
        plateCrop = crop_by_bounds(img, xMin, xMax, yMin, yMax);
        overlay = draw_box(im2double(img), bbox, [1 1 0]);
    else
        plateCrop = [];
        overlay = im2double(img);
    end

    maskRGB = repmat(uint8(yellowMask) * 255, [1 1 3]);
    cropPreview = montage_placeholder(img, plateCrop);
    figure('Name', 'india_v1 - Yellow Fallback', 'NumberTitle', 'off');
    montage({maskRGB, im2uint8(overlay), cropPreview}, 'Size', [1 3], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Yellow mask | Largest yellow region | Fallback crop');
end

function processedImg = choose_best_crop_block(img, houghCrop, fallbackCrop, verticalLines)
% Prefer the Hough crop when it exists; otherwise use the yellow-mask fallback.
    if ~isempty(houghCrop)
        processedImg = houghCrop;
        choiceLabel = sprintf('Selected Hough crop (%d vertical lines)', numel(verticalLines));
    elseif ~isempty(fallbackCrop)
        processedImg = fallbackCrop;
        choiceLabel = 'Selected yellow-mask fallback crop';
    else
        processedImg = img;
        choiceLabel = 'No crop found, returning original image';
    end

    choicePreview = montage_placeholder(img, processedImg);
    figure('Name', 'india_v1 - Final Choice', 'NumberTitle', 'off');
    montage({im2uint8(img), choicePreview}, 'Size', [1 2], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Original | %s', choiceLabel));
end

function cropped = crop_by_bounds(img, xMin, xMax, yMin, yMax)
% Crop using integer bounds without relying on imcrop.
    xMin = max(1, round(xMin));
    xMax = min(size(img, 2), round(xMax));
    yMin = max(1, round(yMin));
    yMax = min(size(img, 1), round(yMax));

    if xMax < xMin || yMax < yMin
        cropped = [];
        return;
    end

    cropped = img(yMin:yMax, xMin:xMax, :);
end

function closedMask = close_with_primitives(mask, rectSize)
% Approximate morphological closing with allowed primitives.
    se = strel('rectangle', rectSize);
    dilatedMask = imdilate(mask, se);
    closedMask = imerode(dilatedMask, se);
end

function [bbox, hasRegion] = largest_component_bbox(mask)
% Compute the largest connected component bounding box using bwconncomp only.
    bbox = [1 1 size(mask, 2) size(mask, 1)];
    hasRegion = false;

    if ~any(mask(:))
        return;
    end

    conn = bwconncomp(mask);
    bestArea = 0;

    for idx = 1:conn.NumObjects
        pixelIdx = conn.PixelIdxList{idx};
        area = numel(pixelIdx);
        if area <= bestArea
            continue;
        end

        [rows, cols] = ind2sub(size(mask), pixelIdx);
        xMin = min(cols);
        xMax = max(cols);
        yMin = min(rows);
        yMax = max(rows);

        bbox = [xMin, yMin, xMax - xMin + 1, yMax - yMin + 1];
        bestArea = area;
        hasRegion = true;
    end
end

function overlay = draw_line_set(baseImg, lines, color)
% Draw Hough line segments into an RGB image.
    overlay = baseImg;
    for idx = 1:numel(lines)
        overlay = draw_line(overlay, lines(idx).point1, lines(idx).point2, color);
    end
end

function imgOut = draw_line(imgIn, point1, point2, color)
% Rasterize a line segment by interpolating along the dominant axis.
    imgOut = imgIn;
    x1 = round(point1(1));
    y1 = round(point1(2));
    x2 = round(point2(1));
    y2 = round(point2(2));

    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1;
    xSamples = round(linspace(x1, x2, steps));
    ySamples = round(linspace(y1, y2, steps));

    xSamples = min(max(xSamples, 1), size(imgOut, 2));
    ySamples = min(max(ySamples, 1), size(imgOut, 1));

    linearIdx = sub2ind(size(imgOut(:, :, 1)), ySamples, xSamples);
    for channel = 1:3
        channelPlane = imgOut(:, :, channel);
        channelPlane(linearIdx) = color(channel);
        imgOut(:, :, channel) = channelPlane;
    end
end

function imgOut = draw_box(imgIn, roi, color)
% Draw rectangle edges into an RGB image.
    imgOut = imgIn;
    x = round(roi(1));
    y = round(roi(2));
    width = round(roi(3));
    height = round(roi(4));

    cols = max(1, x):min(size(imgOut, 2), x + width - 1);
    rows = max(1, y):min(size(imgOut, 1), y + height - 1);

    if isempty(cols) || isempty(rows)
        return;
    end

    for channel = 1:3
        imgOut(rows(1), cols, channel) = color(channel);
        imgOut(rows(end), cols, channel) = color(channel);
        imgOut(rows, cols(1), channel) = color(channel);
        imgOut(rows, cols(end), channel) = color(channel);
    end
end

function preview = montage_placeholder(referenceImg, candidateImg)
% Return a displayable crop preview, falling back to a blank frame when needed.
    if isempty(candidateImg)
        preview = zeros(size(referenceImg), 'uint8');
    else
        preview = im2uint8(candidateImg);
    end
end
