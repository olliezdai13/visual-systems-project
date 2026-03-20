function result = oliver_v4(img)
% K-means plate candidate selection using rotated-box fill for the final pick.

    validateattributes(img, {'uint8', 'uint16', 'single', 'double'}, ...
        {'nonempty', 'size', [NaN NaN 3]}, mfilename, 'img');

    [clusterLabels, segmentedRGB] = kmeans_segmentation_block(img);
    nonRedMask = exclude_red_components_block(img);
    [componentMasks, componentStatsList] = select_plate_component_block(img, clusterLabels, nonRedMask);
    [rotatedBoxes, rotatedFillRatios, bestClusterIdx] = rotated_bbox_candidates_block(componentMasks);

    bestMask = false(size(clusterLabels));
    bestStats = default_stats();
    selectedRotatedBox = [];

    if bestClusterIdx > 0
        bestMask = componentMasks{bestClusterIdx};
        bestStats = componentStatsList{bestClusterIdx};
        bestStats.rotatedFillRatio = rotatedFillRatios(bestClusterIdx);
        selectedRotatedBox = rotatedBoxes{bestClusterIdx};
    end

    croppedComponent = crop_selected_component_block(img, bestMask, selectedRotatedBox);
    ocrText = ocr_license_plate_block(croppedComponent);
    result = render_selected_component_block(img, segmentedRGB, bestMask, croppedComponent, ocrText, bestClusterIdx, bestStats, selectedRotatedBox);
end

function [clusterLabels, segmentedRGB] = kmeans_segmentation_block(img)
% Run color clustering and show the per-cluster previews.
    clusterCount = 8;

    sourceImg = im2single(img);
    clusterLabels = imsegkmeans(sourceImg, clusterCount);
    segmentedRGB = label2rgb(clusterLabels, 'jet', 'k', 'shuffle');
    segmentedTiles = build_segment_tiles(img, clusterLabels, clusterCount);

    figure('Name', 'oliver_v4 - K-Means Segmentation', 'NumberTitle', 'off');
    montage(segmentedTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Raw k-means segment previews (%d clusters)', clusterCount));
end

function nonRedMask = exclude_red_components_block(img)
% Explicitly suppress red tail-light pixels before any cluster scoring.
    hsvImg = rgb2hsv(img);
    hue = hsvImg(:, :, 1);
    sat = hsvImg(:, :, 2);
    val = hsvImg(:, :, 3);

    lowRedMask = (hue <= 0.05) & (sat >= 0.35) & (val >= 0.20);
    highRedMask = (hue >= 0.95) & (sat >= 0.35) & (val >= 0.20);
    redMask = lowRedMask | highRedMask;

    nonRedMask = ~redMask;
    huePreview = repmat(im2uint8(hue), [1 1 3]);
    satPreview = repmat(im2uint8(sat), [1 1 3]);
    valPreview = repmat(im2uint8(val), [1 1 3]);

    maskedPreview = img;
    for channelIdx = 1:size(img, 3)
        channelPlane = maskedPreview(:, :, channelIdx);
        channelPlane(~nonRedMask) = 0;
        maskedPreview(:, :, channelIdx) = channelPlane;
    end

    figure('Name', 'oliver_v4 - Red Exclusion', 'NumberTitle', 'off');
    montage({im2uint8(img), huePreview, satPreview, valPreview, mask_to_rgb(redMask), maskedPreview}, ...
        'Size', [1 6], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Original | Hue | Saturation | Value | Explicit red mask | Red-suppressed image');
end

function [bestComponentMasks, bestComponentStats] = select_plate_component_block(img, clusterLabels, nonRedMask)
% Pick one strong candidate component from each cluster after red suppression.
    clusterCount = max(clusterLabels(:));
    imageArea = size(img, 1) * size(img, 2);
    bestComponentMasks = cell(1, clusterCount);
    bestComponentStats = cell(1, clusterCount);

    clusterPreviewTiles = cell(1, clusterCount);
    processedPreviewTiles = cell(1, clusterCount);
    candidatePreviewTiles = cell(1, clusterCount);

    for clusterIdx = 1:clusterCount
        rawMask = (clusterLabels == clusterIdx) & nonRedMask;
        [processedMask, candidateMask, candidateStats] = find_best_component_in_cluster(rawMask, imageArea);

        clusterPreviewTiles{clusterIdx} = mask_to_rgb(rawMask);
        processedPreviewTiles{clusterIdx} = mask_to_rgb(processedMask);
        candidatePreviewTiles{clusterIdx} = mask_to_rgb(candidateMask);
        bestComponentMasks{clusterIdx} = candidateMask;
        bestComponentStats{clusterIdx} = candidateStats;
    end

    figure('Name', 'oliver_v4 - Plate Component Selection - Filtered Clusters', 'NumberTitle', 'off');
    montage(clusterPreviewTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Raw cluster masks');

    figure('Name', 'oliver_v4 - Plate Component Selection - Processed Masks', 'NumberTitle', 'off');
    montage(processedPreviewTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Rectangle-friendly processed cluster masks');

    figure('Name', 'oliver_v4 - Plate Component Selection - Best Per Cluster', 'NumberTitle', 'off');
    montage(candidatePreviewTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Best rectangular component from each cluster');

end

function [rotatedBoxes, rotatedFillRatios, bestClusterIdx] = rotated_bbox_candidates_block(componentMasks)
% Compute rotated bounding boxes and select the candidate with the best fill.
    clusterCount = numel(componentMasks);
    rotatedBoxes = cell(1, clusterCount);
    rotatedFillRatios = -Inf(1, clusterCount);
    bestClusterIdx = 0;
    bestFillRatio = -Inf;
    overlayTiles = cell(1, clusterCount);

    for clusterIdx = 1:clusterCount
        componentMask = componentMasks{clusterIdx};
        [rotatedBoxes{clusterIdx}, rotatedArea] = compute_rotated_bbox(componentMask);
        rotatedFillRatios(clusterIdx) = compute_rotated_fill_ratio(componentMask, rotatedArea);
        if rotatedFillRatios(clusterIdx) > bestFillRatio
            bestFillRatio = rotatedFillRatios(clusterIdx);
            bestClusterIdx = clusterIdx;
        end
    end

    for clusterIdx = 1:clusterCount
        overlayColor = [0 1 0];
        if clusterIdx == bestClusterIdx
            overlayColor = [1 0 0];
        end

        overlayTiles{clusterIdx} = draw_polygon_overlay(mask_to_rgb(componentMasks{clusterIdx}), rotatedBoxes{clusterIdx}, overlayColor);
    end

    figure('Name', 'oliver_v4 - Rotated Bounding Boxes', 'NumberTitle', 'off');
    montage(overlayTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);

    if bestClusterIdx > 0
        title(sprintf('Best component per cluster with rotated-box overlays | selected cluster %d | best fill %.2f', ...
            bestClusterIdx, bestFillRatio));
    else
        title('Best component per cluster with rotated-box overlays | no valid rotated fill found');
    end
end

function croppedComponent = crop_selected_component_block(img, bestMask, rotatedBox)
% Crop the selected rotated rectangle and deskew it to horizontal.
    maskedComponent = apply_mask_to_image(img, bestMask);
    croppedComponent = [];

    if ~isempty(rotatedBox)
        [cropWidth, cropHeight] = rotated_box_size(rotatedBox);
        if cropWidth > 0 && cropHeight > 0
            croppedComponent = sample_rotated_rectangle(img, rotatedBox, cropWidth, cropHeight);
        end
    end

    bboxOverlay = draw_polygon_overlay(img, rotatedBox, [0 1 0]);
    cropPreview = croppedComponent;
    if isempty(cropPreview)
        cropPreview = maskedComponent;
    end

    figure('Name', 'oliver_v4 - Final Crop', 'NumberTitle', 'off');
    montage({im2uint8(img), im2uint8(bboxOverlay), maskedComponent, cropPreview}, ...
        'Size', [1 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Original | Rotated bounding box | Selected component | Deskewed rotated-box crop');
end

function result = render_selected_component_block(img, segmentedRGB, bestMask, croppedComponent, ocrText, bestClusterIdx, bestStats, rotatedBox)
% Return the final selected component crop and show the overall extraction summary.
    maskedComponent = apply_mask_to_image(img, bestMask);
    result = croppedComponent;
    finalPreview = croppedComponent;

    if isempty(finalPreview)
        finalPreview = maskedComponent;
        result = maskedComponent;
    end

    if ~any(bestMask(:)) && isempty(croppedComponent)
        finalPreview = img;
        result = img;
    end

    bboxOverlay = draw_polygon_overlay(img, rotatedBox, [0 1 0]);
    resultTitle = sprintf('Selected cluster %d | aspect %.2f | rotated fill %.2f | area %.3f | OCR %s', ...
        bestClusterIdx, bestStats.aspectRatio, bestStats.rotatedFillRatio, bestStats.areaFraction, fallback_ocr_text(ocrText));

    figure('Name', 'oliver_v4 - Final Selected Segment', 'NumberTitle', 'off');
    montage({im2uint8(img), segmentedRGB, im2uint8(bboxOverlay), maskedComponent, finalPreview}, ...
        'Size', [1 5], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title(['Original | K-means labels | Rotated-box overlay | Selected component | ' resultTitle]);
end

function bestText = ocr_license_plate_block(croppedComponent)
% Run OCR on the deskewed crop using grayscale and binary plate-friendly variants.
    bestText = '';

    if isempty(croppedComponent)
        return;
    end

    cropRgb = im2uint8(croppedComponent);
    enlargedCrop = imresize(cropRgb, 3);
    grayCrop = rgb2gray(enlargedCrop);
    contrastCrop = imadjust(grayCrop);
    smoothedCrop = medfilt2(contrastCrop, [3 3]);
    otsuThreshold = graythresh(smoothedCrop);
    binaryCrop = imbinarize(smoothedCrop, otsuThreshold);
    binaryInvertedCrop = imcomplement(binaryCrop);
    adaptiveBinaryCrop = imbinarize(smoothedCrop, 'adaptive', 'Sensitivity', 0.45);
    adaptiveBinaryInvertedCrop = imcomplement(adaptiveBinaryCrop);
    equalizedCrop = histeq(grayCrop);
    laplacianResponse = imfilter(im2double(equalizedCrop), fspecial('laplacian', 0.2), 'replicate');
    sharpenedCrop = im2uint8(min(max(im2double(equalizedCrop) - 0.7 * laplacianResponse, 0), 1));
    sharpenedCrop = imadjust(sharpenedCrop);
    sharpenedBinaryCrop = imbinarize(sharpenedCrop, graythresh(sharpenedCrop));
    sharpenedBinaryInvertedCrop = imcomplement(sharpenedBinaryCrop);

    [grayText, grayScore, grayDebug] = run_plate_ocr_candidate(smoothedCrop, 'grayscale');
    [binaryText, binaryScore, binaryDebug] = run_plate_ocr_candidate(binaryCrop, 'binary-otsu');
    [invertedText, invertedScore, invertedDebug] = run_plate_ocr_candidate(binaryInvertedCrop, 'binary-otsu-inverted');
    [adaptiveText, adaptiveScore, adaptiveDebug] = run_plate_ocr_candidate(adaptiveBinaryCrop, 'binary-adaptive');
    [adaptiveInvText, adaptiveInvScore, adaptiveInvDebug] = run_plate_ocr_candidate(adaptiveBinaryInvertedCrop, 'binary-adaptive-inverted');
    [equalizedText, equalizedScore, equalizedDebug] = run_plate_ocr_candidate(equalizedCrop, 'equalized-gray');
    [sharpenedText, sharpenedScore, sharpenedDebug] = run_plate_ocr_candidate(sharpenedCrop, 'sharpened-gray');
    [sharpenedBinaryText, sharpenedBinaryScore, sharpenedBinaryDebug] = run_plate_ocr_candidate(sharpenedBinaryCrop, 'sharpened-binary');
    [sharpenedInvText, sharpenedInvScore, sharpenedInvDebug] = run_plate_ocr_candidate(sharpenedBinaryInvertedCrop, 'sharpened-binary-inverted');

    bestText = grayText;
    bestScore = grayScore;

    if binaryScore > bestScore
        bestText = binaryText;
        bestScore = binaryScore;
    end

    if invertedScore > bestScore
        bestText = invertedText;
        bestScore = invertedScore;
    end

    if adaptiveScore > bestScore
        bestText = adaptiveText;
        bestScore = adaptiveScore;
    end

    if adaptiveInvScore > bestScore
        bestText = adaptiveInvText;
        bestScore = adaptiveInvScore;
    end

    if equalizedScore > bestScore
        bestText = equalizedText;
        bestScore = equalizedScore;
    end

    if sharpenedScore > bestScore
        bestText = sharpenedText;
        bestScore = sharpenedScore;
    end

    if sharpenedBinaryScore > bestScore
        bestText = sharpenedBinaryText;
        bestScore = sharpenedBinaryScore;
    end

    if sharpenedInvScore > bestScore
        bestText = sharpenedInvText;
        bestScore = sharpenedInvScore;
    end

    fprintf('oliver_v4 OCR debug:\n');
    print_ocr_image_stats('crop-rgb', cropRgb);
    print_ocr_image_stats('gray', grayCrop);
    print_ocr_image_stats('contrast-median', smoothedCrop);
    print_ocr_image_stats('binary-otsu', binaryCrop);
    print_ocr_image_stats('binary-adaptive', adaptiveBinaryCrop);
    print_ocr_image_stats('equalized', equalizedCrop);
    print_ocr_image_stats('sharpened', sharpenedCrop);
    print_ocr_debug(grayDebug);
    print_ocr_debug(binaryDebug);
    print_ocr_debug(invertedDebug);
    print_ocr_debug(adaptiveDebug);
    print_ocr_debug(adaptiveInvDebug);
    print_ocr_debug(equalizedDebug);
    print_ocr_debug(sharpenedDebug);
    print_ocr_debug(sharpenedBinaryDebug);
    print_ocr_debug(sharpenedInvDebug);
    fprintf('  selected result: %s\n', fallback_ocr_text(bestText));

    grayPreview = repmat(grayCrop, [1 1 3]);
    contrastPreview = repmat(smoothedCrop, [1 1 3]);
    binaryPreview = repmat(uint8(binaryCrop) * 255, [1 1 3]);
    invertedPreview = repmat(uint8(binaryInvertedCrop) * 255, [1 1 3]);
    adaptivePreview = repmat(uint8(adaptiveBinaryCrop) * 255, [1 1 3]);
    adaptiveInvPreview = repmat(uint8(adaptiveBinaryInvertedCrop) * 255, [1 1 3]);
    equalizedPreview = repmat(equalizedCrop, [1 1 3]);
    sharpenedPreview = repmat(sharpenedCrop, [1 1 3]);
    sharpenedBinaryPreview = repmat(uint8(sharpenedBinaryCrop) * 255, [1 1 3]);
    sharpenedInvPreview = repmat(uint8(sharpenedBinaryInvertedCrop) * 255, [1 1 3]);

    figure('Name', 'oliver_v4 - OCR', 'NumberTitle', 'off');
    montage({cropRgb, enlargedCrop, grayPreview, contrastPreview, binaryPreview, invertedPreview, ...
        adaptivePreview, adaptiveInvPreview, equalizedPreview, sharpenedPreview, sharpenedBinaryPreview, sharpenedInvPreview}, ...
        'Size', [3 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title(sprintf('Crop | 3x resize | Gray | Contrast + median | Otsu | Otsu inv | Adaptive | Adaptive inv | Equalized | Sharpened | Sharp bin | Sharp bin inv | OCR: %s', ...
        fallback_ocr_text(bestText)));

    if isempty(bestText)
        fprintf('oliver_v4 OCR: no convincing plate text detected.\n');
    else
        fprintf('oliver_v4 OCR: %s\n', bestText);
    end
end

function [cleanedText, score, debugInfo] = run_plate_ocr_candidate(ocrInput, label)
% Restrict OCR to uppercase letters and digits, then score by confidence and text length.
    cleanedText = '';
    score = -Inf;
    debugInfo = struct( ...
        'label', label, ...
        'rawText', '', ...
        'cleanedText', '', ...
        'meanConfidence', NaN, ...
        'score', -Inf, ...
        'characterCount', 0, ...
        'wordCount', 0, ...
        'status', 'not-run');

    try
        result = ocr(ocrInput, ...
            'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', ...
            'LayoutAnalysis', 'line');
    catch ME
        debugInfo.status = sprintf('failed: %s', ME.message);
        return;
    end

    rawText = strtrim(result.Text);
    cleanedText = clean_plate_text(rawText);
    debugInfo.rawText = rawText;
    debugInfo.cleanedText = cleanedText;
    debugInfo.status = 'ok';

    if isprop(result, 'Words')
        debugInfo.wordCount = numel(result.Words);
    end

    if isempty(cleanedText)
        score = -Inf;
        debugInfo.score = score;
        return;
    end

    if isprop(result, 'CharacterConfidences')
        confidences = double(result.CharacterConfidences);
        confidences = confidences(isfinite(confidences));
    else
        confidences = [];
    end

    if isempty(confidences)
        meanConfidence = 0;
    else
        meanConfidence = mean(confidences);
    end

    debugInfo.meanConfidence = meanConfidence;
    debugInfo.characterCount = numel(cleanedText);
    score = meanConfidence + 0.05 * min(numel(cleanedText), 10);
    debugInfo.score = score;
end

function cleanedText = clean_plate_text(rawText)
% Normalize OCR output to a compact uppercase alphanumeric plate string.
    cleanedText = upper(regexprep(rawText, '[^A-Z0-9]', ''));

    if isempty(cleanedText)
        return;
    end

    cleanedText = regexprep(cleanedText, '^0+', '');
    cleanedText = regexprep(cleanedText, 'O(?=[0-9])', '0');
    cleanedText = regexprep(cleanedText, '(?<=[0-9])O', '0');
    cleanedText = regexprep(cleanedText, 'I(?=[0-9])', '1');
    cleanedText = regexprep(cleanedText, '(?<=[0-9])[IL]', '1');
end

function displayText = fallback_ocr_text(ocrText)
% Keep figure titles readable when OCR does not return anything useful.
    if isempty(ocrText)
        displayText = 'none';
    else
        displayText = ocrText;
    end
end

function print_ocr_debug(debugInfo)
% Print one compact OCR diagnostic line for a candidate input.
    fprintf('  [%s] status=%s | raw=\"%s\" | cleaned=\"%s\" | meanConf=%.3f | chars=%d | words=%d | score=%.3f\n', ...
        debugInfo.label, ...
        debugInfo.status, ...
        debugInfo.rawText, ...
        fallback_ocr_text(debugInfo.cleanedText), ...
        sanitize_debug_number(debugInfo.meanConfidence), ...
        debugInfo.characterCount, ...
        debugInfo.wordCount, ...
        sanitize_debug_number(debugInfo.score));
end

function print_ocr_image_stats(label, img)
% Print compact image diagnostics to explain why OCR may be failing.
    if islogical(img)
        nonzeroFraction = nnz(img) / numel(img);
        fprintf('  [stats:%s] size=%dx%d | foreground=%.3f\n', ...
            label, size(img, 2), size(img, 1), nonzeroFraction);
        return;
    end

    grayImg = img;
    if ndims(img) == 3
        grayImg = rgb2gray(img);
    end

    grayDouble = im2double(grayImg);
    fprintf('  [stats:%s] size=%dx%d | min=%.3f | max=%.3f | mean=%.3f | std=%.3f\n', ...
        label, size(grayImg, 2), size(grayImg, 1), ...
        min(grayDouble(:)), max(grayDouble(:)), mean(grayDouble(:)), std(grayDouble(:)));
end

function value = sanitize_debug_number(value)
% Keep debug output readable when values are NaN or Inf.
    if ~isfinite(value)
        value = -1;
    end
end

function segmentedTiles = build_segment_tiles(img, clusterLabels, clusterCount)
% Create one RGB preview per k-means cluster.
    segmentedTiles = cell(1, clusterCount);

    for clusterIdx = 1:clusterCount
        clusterMask = clusterLabels == clusterIdx;
        clusterPreview = img;

        for channelIdx = 1:size(img, 3)
            channelPlane = clusterPreview(:, :, channelIdx);
            channelPlane(~clusterMask) = 0;
            clusterPreview(:, :, channelIdx) = channelPlane;
        end

        segmentedTiles{clusterIdx} = clusterPreview;
    end
end

function [processedMask, bestComponentMask, bestStats] = find_best_component_in_cluster(rawMask, imageArea)
% Merge nearby foreground and score the most rectangle-like connected component.
    mergeElement = strel('rectangle', [5 21]);
    processedMask = imdilate(rawMask, mergeElement);
    processedMask = imerode(processedMask, mergeElement);
    processedMask = imfill(processedMask, 'holes');
    processedMask = remove_thin_lines_block(processedMask);
    processedMask = remove_edge_touching_components_block(rawMask, processedMask);

    conn = bwconncomp(processedMask);
    bestComponentMask = false(size(rawMask));
    bestStats = default_stats();

    for objectIdx = 1:conn.NumObjects
        componentMask = false(size(rawMask));
        componentMask(conn.PixelIdxList{objectIdx}) = true;
        stats = compute_component_stats(conn.PixelIdxList{objectIdx}, size(rawMask), imageArea);
        stats.score = score_component(stats);

        if stats.score > bestStats.score
            bestStats = stats;
            bestComponentMask = componentMask;
        end
    end
end

function cleanedMask = remove_thin_lines_block(mask)
% Suppress narrow line artifacts before connected-component analysis.
    thinLineElement = strel('rectangle', [3 3]);
    cleanedMask = imopen(mask, thinLineElement);
    cleanedMask = bwmorph(cleanedMask, 'clean');
end

function filteredMask = remove_edge_touching_components_block(rawMask, processedMask)
% Remove connected components that touch any image border.
    filteredMask = processedMask;
    conn = bwconncomp(processedMask);
    imageHeight = size(processedMask, 1);
    imageWidth = size(processedMask, 2);

    for objectIdx = 1:conn.NumObjects
        pixelIdxList = conn.PixelIdxList{objectIdx};
        [rows, cols] = ind2sub(size(processedMask), pixelIdxList);

        touchesTop = any(rows == 1);
        touchesBottom = any(rows == imageHeight);
        touchesLeft = any(cols == 1);
        touchesRight = any(cols == imageWidth);

        if touchesTop || touchesBottom || touchesLeft || touchesRight
            filteredMask(pixelIdxList) = false;
        end
    end
end

function stats = compute_component_stats(pixelIdxList, imageSize, imageArea)
% Compute bounding-box based geometry without regionprops.
    [rows, cols] = ind2sub(imageSize, pixelIdxList);

    rowMin = min(rows);
    rowMax = max(rows);
    colMin = min(cols);
    colMax = max(cols);

    width = colMax - colMin + 1;
    height = rowMax - rowMin + 1;
    area = numel(pixelIdxList);
    bboxArea = width * height;

    stats = default_stats();
    stats.area = area;
    stats.bbox = [rowMin, rowMax, colMin, colMax];
    stats.width = width;
    stats.height = height;
    stats.aspectRatio = width / max(height, eps);
    stats.fillRatio = area / max(bboxArea, 1);
    stats.areaFraction = area / imageArea;
    stats.widthFraction = width / imageSize(2);
end

function [rotatedBox, rotatedArea] = compute_rotated_bbox(componentMask)
% Estimate a minimum-area rotated rectangle from the component orientation.
    rotatedBox = [];
    rotatedArea = 0;

    if ~any(componentMask(:))
        return;
    end

    props = regionprops(componentMask, 'Orientation', 'Centroid');
    if isempty(props)
        return;
    end

    boundaryMask = bwmorph(componentMask, 'remove');
    [boundaryRows, boundaryCols] = find(boundaryMask);
    if isempty(boundaryRows)
        [boundaryRows, boundaryCols] = find(componentMask);
    end

    centroid = props(1).Centroid;
    theta = props(1).Orientation * pi / 180;

    centeredX = boundaryCols - centroid(1);
    centeredY = -(boundaryRows - centroid(2));

    alignRotation = [cos(theta), sin(theta); -sin(theta), cos(theta)];
    alignedPoints = alignRotation * [centeredX.'; centeredY.'];

    minX = min(alignedPoints(1, :));
    maxX = max(alignedPoints(1, :));
    minY = min(alignedPoints(2, :));
    maxY = max(alignedPoints(2, :));
    rotatedArea = (maxX - minX) * (maxY - minY);

    alignedCorners = [ ...
        minX, minY; ...
        maxX, minY; ...
        maxX, maxY; ...
        minX, maxY];

    inverseRotation = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    restoredCorners = inverseRotation * alignedCorners.';

    rotatedX = restoredCorners(1, :).'+ centroid(1);
    rotatedY = -restoredCorners(2, :).'+ centroid(2);
    rotatedBox = [rotatedX, rotatedY];
end

function fillRatio = compute_rotated_fill_ratio(componentMask, rotatedArea)
% Measure how tightly the component fills its rotated bounding rectangle.
    if ~any(componentMask(:)) || rotatedArea <= 0
        fillRatio = -Inf;
        return;
    end

    fillRatio = nnz(componentMask) / rotatedArea;
end

function [cropWidth, cropHeight] = rotated_box_size(rotatedBox)
% Estimate the axis-aligned crop size implied by the rotated rectangle edges.
    cropWidth = 0;
    cropHeight = 0;

    if isempty(rotatedBox) || size(rotatedBox, 1) < 4
        return;
    end

    topWidth = norm(rotatedBox(2, :) - rotatedBox(1, :));
    bottomWidth = norm(rotatedBox(3, :) - rotatedBox(4, :));
    leftHeight = norm(rotatedBox(4, :) - rotatedBox(1, :));
    rightHeight = norm(rotatedBox(3, :) - rotatedBox(2, :));

    cropWidth = max(1, round(mean([topWidth, bottomWidth])) + 1);
    cropHeight = max(1, round(mean([leftHeight, rightHeight])) + 1);
end

function croppedImg = sample_rotated_rectangle(img, rotatedBox, cropWidth, cropHeight)
% Reverse-map the rotated rectangle into a horizontal crop using affine sampling.
    croppedImg = zeros(cropHeight, cropWidth, size(img, 3), 'like', img);

    if isempty(rotatedBox) || cropWidth <= 0 || cropHeight <= 0
        return;
    end

    % compute_rotated_bbox returns corners in image order:
    % [bottom-left; bottom-right; top-right; top-left].
    sourceTriangle = [ ...
        rotatedBox(4, 1), rotatedBox(4, 2), 1; ...
        rotatedBox(3, 1), rotatedBox(3, 2), 1; ...
        rotatedBox(1, 1), rotatedBox(1, 2), 1].';
    destinationTriangle = [ ...
        1, 1, 1; ...
        cropWidth, 1, 1; ...
        1, cropHeight, 1].';

    destinationToSource = sourceTriangle * inv(destinationTriangle);

    [destCols, destRows] = meshgrid(1:cropWidth, 1:cropHeight);
    destinationHomogeneous = [destCols(:).'; destRows(:).'; ones(1, numel(destCols))];
    sourceCoords = destinationToSource * destinationHomogeneous;

    sourceCols = reshape(sourceCoords(1, :), cropHeight, cropWidth);
    sourceRows = reshape(sourceCoords(2, :), cropHeight, cropWidth);
    croppedImg = bilinear_sample_image(img, sourceRows, sourceCols);
end

function sampledImg = bilinear_sample_image(img, sampleRows, sampleCols)
% Bilinearly sample an RGB image at floating-point row/column locations.
    imageHeight = size(img, 1);
    imageWidth = size(img, 2);
    channelCount = size(img, 3);
    sampledImgDouble = zeros(size(sampleRows, 1), size(sampleRows, 2), channelCount);

    validMask = sampleRows >= 1 & sampleRows <= imageHeight & sampleCols >= 1 & sampleCols <= imageWidth;
    sampleRows = min(max(sampleRows, 1), imageHeight);
    sampleCols = min(max(sampleCols, 1), imageWidth);

    rowFloor = floor(sampleRows);
    rowCeil = min(rowFloor + 1, imageHeight);
    colFloor = floor(sampleCols);
    colCeil = min(colFloor + 1, imageWidth);

    rowWeight = sampleRows - rowFloor;
    colWeight = sampleCols - colFloor;

    for channelIdx = 1:channelCount
        channelPlane = double(img(:, :, channelIdx));
        topLeft = channelPlane(sub2ind([imageHeight, imageWidth], rowFloor, colFloor));
        topRight = channelPlane(sub2ind([imageHeight, imageWidth], rowFloor, colCeil));
        bottomLeft = channelPlane(sub2ind([imageHeight, imageWidth], rowCeil, colFloor));
        bottomRight = channelPlane(sub2ind([imageHeight, imageWidth], rowCeil, colCeil));

        topBlend = (1 - colWeight) .* topLeft + colWeight .* topRight;
        bottomBlend = (1 - colWeight) .* bottomLeft + colWeight .* bottomRight;
        sampledChannel = (1 - rowWeight) .* topBlend + rowWeight .* bottomBlend;
        sampledChannel(~validMask) = 0;
        sampledImgDouble(:, :, channelIdx) = sampledChannel;
    end

    sampledImg = cast_to_image_class(sampledImgDouble, img);
end

function score = score_component(stats)
% Prefer plate-like rectangles: wide, compact, and not too small or large.
    targetAspect = 4.0;
    maxAspectError = 4.5;
    targetFill = 0.90;
    minFillRatio = 0.40;
    minAreaFraction = 0.009;
    maxAreaFraction = 0.06;
    targetAreaFraction = 0.030;
    maxWidthFraction = 0.65;

    aspectScore = max(0, 1 - abs(stats.aspectRatio - targetAspect) / maxAspectError);
    fillScore = max(0, 1 - abs(stats.fillRatio - targetFill) / targetFill);

    if stats.areaFraction < minAreaFraction || stats.areaFraction > maxAreaFraction
        sizeScore = 0;
    else
        sizeScore = max(0, 1 - abs(stats.areaFraction - targetAreaFraction) / (0.60 * targetAreaFraction));
    end

    if stats.areaFraction < targetAreaFraction
        smallPenalty = min(1, (targetAreaFraction - stats.areaFraction) / targetAreaFraction);
    else
        smallPenalty = 0;
    end

    score = 0.15 * aspectScore + 0.30 * fillScore + 0.55 * sizeScore - 0.20 * smallPenalty;

    if stats.aspectRatio < 1.6 || ...
       stats.aspectRatio > 9.0 || ...
       stats.areaFraction < minAreaFraction || ...
       stats.fillRatio < minFillRatio || ...
       stats.widthFraction > maxWidthFraction
        score = -Inf;
    end
end

function overlay = draw_polygon_overlay(img, polygon, color)
% Draw a thin quadrilateral directly into an RGB image.
    overlay = im2double(img);

    if isempty(polygon) || any(isnan(polygon(:)))
        return;
    end

    for cornerIdx = 1:size(polygon, 1)
        nextIdx = mod(cornerIdx, size(polygon, 1)) + 1;
        overlay = draw_line_segment(overlay, polygon(cornerIdx, :), polygon(nextIdx, :), color);
    end
end

function overlay = draw_line_segment(overlay, startPoint, endPoint, color)
% Rasterize a single line segment between two floating-point vertices.
    pointCount = max(ceil(max(abs(endPoint - startPoint))), 1) + 1;
    cols = round(linspace(startPoint(1), endPoint(1), pointCount));
    rows = round(linspace(startPoint(2), endPoint(2), pointCount));

    validMask = rows >= 1 & rows <= size(overlay, 1) & cols >= 1 & cols <= size(overlay, 2);
    rows = rows(validMask);
    cols = cols(validMask);

    if isempty(rows)
        return;
    end

    pixelIdx = sub2ind([size(overlay, 1), size(overlay, 2)], rows, cols);

    for channelIdx = 1:size(overlay, 3)
        channelPlane = overlay(:, :, channelIdx);
        channelPlane(pixelIdx) = color(channelIdx);
        overlay(:, :, channelIdx) = channelPlane;
    end
end

function rgb = mask_to_rgb(mask)
% Convert a logical mask into a 3-channel preview for montage.
    rgb = repmat(im2uint8(mask), [1 1 3]);
end

function maskedImg = apply_mask_to_image(img, mask)
% Zero everything outside the selected logical mask.
    maskedImg = img;

    for channelIdx = 1:size(img, 3)
        channelPlane = maskedImg(:, :, channelIdx);
        channelPlane(~mask) = 0;
        maskedImg(:, :, channelIdx) = channelPlane;
    end
end

function castImg = cast_to_image_class(imgDouble, referenceImg)
% Cast floating-point sampled values back to the input image class.
    if isa(referenceImg, 'uint8')
        castImg = uint8(min(max(round(imgDouble), 0), 255));
    elseif isa(referenceImg, 'uint16')
        castImg = uint16(min(max(round(imgDouble), 0), 65535));
    elseif isa(referenceImg, 'single')
        castImg = single(imgDouble);
    else
        castImg = cast(imgDouble, 'like', referenceImg);
    end
end

function stats = default_stats()
% Default empty component stats.
    stats = struct( ...
        'score', -Inf, ...
        'area', 0, ...
        'bbox', [], ...
        'width', 0, ...
        'height', 0, ...
        'aspectRatio', 0, ...
        'fillRatio', 0, ...
        'rotatedFillRatio', 0, ...
        'areaFraction', 0, ...
        'widthFraction', 0);
end
