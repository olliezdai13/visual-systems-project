function result = oliver_v4(img)
% K-means plate candidate selection using simplified rectangle scoring.

    validateattributes(img, {'uint8', 'uint16', 'single', 'double'}, ...
        {'nonempty', 'size', [NaN NaN 3]}, mfilename, 'img');

    [clusterLabels, segmentedRGB] = kmeans_segmentation_block(img);
    nonRedMask = exclude_red_components_block(img);
    [bestMask, bestClusterIdx, bestStats, bestComponentMasks] = select_plate_component_block(img, clusterLabels, nonRedMask);
    rotatedBoxes = rotated_bbox_candidates_block(bestComponentMasks);

    selectedRotatedBox = [];
    if bestClusterIdx > 0
        selectedRotatedBox = rotatedBoxes{bestClusterIdx};
    end

    result = render_selected_component_block(img, segmentedRGB, bestMask, bestClusterIdx, bestStats, selectedRotatedBox);
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

function [bestMask, bestClusterIdx, bestStats, bestComponentMasks] = select_plate_component_block(img, clusterLabels, nonRedMask)
% Score k-means cluster components by simple rectangular geometry after red suppression.
    clusterCount = max(clusterLabels(:));
    imageArea = size(img, 1) * size(img, 2);
    bestScore = -Inf;
    bestMask = false(size(clusterLabels));
    bestClusterIdx = 0;
    bestStats = default_stats();
    bestComponentMasks = cell(1, clusterCount);

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

        if candidateStats.score > bestScore
            bestScore = candidateStats.score;
            bestMask = candidateMask;
            bestClusterIdx = clusterIdx;
            bestStats = candidateStats;
        end
    end

    if bestClusterIdx == 0
        bestMask = false(size(clusterLabels));
        bestStats = default_stats();
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

function rotatedBoxes = rotated_bbox_candidates_block(componentMasks)
% Compute and display rotated bounding boxes for each selected cluster component.
    clusterCount = numel(componentMasks);
    rotatedBoxes = cell(1, clusterCount);
    overlayTiles = cell(1, clusterCount);

    for clusterIdx = 1:clusterCount
        componentMask = componentMasks{clusterIdx};
        rotatedBoxes{clusterIdx} = compute_rotated_bbox(componentMask);
        overlayTiles{clusterIdx} = draw_polygon_overlay(mask_to_rgb(componentMask), rotatedBoxes{clusterIdx}, [0 1 0]);
    end

    figure('Name', 'oliver_v4 - Rotated Bounding Boxes', 'NumberTitle', 'off');
    montage(overlayTiles, ...
        'Size', [2 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title('Best component per cluster with rotated bounding-box overlays');
end

function result = render_selected_component_block(img, segmentedRGB, bestMask, bestClusterIdx, bestStats, rotatedBox)
% Return the selected segmented region and show the final extraction.
    result = img;

    for channelIdx = 1:size(img, 3)
        channelPlane = result(:, :, channelIdx);
        channelPlane(~bestMask) = 0;
        result(:, :, channelIdx) = channelPlane;
    end

    bboxOverlay = draw_polygon_overlay(img, rotatedBox, [0 1 0]);
    resultTitle = sprintf('Selected cluster %d | aspect %.2f | fill %.2f | area %.3f', ...
        bestClusterIdx, bestStats.aspectRatio, bestStats.fillRatio, bestStats.areaFraction);

    figure('Name', 'oliver_v4 - Final Selected Segment', 'NumberTitle', 'off');
    montage({im2uint8(img), segmentedRGB, bboxOverlay, result}, ...
        'Size', [1 4], ...
        'BackgroundColor', 'white', 'BorderSize', 8);
    title(['Original | K-means labels | Selected component overlay | ' resultTitle]);
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

function rotatedBox = compute_rotated_bbox(componentMask)
% Estimate a minimum-area rotated rectangle from the component orientation.
    rotatedBox = [];

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
        'areaFraction', 0, ...
        'widthFraction', 0);
end
