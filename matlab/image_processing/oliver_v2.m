% I'm happy with this overall approach. But it doesn't work well enough on angled plates. V3 will fix that.

function result = oliver_v2(img)
% HSV-based plate isolation followed by aggressive erosion + reconstruction to keep only large regions.
    plateMask = plate_masking_block(img);
    largeComponents = recover_large_components_block(plateMask);
    best_geometry_mask = plate_geometry_filter_block(largeComponents);
    filteredMask = hough_filter_rectangle(best_geometry_mask);
    [rectMask, rectBBox, lineSet, corners] = hough_find_corners(filteredMask);

    [croppedPlate, cropOffset] = crop_plate_block(img, rectBBox, rectMask);
    rectifiedPlate = rectify_plate_block(croppedPlate, corners, cropOffset);

    result = rectifiedPlate;
end

function plateMask = plate_masking_block(img)
% Full masking pipeline: HSV convert, color gates, denoise, morphology, and montage.
    hsvImg = rgb2hsv(img);
    hue = hsvImg(:, :, 1);
    sat = hsvImg(:, :, 2);
    val = hsvImg(:, :, 3);

    hueLower = 0.10;   % about 36 degrees
    hueUpper = 0.18;   % about 65 degrees
    yellowMask = (hue >= hueLower) & (hue <= hueUpper) & (sat >= 0.35) & (val >= 0.35);

    % Stricter white threshold
    % whiteMask = (sat <= 0.18) & (val >= 0.80);

    % Relaxed white threshold
    whiteMask = (sat <= 0.25) & (val >= 0.65);

    combinedMask = yellowMask | whiteMask;

    se = strel('rectangle', [5 9]);
    closedMask = imclose(combinedMask, se);

    plateMask = imfill(closedMask, 'holes');

    steps = {img, hue, sat, val, yellowMask, whiteMask, combinedMask, closedMask, plateMask};
    titles = {'Original','Hue','Saturation','Value','Yellow Gate','White Gate','Combined','Closing','Filled'};
    figure('Name','oliver_v2 - Pipeline','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'oliver_v2 pipeline progression');
end

function refinedMask = recover_large_components_block(mask)
% Remove thin/noisy bits via strong erosion, then geodesic dilation under the original mask to restore only large structures.
    strongSe = strel('rectangle', [12 18]);
    erodedMask = imerode(mask, strongSe);

    % Use eroded mask as the marker; original mask is the constraint for geodesic dilation.
    refinedMask = imreconstruct(erodedMask, mask);

    steps = {mask, erodedMask, refinedMask};
    titles = {'Input Mask','Strong Erosion','Reconstructed (Geodesic Dilation)'};
    figure('Name','oliver_v2 - Large Component Recovery','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'Aggressive erosion + reconstruction to keep large components');
end

function plateMask = plate_geometry_filter_block(mask)
% Keep only connected components that match license plate geometry (aspect, area, extent).
    % Close thin horizontal gaps so fractured plates are merged before geometry checks.
    closeSe = strel('line', 15, 0);
    closedMask = imclose(mask, closeSe);

    conn = bwconncomp(closedMask);
    props = regionprops(conn, 'BoundingBox', 'Area', 'Extent');

    totalPixels = numel(closedMask);
    minArea = 0.001 * totalPixels; % small noise rejection
    maxArea = 0.10  * totalPixels; % avoid swallowing the whole car

    keepIdx = false(conn.NumObjects, 1);
    for i = 1:conn.NumObjects
        bb = props(i).BoundingBox;
        width = bb(3);
        height = bb(4);
        aspect = width / max(height, eps);

        area = props(i).Area;
        extent = props(i).Extent;

        if aspect >= 3 && aspect <= 6.0 && ...
           area >= minArea && area <= maxArea && ...
           extent >= 0.55
            keepIdx(i) = true;
        end
    end

    % Build an overlay showing all component bounding boxes before filtering.
    bboxOverlay = repmat(closedMask, [1 1 3]); % start from binary mask as RGB
    for i = 1:conn.NumObjects
        bb = props(i).BoundingBox;
        r0 = max(1, floor(bb(2)));
        c0 = max(1, floor(bb(1)));
        r1 = min(size(mask, 1), ceil(bb(2) + bb(4) - 1));
        c1 = min(size(mask, 2), ceil(bb(1) + bb(3) - 1));
        bboxOverlay([r0 r1], c0:c1, 1) = 1; % top/bottom red
        bboxOverlay(r0:r1, [c0 c1], 1) = 1; % left/right red
    end
    bboxOverlay = im2double(bboxOverlay); % imshow expects numeric for 3-channel

    plateMask = false(size(closedMask));
    if any(keepIdx)
        keptPixels = vertcat(conn.PixelIdxList{keepIdx});
        plateMask(keptPixels) = true;
    end

    % Visualize raw mask, filtered mask, and labeled kept blobs.
    keptLabels = labelmatrix(conn);
    if conn.NumObjects > 0
        keptLabels(~ismember(keptLabels, find(keepIdx))) = 0;
    end
    labelOverlay = label2rgb(keptLabels, 'spring', 'k', 'shuffle');

    steps = {mask, closedMask, bboxOverlay, plateMask, labelOverlay};
    titles = {'Input Mask','Closed (Horizontal Line)','Bounding Boxes','Geometry Filtered','Kept Components'};
    figure('Name','oliver_v2 - Plate Geometry Filter','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'Aspect/area/extent filter for plate-like blobs');
end

function filteredMask = hough_filter_rectangle(mask)
% Use top 3 Hough peaks as seeds and reconstruct within original mask to suppress stray blobs.
    mask = logical(mask);

    edges = edge(mask, 'canny');

    [H, theta, rho] = hough(edges);

    peakCount = 3;
    peakThresh = 0.35 * max(H(:));
    peaks = houghpeaks(H, peakCount, 'Threshold', peakThresh);

    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 20, 'MinLength', 15);

    seedMask = false(size(mask));
    for i = 1:numel(lines)
        p1 = round(lines(i).point1);
        p2 = round(lines(i).point2);

        p1(1) = min(max(p1(1), 1), size(mask, 2));
        p1(2) = min(max(p1(2), 1), size(mask, 1));
        p2(1) = min(max(p2(1), 1), size(mask, 2));
        p2(2) = min(max(p2(2), 1), size(mask, 1));

        seedMask(p1(2), p1(1)) = true;
        seedMask(p2(2), p2(1)) = true;
    end

    % Slightly dilate seeds so reconstruction can latch onto nearby structure.
    seedMask = imdilate(seedMask, strel('disk', 2));

    if ~any(seedMask(:))
        % If no peaks/lines landed inside the mask, keep the input to avoid a blank result.
        filteredMask = mask;
    else
        filteredMask = imreconstruct(seedMask, mask);
    end

    steps = {mask, edges, seedMask, filteredMask};
    titles = {'Input Mask','Canny Edges','Seeds from 3 Hough Peaks','Reconstructed Mask'};
    figure('Name','oliver_v2 - Hough Filter Rectangle','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing','compact','Padding','compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
        hold on;
        if i == 3
            for k = 1:numel(lines)
                L = lines(k);
                plot([L.point1(1) L.point2(1)], [L.point1(2) L.point2(2)], 'r-', 'LineWidth', 2);
            end
        end
        hold off;
    end
    title(t, '3-peak Hough seeding + geodesic reconstruction');
end

function [rectMask, rectBBox, lines, corners] = hough_find_corners(mask)
% Canny → Hough lines → use extreme endpoints to hypothesize plate rectangle.
    mask = logical(mask);

    % Step 1: edges
    edges = edge(mask, 'canny');

    % Step 2: Hough transform
    [H, theta, rho] = hough(edges);

    % Step 3: peak picking and line extraction
    peakCount   = 30;
    peakThresh  = 0.30 * max(H(:));
    neighborhood = [7 7];
    peaks = houghpeaks(H, peakCount, 'Threshold', peakThresh, 'NHoodSize', neighborhood);

    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 30, 'MinLength', 20);

    rectMask = false(size(mask));
    rectBBox = struct('xLeft', [], 'xRight', [], 'yTop', [], 'yBottom', []);
    corners = compute_line_corners(lines);

    if ~isempty(corners)
        xLeft   = max(1, round(min(corners(:,1))));
        xRight  = min(size(mask,2), round(max(corners(:,1))));
        yTop    = max(1, round(min(corners(:,2))));
        yBottom = min(size(mask,1), round(max(corners(:,2))));

        rectMask(yTop:yBottom, xLeft:xRight) = true;
        rectBBox = struct('xLeft', xLeft, 'xRight', xRight, 'yTop', yTop, 'yBottom', yBottom);
    else
        % Fallback: use bounding box of largest component (no corner estimate).
        conn = bwconncomp(mask);
        if conn.NumObjects > 0
            stats = regionprops(conn, 'BoundingBox', 'Area');
            [~, idx] = max([stats.Area]);
            bb = stats(idx).BoundingBox;
            yTop    = max(1, round(bb(2)));
            xLeft   = max(1, round(bb(1)));
            yBottom = min(size(mask,1), round(bb(2) + bb(4) - 1));
            xRight  = min(size(mask,2), round(bb(1) + bb(3) - 1));

            rectMask(yTop:yBottom, xLeft:xRight) = true;
            rectBBox = struct('xLeft', xLeft, 'xRight', xRight, 'yTop', yTop, 'yBottom', yBottom);
        end
    end

    % Visualization montage (draw overlays with plot to avoid toolbox helpers)
    rgbMask = im2double(cat(3, mask, mask, mask)); % make 3-channel double so imshow accepts it
    steps = {mask, edges, rgbMask, rectMask, rgbMask};
    titles = {'Input Mask','Canny Edges','Detected Lines','Rectangle Mask','Lines + Rectangle'};

    figure('Name','oliver_v2 - Hough Corner Finder','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing','compact','Padding','compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
        hold on;
        if i == 3 || i == 5
            % overlay all detected lines
            for k = 1:numel(lines)
                L = lines(k);
                plot([L.point1(1) L.point2(1)], [L.point1(2) L.point2(2)], 'r-', 'LineWidth', 2);
            end
        end
        if i == 5 && ~isempty(rectBBox.xLeft)
            rectangle('Position', [rectBBox.xLeft, rectBBox.yTop, ...
                                   rectBBox.xRight - rectBBox.xLeft + 1, ...
                                   rectBBox.yBottom - rectBBox.yTop + 1], ...
                      'EdgeColor','y', 'LineWidth', 2);
            if ~isempty(corners)
                plot(corners(:,1), corners(:,2), 'go', 'MarkerSize', 6, 'LineWidth', 1.5);
            end
        end
        hold off;
    end
    title(t, 'Canny → Hough → Rectangle hypothesis');
end

function [croppedPlate, offset] = crop_plate_block(img, rectBBox, rectMask)
% Crop the original RGB image using the rectangle hypothesis.
    croppedPlate = [];
    offset = [0 0];

    if isempty(rectBBox.xLeft)
        % Nothing to crop; still show a montage for consistency.
        steps = {img, rectMask};
        titles = {'Original Image','Rectangle Mask (none found)'};
    else
        rows = rectBBox.yTop:rectBBox.yBottom;
        cols = rectBBox.xLeft:rectBBox.xRight;
        croppedPlate = img(rows, cols, :);
        offset = [rectBBox.xLeft - 1, rectBBox.yTop - 1]; % zero-based for later corner shift

        % Build a simple overlay to show the crop box on the full image.
        overlay = im2double(img);
        maskRgb = cat(3, rectMask, rectMask, zeros(size(rectMask)));
        overlay = min(overlay + 0.35 * maskRgb, 1);

        steps = {img, overlay, croppedPlate};
        titles = {'Original Image','Crop Overlay','Cropped Plate'};
    end

    figure('Name','oliver_v2 - Cropping','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing','compact','Padding','compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'Crop original image to rectangle hypothesis');
end

function rectifiedPlate = rectify_plate_block(croppedPlate, corners, offset)
% Use the Hough-derived corners to warp the cropped plate to a rectangle.
    rectifiedPlate = [];

    if isempty(croppedPlate) || isempty(corners)
        steps = {croppedPlate};
        titles = {'Cropped plate (missing for rectification)'};
    else
        % Shift corner coordinates into the cropped frame.
        cropCorners = corners - offset;

        % Guard against out-of-bounds corners.
        cropCorners(:,1) = min(max(cropCorners(:,1), 1), size(croppedPlate,2));
        cropCorners(:,2) = min(max(cropCorners(:,2), 1), size(croppedPlate,1));

        % Estimate target rectangle size from corner pairs.
        topWidth    = norm(cropCorners(2,:) - cropCorners(1,:));
        bottomWidth = norm(cropCorners(3,:) - cropCorners(4,:));
        leftHeight  = norm(cropCorners(4,:) - cropCorners(1,:));
        rightHeight = norm(cropCorners(3,:) - cropCorners(2,:));

        targetWidth  = max(1, round(max([topWidth, bottomWidth])));
        targetHeight = max(1, round(max([leftHeight, rightHeight])));

        destPts = [1 1; targetWidth 1; targetWidth targetHeight; 1 targetHeight];

        % Fit projective transform and warp.
        tform = fitgeotrans(cropCorners, destPts, 'projective');
        outputRef = imref2d([targetHeight, targetWidth]);
        rectifiedPlate = imwarp(croppedPlate, tform, 'OutputView', outputRef);

        % For visualization: overlay corner markers on cropped plate.
        plateWithCorners = insert_shape_points(croppedPlate, cropCorners);

        steps = {plateWithCorners, rectifiedPlate};
        titles = {'Cropped with Corners','Rectified Plate'};
    end

    figure('Name','oliver_v2 - Rectification','NumberTitle','off');
    t = tiledlayout(1, numel(steps), 'TileSpacing','compact','Padding','compact');
    for i = 1:numel(steps)
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'Perspective warp to rectangularize plate');
end

function corners = compute_line_corners(lines)
% Pick plate corners from endpoints of all detected Hough lines.
    if isempty(lines)
        corners = [];
        return;
    end

    % Collect endpoints from every detected line.
    pts = zeros(numel(lines) * 2, 2);
    for i = 1:numel(lines)
        pts(2*i-1, :) = lines(i).point1;
        pts(2*i,   :) = lines(i).point2;
    end

    % Corner selection using sum/difference heuristics to reduce duplicate picks.
    sums  = pts(:,1) + pts(:,2);
    diffs = pts(:,1) - pts(:,2);

    [~, idxTL] = min(sums);   % Top-Left: smallest x+y
    [~, idxBR] = max(sums);   % Bottom-Right: largest x+y
    [~, idxTR] = max(diffs);  % Top-Right: largest x-y (large x, small y)
    [~, idxBL] = min(diffs);  % Bottom-Left: smallest x-y (small x, large y)

    corners = [
        pts(idxTL, :);  % Top-Left
        pts(idxTR, :);  % Top-Right
        pts(idxBR, :);  % Bottom-Right
        pts(idxBL, :);  % Bottom-Left
    ];

    if any(isnan(corners(:))) || any(isempty(corners(:)))
        corners = [];
    end
end

function imgOut = insert_shape_points(imgIn, pts)
% Lightweight corner overlay without toolbox helpers.
    imgOut = im2double(imgIn);
    radius = 4;
    for i = 1:size(pts,1)
        xc = round(pts(i,1));
        yc = round(pts(i,2));
        rr = max(1, yc - radius):min(size(imgOut,1), yc + radius);
        cc = max(1, xc - radius):min(size(imgOut,2), xc + radius);
        imgOut(rr, cc, 1) = 1; % highlight in red
        imgOut(rr, cc, 2) = 0;
        imgOut(rr, cc, 3) = 0;
    end
end
