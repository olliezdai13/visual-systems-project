% I'm not happy with this overall approach. Scrapping this.

function processedImg = oliver_v1(img)
% Basic plate isolation: grayscale → vertical Sobel → closing → aspect filter.
% Collects intermediate outputs and shows a single montage at the end.

    gray = convert_to_grayscale(img);
    verticalEdges = detect_vertical_edges(gray);
    filteredEdges = suppress_noise(verticalEdges);
    edgeMask = binarize_edges(filteredEdges);
    cleanedMask = remove_salt_noise(edgeMask);
    closedMask = close_gaps(cleanedMask);

    show_montage({img, gray, verticalEdges, filteredEdges, edgeMask, cleanedMask, closedMask}, ...
                 {'Original','Grayscale','Vertical Sobel','Median Filter','Binary Mask','Morph Filter','Closing'});

    processedImg = closedMask;
end

function gray = convert_to_grayscale(img)
% Normalize to grayscale double for processing and show result.
    gray = im2double(im2gray(img));
end

function verticalEdges = detect_vertical_edges(gray)
% Emphasize vertical structure with Sobel x-derivative and show magnitude.
    [Gx, ~] = imgradientxy(gray, 'sobel');
    verticalEdges = mat2gray(abs(Gx));
end

function filteredEdges = suppress_noise(verticalEdges)
% Median filtering to knock down fine texture noise while preserving edges.
    filteredEdges = medfilt2(verticalEdges, [3 3]);
end

function filteredEdges = suppress_noise_v2(verticalEdges)
% Low-pass (box) filter from Lab3-Intensity-transformation Task 4 to smooth noise.
    kernel = fspecial('average', [9 9]);
    filteredEdges = imfilter(verticalEdges, kernel, 0);
end

function edgeMask = binarize_edges(verticalEdges)
% Threshold vertical edge map into a binary mask and visualize it.
    edgeMask = imbinarize(verticalEdges, graythresh(verticalEdges));
end

function cleanedMask = remove_salt_noise(edgeMask)
% Erode then dilate with a small square SE to delete 1-2 px salt noise.
    se = strel('square', 4);
    eroded = imerode(edgeMask, se);
    cleanedMask = imdilate(eroded, se);
end

function closedMask = close_gaps(edgeMask)
% Morphological closing to fuse nearby strokes and fill gaps.
    se = strel('rectangle', [8 16]);
    closedMask = imclose(edgeMask, se);
end

function show_montage(steps, titles)
% Display a single montage of pipeline progression with labels.
    n = numel(steps);
    rows = 1; cols = n;
    figure('Name','oliver_v1 - Pipeline','NumberTitle','off');
    t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:n
        nexttile;
        imshow(steps{i}, []);
        title(titles{i});
    end
    title(t, 'oliver\_v1 pipeline progression');
end
