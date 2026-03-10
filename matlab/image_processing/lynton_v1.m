
% THE ORIGINAL CODE SENT BY LYNTON - No edits so far
function processedImg = lynton_v1(img)
%lynton_v1 UK number plate reader for yellow (rear) + white (front).
%   Input: RGB image array.
%   Output: cropped plate RGB (if found) or original image as fallback.

    if size(img,3) ~= 3
        error('Image must be RGB (3 channels)');
    end

    figure('Name','1 - Original','NumberTitle','off');
    imshow(img); title('Original image');

    %% Convert color spaces
    gray = rgb2gray(img);
    hsvImg  = rgb2hsv(img);

    %% Dual color detection: yellow (rear) + white (front)

    % Yellow branch (rear plate)
    yellowMask = (hsvImg(:,:,1) >= 0.08 & hsvImg(:,:,1) <= 0.22) & ...
                 (hsvImg(:,:,2) >= 0.65) & ...
                 (hsvImg(:,:,3) >= 0.35);

    yellowMask = imopen(yellowMask, strel('disk', 4));
    yellowMask = imclose(yellowMask, strel('rectangle',[6 25]));
    yellowMask = imfill(yellowMask, 'holes');
    yellowMask = bwareaopen(yellowMask, 1200);
    yellowMask = imdilate(yellowMask, strel('rectangle',[3 12]));

    % White branch (front plate)
    whiteMask = (hsvImg(:,:,2) <= 0.18) & ...           % allow some tint
                (hsvImg(:,:,3) >= 0.78);                % bright but not extreme

    whiteMask = imopen(whiteMask, strel('disk', 5));
    whiteMask = imclose(whiteMask, strel('rectangle',[6 25]));
    whiteMask = imfill(whiteMask, 'holes');
    whiteMask = bwareaopen(whiteMask, 1200);
    whiteMask = imdilate(whiteMask, strel('rectangle',[3 12]));

    %% Morphology cleanup – keep largest region only
    yellowMask = bwareafilt(yellowMask, 1, 'largest');
    whiteMask = bwareafilt(whiteMask, 1, 'largest');

    fprintf('After cleanup:\n');
    fprintf(' Yellow pixels: %.2f%%\n', 100 * nnz(yellowMask)/numel(yellowMask));
    fprintf(' White pixels:  %.2f%%\n', 100 * nnz(whiteMask)/numel(whiteMask));

    %% Visualize masks (safe – no crash if no selection yet)
    figure('Name', '5 – Yellow vs White Masks', 'NumberTitle', 'off', ...
           'Position', [150 150 1400 800]);

    subplot(2,3,1); imshow(yellowMask); title('Yellow Mask');
    subplot(2,3,2); imshow(whiteMask);  title('White Mask');
    subplot(2,3,3); imshow(imfuse(yellowMask, whiteMask, 'falsecolor', ...
                                  'ColorChannels',[1 2 0]));
    title('Combined (yellow=red, white=green)');

    subplot(2,3,4); imshow(labeloverlay(img, yellowMask, 'Transparency',0.65,'Colormap',[1 0.5 0]));
    title('Yellow overlaid');
    subplot(2,3,5); imshow(labeloverlay(img, whiteMask, 'Transparency',0.65,'Colormap',[0 0.8 1]));
    title('White overlaid');

    subplot(2,3,6); imshow(img); title('Original + ROI (if selected)');

    sgtitle('Yellow vs White Detection Debug', 'FontSize',16);

    %% Find largest plate-like candidate (relaxed for skew)
    stats = regionprops(yellowMask, 'BoundingBox', 'Area', 'Eccentricity', 'Extent', 'Orientation');

    if isempty(stats)
        warning('No yellow regions found – trying text-based fallback.');
        bboxes = detectTextCRAFT(gray, CharacterThreshold=0.3, LinkThreshold=0.5, MinSize=[20 40]);

        if isempty(bboxes)
            warning('No text regions found. Manual crop recommended.');
            processedImg = img;
            return;
        end

        % Pick largest text region (likely plate)
        areas = prod(bboxes(:,3:4), 2);
        [~, idx] = max(areas);
        roi = bboxes(idx,:);
        orientation = 0;  % no orientation from CRAFT
        fprintf('Used CRAFT fallback – largest text region.\n');
    else
        % Extract properties
        areas       = [stats.Area];
        ecc         = [stats.Eccentricity];
        ext         = [stats.Extent];
        orient      = [stats.Orientation];
        bb          = vertcat(stats.BoundingBox);
        aspectRatio = bb(:,3) ./ bb(:,4);

        % Relaxed filters for skew
        valid = (areas > 1000)          & (areas < 100000) & ...
                (aspectRatio >= 2.0)    & (aspectRatio <= 6.0) & ...
                (ecc < 0.95)            & ...
                (ext > 0.4);

        if any(valid)
            validIdx = find(valid);
            [~, best] = max(areas(validIdx));
            idx = validIdx(best);

            fprintf('Selected candidate:\n');
            fprintf('  Area = %d px\n', areas(idx));
            fprintf('  Aspect ratio = %.2f\n', aspectRatio(idx));
            fprintf('  Extent = %.2f\n', ext(idx));
            fprintf('  Eccentricity = %.3f\n', ecc(idx));
            fprintf('  Orientation = %.1f degrees\n', orient(idx));
        else
            [~, idx] = max(areas);
            fprintf('No strict match – using largest (Area = %d)\n', areas(idx));
        end

        roi = round(stats(idx).BoundingBox);
        orientation = orient(idx);
    end

    % Padding
    pad = 20;
    roi = [roi(1)-pad, roi(2)-pad, roi(3)+2*pad, roi(4)+2*pad];

    % Clip
    roi(1) = max(1, roi(1));
    roi(2) = max(1, roi(2));
    roi(3) = min(size(img,2) - roi(1) + 1, roi(3));
    roi(4) = min(size(img,1) - roi(2) + 1, roi(4));

    %% Crop and deskew
    plate_rgb  = imcrop(img, roi);
    plate_gray = imcrop(gray, roi);

    if abs(orientation) > 0.5
        plate_gray = imrotate(plate_gray, -orientation, 'bilinear', 'crop');
        plate_rgb  = imrotate(plate_rgb, -orientation, 'bilinear', 'crop');
        fprintf('Deskewed by %.1f degrees.\n', -orientation);
    end

    figure('Name','Cropped & Deskewed','NumberTitle','off');
    imshow(plate_rgb); title('Selected Crop');


    %% Preprocess
    bw = imbinarize(plate_gray, "adaptive", "sensitivity", 0.99);
    bw = imcomplement(bw);
    bw = imopen(bw, strel('disk',10));
    bw = imclose(bw, strel('rectangle',[5 10]));


    figure('Name','4 - Preprocessed','NumberTitle','off');
    montage({plate_gray, bw}, 'Size',[1 2], 'BackgroundColor','w', 'BorderSize',10);
    title('Grayscale | Binary inverted');


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

    disp(' ');
    disp('If result is empty/wrong:');
    disp(' • Try manual crop (code below)');
    disp(' • Lower sensitivity to 0.95–0.97 if too noisy');
    disp(' • Increase sharpen Amount/Radius');

    %% Optional manual crop rescue
    disp('Manual crop rescue (uncomment if needed):');
    % figure; imshow(img);
    % title('Tight crop around plate → double-click');

    processedImg = plate_rgb;
end
