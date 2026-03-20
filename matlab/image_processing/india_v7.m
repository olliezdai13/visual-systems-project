function processedImg = india_v7(img)
% Boilerplate controller for India pipeline v7.

    validateattributes(img, {'uint8', 'uint16', 'single', 'double', 'logical'}, ...
        {'nonempty'}, mfilename, 'img');

    india_v7_selected_image('set', img);
    processedImg = india_v7_experimental_block();
end

function processedImg = india_v7_experimental_block()
% Experimental block wrapped without changing the original script lines.

%trying something new

testName = "C:\Users\44793\Documents\VS Project\visual-systems-project\dataset\040603\P1010001.jpg";
licensePlate = imread(testName);
whos licensePlate
imshow(licensePlate)

licenseplateGrey = rgb2gray(licensePlate);
max(licenseplateGrey(:))

histogram(licenseplateGrey)
% Apply a threshold to create a binary image
threshold = 72; % Example threshold value
binaryImage = licenseplateGrey > threshold;
licenseplateBW = bwareaopen(binaryImage, 70);
imshow(licenseplateBW)

whiteCountPerRow = sum(licenseplateBW,2);
%colThreshold = mean(whiteCountPerCol) * 1.1; % Adjust the 0.9 to be more/less sensitive
%regions = whiteCountPerCol > colThreshold;
plot(1: length(whiteCountPerRow),whiteCountPerRow)
regions = 253 > whiteCountPerRow > 352;
hold on
plot(regions*400)
hold off
plot(whiteCountPerRow)
xlabel('Row Number (up-down)')
ylabel('No of white pixels')
grid on
axis tight
%legend('White Count', 'Regions')

plot(diff(regions))
%Identify the start and end of regions based on the binary image
%regionStarts = [1; v(:)];
regionStarts = [1; find(diff(regions) == 1)];

regionEnds = [find(diff(regions) == 1); length(regions)];
widestRegionIdx = regionEnds-regionStarts;
[~,widestRegionIdx] = max(regionEnds-regionStarts);
upperlimitROI = regionStarts(widestRegionIdx);
lowerlimitROI = regionEnds(widestRegionIdx);
licensePlateROI = licenseplateBW(upperlimitROI:lowerlimitROI,:);
imshow(licensePlateROI)

% 1. Count white pixels per column
whiteCountPerCol = sum(licensePlateROI, 1);

% 2. Adaptive Thresholding
% Instead of a hard number, let's look for columns that are 
% significantly brighter than the average column in this strip.
%colThreshold = mean(whiteCountPerCol) * 0.9; % Adjust the 0.9 to be more/less sensitive
%colRegions = whiteCountPerCol > colThreshold;


    %plateHeight = size(licensePlateROI, 1);
    %widths = regionEnds - regionStarts;
    
    % Find regions that match a plate's aspect ratio
    % validIdx = find(widths > (plateHeight * 2) & widths < (plateHeight * 6));
    
    %leftLimitROI = regionStarts(finalIdx);
    %rightLimitROI = regionEnds(finalIdx);

    % 5. Final Crop
    %tightLicensePlateROI = licensePlateROI(:, leftLimitROI:rightLimitROI);
    %show(tightLicensePlateROI)

    processedImg = licensePlateROI;
end

function out = imread(varargin)
% Redirect the script's hard-coded read to the selected pipeline image.

    selectedImg = india_v7_selected_image('get');
    if ~isempty(selectedImg) && nargin == 1
        out = selectedImg;
        return;
    end

    out = builtin('imread', varargin{:});
end

function value = india_v7_selected_image(action, img)
% Store the selected image for the wrapped experimental script.

    persistent selectedImg

    switch action
        case 'set'
            selectedImg = img;
            value = [];
        case 'get'
            value = selectedImg;
        otherwise
            error('india_v7:invalidAction', 'Unsupported action: %s', action);
    end
end
