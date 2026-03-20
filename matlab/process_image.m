function processedImg = process_image(img)
% This function runs image processing on the given image.

    % Ensure helpers in image_processing/ are on the MATLAB path even when
    % process_image is invoked directly (outside main.m).
    persistent pathSet;
    if isempty(pathSet)
        pipelineDir = fullfile(fileparts(mfilename('fullpath')), 'image_processing');
        if exist(pipelineDir, 'dir')
            addpath(pipelineDir);
            pathSet = true;
        else
            error('process_image:missingPipelineDir', ...
                  'Expected helper folder at %s', pipelineDir);
        end
    end

% Please write new function versions for major iterations of our image processing pipeline
    % processedImg = lynton_v2(img);
    % processedImg = india_v1(img);
    % processedImg = oliver_v2(img);
    % processedImg = oliver_v3(img);
    processedImg = india_v1(img);
end
