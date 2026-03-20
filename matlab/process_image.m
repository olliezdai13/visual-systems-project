function processedImg = process_image(img, pipelineName)
% This function runs image processing on the given image.

    pipelineDir = fullfile(fileparts(mfilename('fullpath')), 'image_processing');

    % Ensure helpers in image_processing/ are on the MATLAB path even when
    % process_image is invoked directly (outside main.m).
    persistent pathSet;
    if isempty(pathSet)
        if exist(pipelineDir, 'dir')
            addpath(pipelineDir);
            pathSet = true;
        else
            error('process_image:missingPipelineDir', ...
                  'Expected helper folder at %s', pipelineDir);
        end
    end

    if nargin < 2 || isempty(pipelineName)
        pipelineName = 'oliver_v4';
    end

    if ~(ischar(pipelineName) || (isstring(pipelineName) && isscalar(pipelineName)))
        error('process_image:invalidPipelineName', ...
            'Pipeline name must be a character vector or string scalar.');
    end

    pipelineName = char(pipelineName);
    pipelinePath = fullfile(pipelineDir, [pipelineName '.m']);
    if ~exist(pipelinePath, 'file')
        error('process_image:missingPipeline', ...
            'Pipeline "%s" was not found at %s', pipelineName, pipelinePath);
    end

    pipelineFn = str2func(pipelineName);
    processedImg = pipelineFn(img);
end
