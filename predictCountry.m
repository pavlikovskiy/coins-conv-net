% Runs prediction on unlabeled data

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% ! Check / setup parameters before run

datasetDir = 'C:/share/dataset-test-all2/'; % dataset root dir
%imageDir = strcat(datasetDir, 'ci_images/out/'); % subdir with unlabeled images
imageDir = strcat(datasetDir, 'ci_images/out2/img_grayscale/'); % subdir with unlabeled images
tempDir = 'temp/'; % for pooled features used with mini batch

% configs are in separate file to easy share between train.m / test.m
configMaster;
%numClassesL3 = 6; % amount of output lables, classes (e.g. coins)

amountConvLayers = 2;

for convLayerIndex = 1 : amountConvLayers
    fprintf(' Parameters for L%u  \n', convLayerIndex + 1);
    cnn{convLayerIndex}
end


% show matrix size transformation between layers
for convLayerIndex = 1 : amountConvLayers
    fprintf('\nL%u -> L%u  (%u X %u X %u) -> (%u X %u X %u) / (%u -> %u) \n', convLayerIndex, convLayerIndex + 1, ...
                                        cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth, cnn{convLayerIndex}.outputHeight, cnn{convLayerIndex}.outputChannels, ...
                                        cnn{convLayerIndex}.inputWidth * cnn{convLayerIndex}.inputHeight * cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth * cnn{convLayerIndex}.outputHeight * cnn{convLayerIndex}.outputChannels);

    % loadinng matrixes
    load(strcat(datasetDir, tempDir, 'L', num2str(convLayerIndex + 1), '_SAE_FEATURES.mat'));
    cnn{convLayerIndex}.theta = saeOptTheta;
    cnn{convLayerIndex}.meanPatch = meanPatch;
                                    
end
                                    
fprintf('\nLayer FC1 -> Layer FC2  %u -> %u \n', cnn{2}.outputWidth * cnn{2}.outputHeight * cnn{2}.outputChannels, inputSizeFCL2);
fprintf('\nLayer FC2 -> Out  %u -> %u \n', inputSizeFCL2, numOutputClasses);

% loadinng matrixes
load(strcat(datasetDir, tempDir, 'LFC1_THETA.mat'));
load(strcat(datasetDir, tempDir, 'LFC2_THETA.mat'));


%% ========================

% loading image files
imgDirFullPath = imageDir; % dir with unlabeled images
imgFiles = dir(fullfile(imgDirFullPath, '*.jpg')); % img files
m = length(imgFiles); % number of images

fprintf('Loading %u images for prediction ...\n', m);
unlabeledImagesX = zeros(imgW*imgH, m); % unlabeled images

sampleId = cell(length(imgFiles), 1);

% loop over files and load images into matrix
for idx = 1:m
    gImg = imread([imgDirFullPath imgFiles(idx).name]);
    imgV = reshape(gImg, 1, imgW*imgH); % unroll       
    unlabeledImagesX(:, idx) = imgV; 
    sampleId{idx, 1} = imgFiles(idx).name;
end


[prediction, mlp_confidence] = netPredict(unlabeledImagesX, cnn, Theta3, Theta4, 1)

for idx = 1:m
    fprintf('%s -> %u (%1.2f) \n', imgFiles(idx).name, prediction(idx), mlp_confidence(idx));
end

correctCountryIdx = 28;
correctConfidence = sum(mlp_confidence(find(prediction(:) == correctCountryIdx)'))
wrongConfidence = sum(mlp_confidence(find(prediction(:) ~= correctCountryIdx)'))

