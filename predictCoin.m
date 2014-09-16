% Runs prediction on unlabeled data

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% ! Check / setup parameters before run
countryId = 'yjV_AAEB60EAAAEj7Oxucewv';
datasetDir = 'C:/share/dataset-test-all2/countries/'; % dataset root dir
datasetDir = strcat(datasetDir, countryId, '/');

imageDir = strcat(datasetDir, 'ci_images/'); % subdir with unlabeled images
%imageDir = strcat(datasetDir, 'ci_images/out2/img_grayscale/'); % subdir with unlabeled images
tempDir = 'temp/'; % for pooled features used with mini batch

% configs are in separate file to easy share between train.m / test.m
config_coin;
%numClassesL3 = 6; % amount of output lables, classes (e.g. coins)
%pause;

% getting  numOutputClasses for display info 
    csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    
    y = csvdata(:, 2); % second column is coinIdx
    numOutputClasses = max(y);
    

amountConvLayers = size(cnn, 2);

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
                                    
fprintf('\nLayer FC1 -> Layer FC2  %u -> %u \n', cnn{amountConvLayers}.outputWidth * cnn{amountConvLayers}.outputHeight * cnn{amountConvLayers}.outputChannels, numOutputClasses);

% loadinng matrixes
load(strcat(datasetDir, tempDir, 'LFC1_THETA.mat'));
%load(strcat(datasetDir, tempDir, 'LFC2_THETA.mat'));


%% = load countries map (idx, chnID) =======================
coinsAllFile = strcat(datasetDir, 'coin.map.csv')

fileID = fopen(coinsAllFile);
coinsAll = textscan(fileID, '%s %s', 'delimiter',',');
fclose(fileID);
%% ========================

% loading image files
imgDirFullPath = imageDir; % dir with unlabeled images
imgFiles = dir(fullfile(imgDirFullPath, '*.jpg')); % img files
m = length(imgFiles); % number of images

fprintf('Loading %u images for prediction ...\n', m);

%sampleId = cell(length(imgFiles), 1);

coinPredictFileId = fopen(strcat(datasetDir, tempDir, 'coin.predictions.chn.csv'),'w');
% loop over files
for idx = 1:m
    unlabeledImagesX = zeros(imgW*imgH, 1); % unlabeled image (1 image)
    gImg = imread([imgDirFullPath imgFiles(idx).name]);
    imgV = reshape(gImg, 1, imgW*imgH); % unroll       
    unlabeledImagesX(:, 1) = imgV; 
    %sampleId{idx, 1} = imgFiles(idx).name;
    [prediction, mlp_confidence] = netPredictSoftmax(unlabeledImagesX, cnn, Theta3, maxTopPredictions);

    imgFileName = imgFiles(idx).name;
    % print imageID (substring just ID 0TwKb0OMRnkAAAFFeQBqTMon.jpg -> 0TwKb0OMRnkAAAFFeQBqTMon), countryId
%    fprintf( coinPredictFileId, '%s,%s,', imgFileName(1:24), countryId); 
%    fprintf( coinPredictFileId, 'INSERT INTO `BIND_SUGGEST` (`CIID`, `SUGGESTCOUNTRY`, `SUGGESTLIST`) VALUES ("%s", "%s", "1");', imgFileName(1:24), countryId); 
    fprintf( coinPredictFileId, 'UPDATE BIND_SUGGEST '); 
    fprintf( coinPredictFileId, 'SET SUGGESTLIST = "'); 
            
    for maxTopPredictionsIterNum = 1 : maxTopPredictions
        for coinIterNum = 1 : size(coinsAll{1})

            if prediction(maxTopPredictionsIterNum) == str2num(coinsAll{1}{coinIterNum})
%                fprintf('%s -> %u ( %s ) \n', imgFiles(idx).name, prediction(maxTopPredictionsIterNum), coinsAll{2}{coinIterNum}); 
                fprintf(coinPredictFileId, '%s|', coinsAll{2}{coinIterNum}); 
                break
            end
        end % for coinIterNum = 1 : size(coinsAll{1})
    end
    fprintf( coinPredictFileId, '"'); 
    fprintf( coinPredictFileId, ' WHERE CIID = "%s";', imgFileName(1:24)); 
    fprintf( coinPredictFileId, '\n'); % print imageID 
    
 %{   
    for maxTopPredictionsIterNum = 1 : maxTopPredictions
        countryPred = prediction(maxTopPredictionsIterNum)
        for j = 1 : size(coinsAll{1})
            if countryPred == str2num(coinsAll{6}{j})
                % print to console
                fprintf('\n %u from %u: %s -> %u / %s (%1.2f) \n', idx, m, imgFiles(idx).name, prediction(maxTopPredictionsIterNum), coinsAll{1}{j}, mlp_confidence(maxTopPredictionsIterNum));
                % print to file
                fprintf( coinPredictFileId, '%s,%s\n', imgFiles(idx).name, coinsAll{1}{j} ); % print imageID 
                break;
            end
        end
    end
%}
    
    
end
fclose(coinPredictFileId);



%correctCountryIdx = 28;
%correctConfidence = sum(mlp_confidence(find(prediction(:) == correctCountryIdx)'))
%wrongConfidence = sum(mlp_confidence(find(prediction(:) ~= correctCountryIdx)'))

