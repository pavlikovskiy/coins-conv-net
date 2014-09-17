% Runs prediction on unlabeled data

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% ! Check / setup parameters before run

datasetDir = 'C:/share/dataset-test-all2/'; % dataset root dir
imageDir = strcat(datasetDir, 'ci_images/'); % subdir with unlabeled images
tempDir = 'temp/'; % for pooled features used with mini batch

% configs are in separate file to easy share 
% between train.m / test.m / predict.m
config_coin;



maxTopPredictions = 3;
%% = load CIID - SUGGESTED_COUNTRY map (chnCiid, chnCountryId) =======================
ciidCountryMapFile = strcat(datasetDir, 'ci_country_export.csv')

fileID = fopen(ciidCountryMapFile);
ciidCountriesMap = textscan(fileID, '%s %s', 'delimiter',',');
fclose(fileID);
%% ========================
% loading image files
imgDirFullPath = imageDir; % dir with unlabeled images
imgFiles = dir(fullfile(imgDirFullPath, '*.jpg')); % img files
m = length(imgFiles); % number of images

fprintf('Loading %u images for prediction ...\n', m);

coinPredictFileId = fopen(strcat(datasetDir, tempDir, 'coin.predictions.chn.', datestr(now,'yyyymmdd_HHMMSS'), '.csv'),'w');

% loop over files and load images into matrix
for idx = 1:m
    imgFileName = imgFiles(idx).name; 
    ciid = imgFileName(1:24);    
    countryId = predictCountryFromFile(ciid, ciidCountriesMap);
    
    fprintf('%u from %u. Image to country: %s ...  %s -> %s \n', idx, m, imgFileName, ciid, countryId);
    
    countryRootDir = strcat(datasetDir, 'countries/', countryId, '/');
    countrySoftmaxtThetaFile = strcat(countryRootDir, tempDir, 'LFC1_THETA.mat');
    coinsAllFile = strcat(countryRootDir, 'coin.map.csv');

    if exist(countrySoftmaxtThetaFile, 'file') && exist(coinsAllFile, 'file')
%        addpath(countryRootDir);
%        config_coin;
        %% loadinng data & matrixes
        fprintf('   LOADING COUNTRY FILES  %s  AND  %s \n', countrySoftmaxtThetaFile, coinsAllFile);
        load(countrySoftmaxtThetaFile);

        % loadinng convolution matrixes
        amountConvLayers = size(cnn, 2);
        for convLayerIndex = 1 : amountConvLayers
            load(strcat(countryRootDir, tempDir, 'L', num2str(convLayerIndex + 1), '_SAE_FEATURES.mat'));
            cnn{convLayerIndex}.theta = saeOptTheta;
            cnn{convLayerIndex}.meanPatch = meanPatch;
        end        
        
        % load countries map (idx, chnID) 
        fileID = fopen(coinsAllFile);
        coinsAll = textscan(fileID, '%s %s', 'delimiter',',');
        fclose(fileID);
        
        %% predicting        
        gImg = imread([imgDirFullPath imgFiles(idx).name]);
        imgV = reshape(gImg, 1, imgW*imgH); % unroll       
        [prediction, mlp_confidence] = netPredictSoftmax(imgV', cnn, Theta3, maxTopPredictions);
        %% saving to file (generated SQL)
        fprintf( coinPredictFileId, 'UPDATE BIND_SUGGEST '); 
        fprintf( coinPredictFileId, 'SET SUGGESTLIST = "'); 
        
        for maxTopPredictionsIterNum = 1 : maxTopPredictions
            for coinIterNum = 1 : size(coinsAll{1})
                if prediction(maxTopPredictionsIterNum) == str2num(coinsAll{1}{coinIterNum})
                    % fprintf('%s -> %u ( %s ) \n', imgFiles(idx).name, prediction(maxTopPredictionsIterNum), coinsAll{2}{coinIterNum}); 
                    fprintf(coinPredictFileId, '%s|', coinsAll{2}{coinIterNum}); 
                    break
                end
            end % for coinIterNum = 1 : size(coinsAll{1})
        end
        fprintf( coinPredictFileId, '"'); 
        fprintf( coinPredictFileId, ' WHERE CIID = "%s";', imgFileName(1:24)); 
        fprintf( coinPredictFileId, '\n'); % print imageID 
    
    else
        fprintf('   NO COUNTRY FILES CHECK %s  OR  %s \n', countrySoftmaxtThetaFile, coinsAllFile);
    end

%% ========================    
    
end

fclose(coinPredictFileId);


