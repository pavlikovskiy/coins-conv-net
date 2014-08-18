% Runs prediction on unlabeled data

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% ! Check / setup parameters before run

datasetDir = 'C:/Develop/_chn-data/dataset-test/'; % dataset root dir
imageDir = strcat(datasetDir, 'ci_images/'); % subdir with unlabeled images
tempDir = 'temp/'; % for pooled features used with mini batch

% configs are in separate file to easy share 
% between train.m / test.m / predict.m
config;

fprintf(' Parameters for L2  \n');
cnn{1}

% show matrix size transformation between layers
fprintf('\nL1 -> L2  (%u X %u X %u) -> (%u X %u X %u) / (%u -> %u) \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputChannels, cnn{1}.outputWidth, cnn{1}.outputHeight, cnn{1}.outputChannels, ...
                                        cnn{1}.inputWidth * cnn{1}.inputHeight * cnn{1}.inputChannels, cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels);                                    
%fprintf('\nL2 -> L3 %u -> %u \n', cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels, numClassesL3);

maxTopPredictions = 3;

mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist
addpath ../libs/         % load libs

%% ========================
% loadinng matrixes
fprintf('\nLoading L2 features (sae2OptTheta, meanPatchL2) from %s  \n', strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));
load(strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));
fprintf('\nLoading L3 softmaxTheta from %s  \n', strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat'));
load(strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat'));

W = reshape(sae2OptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
fprintf('sae2OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatchL2, 2), size(meanPatchL2, 1));

cnn{1}.theta = sae2OptTheta;
cnn{1}.meanPatch = meanPatchL2;

fprintf('softmaxTheta: %u x %u \n', size(softmaxTheta, 1), size(softmaxTheta, 2));
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


%% = load countries map (idx, chnID) =======================
countriesAllFile = strcat(datasetDir, 'country.all.csv')

fileID = fopen(countriesAllFile);
countriesAll = textscan(fileID, '%s %s %s %s %s %s', 'delimiter',',');
fclose(fileID);
%% ========================
% run prediction for countries
[countryPredict] = netPredict(unlabeledImagesX, cnn, softmaxTheta, convolutionsStepSize, 1)

%prediction = [sampleId, pred, zeros(numTestImages, maxTopPredictions)]

coinPredictFileId = fopen(strcat(datasetDir, tempDir, 'coin.predictions.chn.csv'),'w');

for imageIterNum=1:m  
      
    countryPred = countryPredict(imageIterNum);
      
    for j = 1 : size(countriesAll{1})
        % do prediction for country (1 photo per iteration only)
        if countryPred == str2num(countriesAll{6}{j})
            fprintf('%s -> %u ( %s ) \n', imgFiles(imageIterNum).name, countryPred, countriesAll{1}{j}); 
            fprintf( coinPredictFileId, '%s,%s,', imgFiles(imageIterNum).name, countriesAll{1}{j} ); % print imageID 
            
            countryRootDir = strcat(datasetDir, 'countries/', countriesAll{1}{j}, '/')
            countrySoftmaxtThetaFile = strcat(countryRootDir, tempDir, 'L3_SOFTMAX_THETA.mat');

            if exist(countrySoftmaxtThetaFile, 'file')
                fprintf('\nLoading L3 softmaxTheta from %s  \n', countrySoftmaxtThetaFile);
                load(countrySoftmaxtThetaFile);

                [coinPred] = netPredict(unlabeledImagesX(:, imageIterNum), cnn, softmaxTheta, convolutionsStepSize, maxTopPredictions)
                
                coinsFile = strcat(countryRootDir, 'coin.all.csv')
                coinsFileFileID = fopen(coinsFile);
                coinsAll = textscan(coinsFileFileID, '%s %s %s %s %s %s', 'delimiter',',');
                fclose(coinsFileFileID);
                
                for maxTopPredictionsIterNum = 1 : maxTopPredictions
                    for coinIterNum = 1 : size(coinsAll{1})

                        if coinPred(maxTopPredictionsIterNum) == str2num(coinsAll{3}{coinIterNum})
                            fprintf('%s -> %u ( %s ) -> %u ( %s ) \n', imgFiles(imageIterNum).name, countryPred, countriesAll{1}{j}, coinPred(maxTopPredictionsIterNum), coinsAll{2}{coinIterNum}); 
                            fprintf(coinPredictFileId, '%s|', coinsAll{2}{coinIterNum}); 
                            break
                        end
                    end % for coinIterNum = 1 : size(coinsAll{1})
                end
                    
            end % if exist(countrySoftmaxtThetaFile, 'file')
            
            fprintf(coinPredictFileId, '\n'); 
            break
        end
    end % for j = 1 : size(countriesAll{1})
      
end % for imageIterNum=1:m
fclose(coinPredictFileId);


%% ========================
%{

% save prediction to file
fid = fopen(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'),'w');
for ii=1:m,  %% 1 -> 47165 rows
    fprintf(fid, '%s', imgFiles(ii).name);
    for i=1:size(pred, 2), %% 1 -> 26 columns
%        fprintf(fid, ',%u', data{ii, i}(1));
        fprintf(fid, ',%u', pred(ii, i));
    end
    fprintf(fid,'\n');
end
fclose(fid);
%}