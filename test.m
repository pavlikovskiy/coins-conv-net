% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

datasetDirRoot = 'C:/Develop/_chn-data/dataset-test/'; % dataset root dir

imageDir = strcat(datasetDirRoot, 'img_grayscale/');
tempDir = 'temp/'; % for prediction export

maxTestSamples = 200; % if test set is large - create subset 
maxTopPredictions = 3;

% configs are in separate file to easy share between train.m / test.m
config;

fprintf(' Parameters for L2  \n');
cnn{1}

% loadinng matrixes
fprintf('\nLoading L2 features (sae2OptTheta, meanPatchL2) from %s  \n', strcat(datasetDirRoot, tempDir, 'L2_SAE_FEATURES.mat'));
load(strcat(datasetDirRoot, tempDir, 'L2_SAE_FEATURES.mat'));
%fprintf('\nLoading L3 softmaxTheta from %s  \n', strcat(datasetDirRoot, tempDir, 'L3_SOFTMAX_THETA.mat'));
%load(strcat(datasetDirRoot, tempDir, 'L3_SOFTMAX_THETA.mat'));
%numClassesL3 = size(softmaxTheta, 1)

theta3File = strcat(datasetDirRoot, tempDir, 'L3_THETA.mat');
fprintf('\nLoading L3 Theta3 from %s  \n', theta3File);
load(theta3File);

Theta3File = strcat(datasetDirRoot, tempDir, 'L4_THETA.mat');
fprintf('\nLoading L4 Theta3 from %s  \n', Theta3File);
load(Theta3File);
fprintf('Theta3: %u x %u \n', size(Theta3, 1), size(Theta3, 2));
fprintf('Theta4: %u x %u \n', size(Theta4, 1), size(Theta4, 2));


% show matrix size transformation between layers
fprintf('\nL1 -> L2  (%u X %u X %u) -> (%u X %u X %u) / (%u -> %u) \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputChannels, cnn{1}.outputWidth, cnn{1}.outputHeight, cnn{1}.outputChannels, ...
                                        cnn{1}.inputWidth * cnn{1}.inputHeight * cnn{1}.inputChannels, cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels);
                                    
%fprintf('\nL2 -> L3   %u -> %u \n', cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels, numClassesL3);


fprintf('\nL2 -> L3  %u -> %u \n', cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels, inputSizeL4);

fprintf('\nL3 -> L4  %u -> %u \n', inputSizeL4, numOutputClasses);


%% ========================

W = reshape(sae2OptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
fprintf('sae2OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatchL2, 2), size(meanPatchL2, 1));

cnn{1}.theta = sae2OptTheta;
cnn{1}.meanPatch = meanPatchL2;

%fprintf('softmaxTheta: %u x %u \n', size(softmaxTheta, 1), size(softmaxTheta, 2));

%csvdata = csvread(strcat(datasetDir, 'country.all.csv'))
countriesAllFile = strcat(datasetDirRoot, 'country.all.csv')

fileID = fopen(countriesAllFile);
countriesAll = textscan(fileID, '%s %s %s %s %s %s', 'delimiter',',');
fclose(fileID);
%celldisp(countriesAll); % N(6) x M


% prediction test on cross validation dataset
%prediction = testPrediction(imageDir, strcat(datasetDir, 'country.cv.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
%dlmwrite(strcat(datasetDir, tempDir, 'country.cv_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on test dataset
%prediction = testPrediction(imageDir, strcat(datasetDir, 'country.tst.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
%dlmwrite(strcat(datasetDir, tempDir, 'country.tst_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on training dataset 
[prediction, acc] = testCountryPrediction(imageDir, strcat(datasetDirRoot, 'country.tr.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples);
fprintf('\nAccuracy for countries (not for coins inside): %2.3f%%\n', acc * 100);

[prediction, acc] = testCountryPrediction(imageDir, strcat(datasetDirRoot, 'country.cv.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples);
fprintf('\nAccuracy for countries (not for coins inside): %2.3f%%\n', acc * 100);

[prediction, acc] = testCountryPrediction(imageDir, strcat(datasetDirRoot, 'country.tst.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples);
fprintf('\nAccuracy for countries (not for coins inside): %2.3f%%\n', acc * 100);

pause;


%{

countriesDirStr = strcat(datasetDirRoot, 'countries/'); % dir with unlabeled images
countriesDir = dir(fullfile(countriesDirStr)); % img files
% loop over files and load images into matrix
% start from 3 (1 is . 2 is ..)
for idx = 3:length(countriesDir)
    datasetDir = strcat(datasetDirRoot, 'countries/', countriesDir(idx).name, '/');

    fprintf('\nLoading L3 softmaxTheta from %s  \n', strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat'));
    load(strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat'));
    
    fprintf('Start raining for %s (%u from %u) \n', datasetDir, idx, length(countriesDir));
    prediction = testCoinPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
    
    
%    trainCoins4Country;
%    fprintf('Training complete. for %u from %u \n', idx, length(countriesDir));

end
%} 
%

%prediction = testPrediction(datasetDir, imageDir, countriesAll, strcat(datasetDir, 'country.tr.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
%dlmwrite(strcat(datasetDir, tempDir, 'country.tr_predict.csv'), prediction, 'precision',15); % export prediction


