clear ; close all; clc % cleanup
%%======================================================================
%% Configuration
%  ! Setup and check all parameters before run

datasetDirRoot = 'C:/share/dataset-test-all2/'; % dataset root dir
trainSetCSVFile = 'coin.tr.shuffled.csv'; % this file will be generated from 'coin.tr.csv'

unlabeledImgDir = 'img_grayscale/'; % sub directory with images for auto-encoder training (unlabeled/for unsupervised feature extraction)
imgDir = 'img_grayscale/'; % sub directory with images
tempDir = 'temp/'; % for pooled features used with mini batch


countriesDirStr = strcat(datasetDirRoot, 'countries/'); % dir with unlabeled images
countriesDir = dir(fullfile(countriesDirStr)); % img files


%% test with Mexico
    %countriesDir(idx).name
%    mexicoDir = 'dsJ_AAEBV2EAAAEjlOpucewv';
    datasetDir = strcat(datasetDirRoot, 'countries/', 'Z5N_AAEBUIsAAAEjidducewv', '/');

    csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    
    y = csvdata(:, 2); % second column is coinIdx
    numClassesL3 = max(y);

    fprintf('Start raining for %s  \n', datasetDir);
    trainCoins4Country;
    fprintf('Training complete. ');
%%

%{
% loop over files and load images into matrix
% start from 3 (1 is . 2 is ..)
for idx = 3:length(countriesDir)
    countriesDir(idx).name
    datasetDir = strcat(datasetDirRoot, 'countries/', countriesDir(idx).name, '/');

    csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    
    y = csvdata(:, 2); % second column is coinIdx
    numClassesL3 = max(y);

    fprintf('Start raining for %s (%u from %u) \n', datasetDir, idx, length(countriesDir));
    trainCoins4Country;
    fprintf('Training complete. for %u from %u \n', idx, length(countriesDir));

end
%}

%%======================================================================
fprintf('Training complete. \n');
