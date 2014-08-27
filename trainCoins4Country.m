%clear ; close all; clc % cleanup
%%======================================================================
%% Configuration
%  ! Setup and check all parameters before run

%datasetDir = 'C:/Develop/_chn-data/dataset-test/'; % dataset root dir

% configs are in separate file to easy share between train.m / test.m
%config;
trainSetCSVFile = 'coin.tr.shuffled.restricted.csv'; % this file will be generated from 'coin.tr.csv'

% configs are in separate file to easy share between train.m / test.m
config_coin;
%numClassesL3 = 6; % amount of output lables, classes (e.g. coins)

for convLayerIndex = 1 : amountConvLayers
    fprintf(' Parameters for L%u  \n', convLayerIndex + 1);
    cnn{convLayerIndex}
end




%% Initializatoin

% create suffled training set - if doesn't created
if ~exist(strcat(datasetDir, 'coin.tr.shuffled.csv'), 'file')
    fprintf('Generating shuffled training set coin.tr.shuffled.csv from coin.tr.csv \n');
    shuffleTrainingSet(datasetDir, 'coin.tr.csv', 'coin.tr.shuffled.csv');
end

if ~exist(strcat(datasetDir, 'coin.tr.shuffled.restricted.csv'), 'file')
    fprintf('Generating shuffled restricted training set coin.tr.shuffled.restricted.csv from coin.tr.shuffled.csv \n');
 
    csvdata = csvread(strcat(datasetDir, 'coin.tr.shuffled.csv'));  
    csvdata = filterDataset(csvdata, max_class_samples);
    
    % shuffle it again 
    shuffledOrder = randperm(size(csvdata,1))';
    shuffled_csvdata = csvdata(shuffledOrder, :);
    
    dlmwrite(strcat(datasetDir, 'coin.tr.shuffled.restricted.csv'), shuffled_csvdata, 'precision',15);
    clear csvdata;

end

mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist

csvdata = csvread(strcat(datasetDir, trainSetCSVFile));  

% show distribution test samples over output labels
unique_labels = unique(csvdata(:, 2));
hist(csvdata(:, 2), max(unique_labels))

numOutputClasses = max(unique_labels);

% show matrix size transformation between layers
for convLayerIndex = 1 : amountConvLayers
    fprintf('\nL%u -> L%u  (%u X %u X %u) -> (%u X %u X %u) / (%u -> %u) \n', convLayerIndex, convLayerIndex + 1, ...
                                        cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth, cnn{convLayerIndex}.outputHeight, cnn{convLayerIndex}.outputChannels, ...
                                        cnn{convLayerIndex}.inputWidth * cnn{convLayerIndex}.inputHeight * cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth * cnn{convLayerIndex}.outputHeight * cnn{convLayerIndex}.outputChannels);
end
                                    
fprintf('\nLayer FC1 -> Out  %u -> %u \n', cnn{amountConvLayers}.outputWidth * cnn{amountConvLayers}.outputHeight * cnn{amountConvLayers}.outputChannels, numOutputClasses);
%fprintf('\nL3 -> L4  %u -> %u \n', cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels, inputSizeFCL2);

%fprintf('\nLayer FC2 -> Out  %u -> %u \n', inputSizeFCL2, numOutputClasses);


pause;


sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
trainSamplesAmount = size(csvdata, 1); % amount of training examples
batchIterationCount = ceil(trainSamplesAmount / batchSize);



%% Visualize some full size images from training set
% make sure visualy we work on the right dataset

visualAmount = 3^2;
fprintf('Visualize %u full size images ...\n', visualAmount);
[previewX] = loadImageSet(csvdata(1:visualAmount, 1), strcat(datasetDir, imgDir), imgW, imgH);
fullSizeImages = zeros(imgW^2, visualAmount);
for i = 1:visualAmount
    % visualization works for squared matrixes
    % before visualization convert img_h x img_w -> img_w * img_w
    fullSizeImages(:, i) = resizeImage2Square(previewX(:, i), imgW, imgH);
end;

display_network(fullSizeImages);

clear previewX fullSizeImages;

fprintf(' Program is paused. Press ENTER to continue  \n');
pause;

%%======================================================================

%% L2 training (patches extraction, SAE training, convelution & pooling)
fprintf('\nL2 training (patches extraction, SAE training, convelution & pooling) ... (%u X %u X %u) -> (%u X %u X %u) \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputChannels, cnn{1}.outputWidth, cnn{1}.outputHeight, cnn{1}.outputChannels);

%% L2 Patches for auto-encoders training
fprintf('\nL2 - patches extraction for SAE training ...\n')
saeL2PatchesFile = strcat(datasetDir, tempDir, 'L2_PATCHES.mat');
if exist(saeL2PatchesFile, 'file')
    % PATCHES.mat file exists. 
    fprintf('Loading patches for sparse auto-encoder training from %s  \n', saeL2PatchesFile);
    load(saeL2PatchesFile);
else
    % PATCHES.mat File does not exist. do generation
    fprintf('Cant load patches for sparse auto-encoder training from %s  \n', saeL2PatchesFile);
    fprintf('  Do patch geenration \n');
    
    unlabeledImgDirFullPath = strcat(datasetDir, unlabeledImgDir); % dir with unlabeled images
    
    maxFiles4Patches = 10000; % if unlabeled files too many -> get randomly some of them
    if maxFiles4Patches > size(csvdata, 1)
        maxFiles4Patches = size(csvdata, 1)
    end
    fprintf('Loading %u random images for patches ...\n', maxFiles4Patches);

    unlabeledImagesX = zeros(imgW*imgH, maxFiles4Patches); % unlabeled images
    % loop over files and load images into matrix
    for idx = 1:maxFiles4Patches
        randomIdx = randi([1, size(csvdata, 1)]);
        gImg = imread([unlabeledImgDirFullPath strcat(num2str(csvdata(randomIdx, 1)), '.jpg')]);
        imgV = reshape(gImg, 1, imgW*imgH); % unroll       
        unlabeledImagesX(:, idx) = imgV; 
    end
    
    fprintf('Generating %u patches (%u x %u) from images ...\n', cnn{1}.numPatches, cnn{1}.patchSize, cnn{1}.patchSize);
    [patches, meanPatch] = getPatches(unlabeledImagesX, cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.patchSize, cnn{1}.numPatches);


    % remove (clean up some memory)
    clear unlabeledImagesX

    save(saeL2PatchesFile, 'patches', 'meanPatch');
    display_network(patches(:,randi(size(patches,2),200,1)));
    
    %pause;
    fprintf('Patches generation complete ...\n');
end

%%======================================================================

% L2 training (patches extraction, SAE training, convelution & pooling)
convLayer = 2;
cnn{1}.patches = patches;
cnn{1}.meanPatch = meanPatch;

saeOptTheta = trainConvLayer(cnn, convLayer, datasetDir, tempDir, trainSamplesAmount, batchSize, saeOptions, sampleId, strcat(datasetDir, imgDir), imgW, imgH);

% Visualization Sparser Autoencoder Features for L2 to see that the features look good
W = reshape(saeOptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
display_network(W'); % L2


%{

% L3 training (patches extraction, SAE training, convelution & pooling)
convLayer = convLayer + 1;
cnn{2}.patches = [];
cnn{2}.meanPatch = [];
trainConvLayer(cnn, convLayer, datasetDir, tempDir, trainSamplesAmount, batchSize, saeOptions);

% L4 training (patches extraction, SAE training, convelution & pooling)
convLayer = convLayer + 1;
cnn{3}.patches = [];
cnn{3}.meanPatch = [];
trainConvLayer(cnn, convLayer, datasetDir, tempDir, trainSamplesAmount, batchSize, saeOptions);

% L5 training (patches extraction, SAE training, convelution & pooling)
convLayer = convLayer + 1;
cnn{4}.patches = [];
cnn{4}.meanPatch = [];
trainConvLayer(cnn, convLayer, datasetDir, tempDir, trainSamplesAmount, batchSize, saeOptions);
%}


%% L FC1 - L FC2 (MLP) Training
fprintf('\nL FC1 - L FC2 (MLP) Training ... \n')

%-- for error curve (for predictions) --
for convLayerIndex = 1 : amountConvLayers
    load(strcat(datasetDir, tempDir, 'L', num2str(convLayerIndex + 1), '_SAE_FEATURES.mat'));
    cnn{convLayerIndex}.theta = saeOptTheta;
    cnn{convLayerIndex}.meanPatch = meanPatch;
end
%-- for error curve --

mlpInputLayerSize = inputSizeFCL1;
%mlpHiddenLayerSize = inputSizeFCL2;
        
theta3File = strcat(datasetDir, tempDir, 'LFC1_THETA.mat');
if exist(theta3File, 'file')
    % L4_THETA.mat file exists. 
    fprintf('Loading Thetta3 from %s  \n', theta3File);
    load(theta3File);
    initial_Theta3 = Theta3;  
else
    % File does not exist. random initialization
    fprintf('Cant load Thetta4 from %s  \n  Do random initialization for Theta1 \n', theta3File);
%    initial_Theta3 = mlpMatrixLayerInit(mlpInputLayerSize, mlpHiddenLayerSize);
    initial_Theta3 = 0.005 * randn(numOutputClasses * mlpInputLayerSize, 1);
end


fprintf('Theta3: %u x %u \n', size(initial_Theta3, 2), size(initial_Theta3, 1));
%----- end load Thettas -------------------

% Unroll parameters
nn_params = [initial_Theta3(:)];

costs = zeros(numTrainIterFC, 1); % cost func over training iterations
predError = zeros(numTrainIterFC, 3); % prediction accuracy over training iterations

for trainingIter = 1 : numTrainIterFC % loop over training iterations
    fprintf('\nStarting training iteration %u from %u \n', trainingIter, numTrainIterFC);
    % loop over batches (training examples)
    
    iterCost = 0;
    for batchIter = 1 : batchIterationCount

        startPosition = (batchIter - 1) * batchSize + 1;
        endPosition = startPosition + batchSize - 1;
        if endPosition > trainSamplesAmount
            endPosition = trainSamplesAmount;
        end

        fprintf('\n training iteration (%u / %u): batch sub-iteration (%u / %u): start %u end %u from %u training samples \n', trainingIter, numTrainIterFC, batchIter, batchIterationCount, startPosition, endPosition, trainSamplesAmount);
        
        % loads cpFeaturesL2
        load(strcat(datasetDir, tempDir, 'L', num2str(convLayer), '_CP_FEATURES_', num2str(batchIter), '.mat')); % file must exist from previous iterations
        
        % Reshape the pooledFeatures to form an input vector for softmax
        miniX = permute(cpFeatures, [4 3 1 2]); % W x H x Ch x tr_num
        numTrainImages = size(cpFeatures, 2);
        
        miniX = reshape(miniX, inputSizeFCL1, numTrainImages);
        
        miniY = y(startPosition:endPosition, :);
        
%{ 
        [nn_params, cost] = minimize(nn_params,'mlpCost', mlpOptions.maxIter, ...
                                                mlpInputLayerSize, ...
                                                mlpHiddenLayerSize, ...
                                                numOutputClasses, softmaxX, softmaxY, mlpLambda);
        
        [nn_params, cost] = minFunc( @(p) mlpCost(p, ...
                                                mlpInputLayerSize, ...
                                                mlpHiddenLayerSize, ...
                                                numOutputClasses, miniX, miniY, mlpLambda), ...
                                    nn_params, mlpOptions);        
        
%}
        
        [nn_params, cost] = minFunc( @(p) softmaxCost(p, ...
                                                numOutputClasses, ...
                                                mlpInputLayerSize, mlpLambda, ...
                                                miniX, miniY), ...
                                    nn_params, mlpOptions);        
                                
                                   
                          
        iterCost = iterCost + mean(cost);
    end; % for batchIter = 1 : batchIterationCount
    iterCost = iterCost/batchIterationCount;
    costs(trainingIter) = iterCost;
    
    % save thetas - can be used if training cycle interrupted 
    Theta3 = reshape(nn_params, numOutputClasses, mlpInputLayerSize);
    save(theta3File, 'Theta3');
    
    fprintf('\nIteration %4i done - Theta3 saved. Average Cost is %4.4f \n', trainingIter, iterCost);

%-------- debug info ------------    
    funcMinChart = figure(2);
    
    s(1) = subplot(2,1,1); % top subplot
    s(2) = subplot(2,1,2); % bottom subplot    
    
    plot(s(1), costs);
    xlabel(s(1), 'Training iterations');
    ylabel(s(1), 'Cost function');    
%-------- debug info ------------    

%-------Run test-------------------------
    % run test when trainingIter > startTestIteration

maxTopPredictions = 3;

    if plotTrainError == 1 && trainingIter > startTestIteration
        [prediction, acc] = testCoinPrediction(datasetDir, strcat(datasetDir, imgDir), 'coin.tr.shuffled.restricted.csv', cnn, amountConvLayers, Theta3, 100, 1); %maxTestSamples = 100
        fprintf('\n tr: Prediction error for coins : %2.3f%%\n', 100 - acc * 100);
        predError(trainingIter, 1) = 100 - acc * 100;
    end

    if plotValidationError == 1 && trainingIter > startTestIteration
        [prediction, acc] = testCoinPrediction(datasetDir, strcat(datasetDir, imgDir), 'coin.cv.csv', cnn, amountConvLayers, Theta3, maxTestSamples, maxTopPredictions);
        fprintf('\n cv: Prediction error for coins : %2.3f%%\n', 100 - acc * 100);
        predError(trainingIter, 2) = 100 - acc * 100;
    end

    if plotTestError == 1 && trainingIter > startTestIteration
        [prediction, acc] = testCoinPrediction(datasetDir, strcat(datasetDir, imgDir), 'coin.tst.csv', cnn, amountConvLayers, Theta3, maxTestSamples, maxTopPredictions);
        fprintf('\n tst: Prediction error for coins : %2.3f%%\n', 100 - acc * 100);
        predError(trainingIter, 3) = 100 - acc * 100;
    end

    plot(s(2), 1 : numTrainIterFC, predError(:, 1), 1 : numTrainIterFC, predError(:, 2), 1 : numTrainIterFC, predError(:, 3) );
    ylabel(s(2), 'Prediction Error %');
    legend('training','validation', 'test');
    drawnow;    

%--------------------------------
    chartFile = strcat(datasetDir, tempDir, 'countries-minimization.jpg');
    saveas(funcMinChart, chartFile, 'jpg');    

end; % for trainingIter = 1 : trainingIterationCount % loop over training iterations

fprintf('Training complete. \n');
