function [saeOptTheta] = trainConvLayer(cnn, layer, datasetDir, tempDir, trainSamplesAmount, batchSize, saeOptions, ...
                                varargin) % this parameters (sampleId, imgDir, imgW, imgH) requied for first layer only. For other layers they should be omited

% e.g. layer start with 
% layer = 3
% cnnLayerIdx = 2
% previous cnnLayerIdx = 1
if (0 < size(varargin))
    sampleId = varargin{1}; 
    imgDir = varargin{2}; 
    imgW = varargin{3}; 
    imgH = varargin{4};
end


cnnLayerIdx = layer - 1;

batchIterationCount = ceil(trainSamplesAmount / batchSize);

%% L3 training (patches extraction, SAE training, convelution & pooling)
fprintf('\nL%u training (patches extraction, SAE training, convolution & pooling) ... (%u X %u X %u) -> (%u X %u X %u) \n', layer, cnn{cnnLayerIdx}.inputWidth, cnn{cnnLayerIdx}.inputHeight, cnn{cnnLayerIdx}.inputChannels, cnn{cnnLayerIdx}.outputWidth, cnn{cnnLayerIdx}.outputHeight, cnn{cnnLayerIdx}.outputChannels);


%% L3 Patches for auto-encoders training
patches = cnn{cnnLayerIdx}.patches;
meanPatch = cnn{cnnLayerIdx}.meanPatch;

if isempty(patches) || isempty(meanPatch)
    fprintf('\nL%u - patches extraction for SAE training ...\n', layer)
    saeL3PatchesFile = strcat(datasetDir, tempDir, 'L', num2str(layer), '_PATCHES.mat');

    if exist(saeL3PatchesFile, 'file')
        % PATCHES.mat file exists. 
        fprintf('Loading patches for sparse auto-encoder training from %s  \n', saeL3PatchesFile);
        load(saeL3PatchesFile);
    else
        % PATCHES.mat File does not exist. do generation
        fprintf('Cant load patches for sparse auto-encoder training from %s  \n', saeL3PatchesFile);
        fprintf('  Do patch geenration \n');

        numPatches = floor(cnn{cnnLayerIdx}.numPatches / batchIterationCount); % get some patches from every batch iteration

        for batchIter = 1 : batchIterationCount
            %load cpFeaturesL2
            cpFeaturesL2File = strcat(datasetDir, tempDir, 'L', num2str(cnnLayerIdx), '_CP_FEATURES_', num2str(batchIter), '.mat');
            load(cpFeaturesL2File);
            % Reshape cpFeaturesL2 
            % numTrainImages = size(cpFeatures, 2);
            outL2 = permute(cpFeatures, [4 3 1 2]); % W x H x Ch x tr_num
            [patchesThis, meanPatchThis] = getPatches2(outL2, cnn{cnnLayerIdx}.patchSize, numPatches);

            if 1 == batchIter
                patches = patchesThis;
                meanPatch = meanPatchThis;
            else 
                patches = [patches patchesThis];
                meanPatch = [meanPatch meanPatchThis];
            end

        end; % for batchIter = 1 : batchIterationCount
        meanPatch = mean(meanPatch, 2);

        save(saeL3PatchesFile, 'patches', 'meanPatch');
    %    display_network(patches(:,randi(size(patches,2),200,1)));
        fprintf('Patches generation complete ...\n');
    end
end


%%======================================================================
%% L3 SAE training
fprintf('\nL%u - SAE training ...\n', layer);

saeFeaturesFile = strcat(datasetDir, tempDir, 'L', num2str(layer), '_SAE_FEATURES.mat');

if exist(saeFeaturesFile, 'file')
    % SAE1_FEATURES.mat file exists. 
    fprintf('Loading sparse auto-encoder features from %s  \n', saeFeaturesFile);    
    load(saeFeaturesFile);
else
    % SAE1_FEATURES.mat File does not exist. do generation
    fprintf('Cant load sparse auto-encoder features from %s  \n', saeFeaturesFile);
    fprintf('  Do features extraction \n');
    
    %  Obtain random parameters theta
    theta = saeMatrixInit(cnn{cnnLayerIdx}.features, cnn{cnnLayerIdx}.inputVisibleSize);

    [saeOptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                       cnn{cnnLayerIdx}.inputVisibleSize, cnn{cnnLayerIdx}.features, ...
                                       cnn{cnnLayerIdx}.saeLambda, cnn{cnnLayerIdx}.saeSparsityParam, ...
                                       cnn{cnnLayerIdx}.saeBeta, patches), ...
                                  theta, saeOptions);

    save(saeFeaturesFile, 'saeOptTheta', 'meanPatch');
end
    
% Visualization Sparser Autoencoder Features to see that the features look good
% W = reshape(sae3OptTheta(1 : cnn{cnnLayerIdx}.inputVisibleSize * cnn{cnnLayerIdx}.features), cnn{cnnLayerIdx}.features, cnn{cnnLayerIdx}.inputVisibleSize);

%display_network(W'); % L3

%pause;
%%======================================================================
%% L3 - Convolution & pooling
fprintf('\n L%u - Feedforward with SAE, Convolve & pool ...\n', layer)

for batchIter = 1 : batchIterationCount

    startPosition = (batchIter - 1) * batchSize + 1;
    endPosition = startPosition + batchSize - 1;
    if endPosition > trainSamplesAmount
        endPosition = trainSamplesAmount;
    end

    fprintf('\n Convolved and pooled (CP) L%u feature extraction: batch sub-iteration (%u / %u): start %u end %u from %u training samples \n', layer, batchIter, batchIterationCount, startPosition, endPosition, trainSamplesAmount);        
    %%------ cache convolved and pooled features - will be used in next layers ----------        
    pooledFeaturesTempFile = strcat(datasetDir, tempDir, 'L', num2str(layer), '_CP_FEATURES_', num2str(batchIter), '.mat');
    if ~exist(pooledFeaturesTempFile, 'file')
        % File does not exist - do convolution and pooling
        fprintf('\nNo file with pooled features for iteration %u. Do convolution and pooling ... \n', batchIter);
        
        if 0 == size(varargin) % optional parameters for first layer only
            % this is 2nd (or higher) convolution layer - get data from
            % stored features for previous layer
            % load cpFeatures for input (layer - 1)
            cpFeaturesL2File = strcat(datasetDir, tempDir, 'L', num2str(layer - 1), '_CP_FEATURES_', num2str(batchIter), '.mat');
            load(cpFeaturesL2File);
            % Reshape cpFeaturesL2 
            numTrainImages = size(cpFeatures, 2);
            outL2 = permute(cpFeatures, [4 3 1 2]); % W x H x Ch x tr_num

            X = reshape(outL2, cnn{cnnLayerIdx - 1}.outputSize, numTrainImages);
        else
            % this is first convolution layer - get data from images
            X = loadImageSet(sampleId(startPosition:endPosition), imgDir, imgW, imgH); % strcat(datasetDir, imgDir)
        end

        
        % feedforward using sae2OptTheta, convolve and pool
        % calculate cpFeatures for output (layer)
        cpFeatures = convolveAndPool(X, saeOptTheta, cnn{cnnLayerIdx}.features, ...
                                        cnn{cnnLayerIdx}.inputHeight, cnn{cnnLayerIdx}.inputWidth, cnn{cnnLayerIdx}.inputChannels, ...
                                        cnn{cnnLayerIdx}.patchSize, meanPatch, cnn{cnnLayerIdx}.poolSize, ...
                                        cnn{cnnLayerIdx}.convolutionsStepSize);
        save(pooledFeaturesTempFile, 'cpFeatures');
    end
    %%----------------------------------------------------------------------------------        
end; % for batchIter = 1 : batchIterationCount


clear patches meanPatch cpFeatures

end