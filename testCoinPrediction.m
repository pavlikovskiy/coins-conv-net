function [pred, acc] = testCoinPrediction(datasetDir, imageDir, datasetFileName, cnn, Theta3, maxTestSamples, maxTopPredictions)

%TESTPREDICTION Implements the test on specific dataset
%
%   imageDir - dir with dataset images
%   datasetFile - CSV file with dataset
%   cnn
%   softmaxTheta
%   maxTestSamples
%   maxTopPredictions

tempDir = 'temp/'; % for prediction export
batchSize = 20;
%maxTopPredictions = 3;

%imageDir = strcat(datasetDir, imgDir); 
datasetFile = strcat(datasetDir, datasetFileName); 

fprintf('\nRunning prediction test ...  ');
fprintf('\n    dataset %s ', datasetFile);
fprintf('\n    image dir %s ', imageDir);


datasetCacheFile = strcat(datasetFile, '.cache.csv');
if exist(datasetCacheFile, 'file')
    fprintf('\nLoad from cache %s \n', datasetCacheFile);
    csvdata = csvread(datasetCacheFile);
else
    fprintf('\nCache not found Creating %s \n', datasetCacheFile);
    csvdata = csvread(datasetFile);

    m = size(csvdata,1); % amount of test samples

    % if test dataset is huge -> shuffle and get random maxTestSamples records
    if m > maxTestSamples
        shuffledOrder = randperm(m)';
        shuffled_csvdata = csvdata(shuffledOrder, :);
        csvdata = shuffled_csvdata(1:maxTestSamples, :);    
    end
    dlmwrite(datasetCacheFile, csvdata, 'precision',15);
end

sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
softmaxY = csvdata(:, 2); % second column is coinIdx
numTestImages = size(csvdata, 1); % amount of training examples
fprintf('\n    %u items in dataset ', numTestImages);


%numTestImages = size(csvdata, 1); % amount of training examples
batchIterationCount = ceil(numTestImages / batchSize);

pred = [];
for batchIter = 1 : batchIterationCount

    startPosition = (batchIter - 1) * batchSize + 1;
    endPosition = startPosition + batchSize - 1;
    if endPosition > numTestImages
        endPosition = numTestImages;
    end

    fprintf('\nBatch sub-iteration (%u / %u): start %u end %u from %u test samples \n', batchIter, batchIterationCount, startPosition, endPosition, numTestImages);

    convPooledFeaturesCacheFile = strcat(datasetDir, tempDir, datasetFileName, '_', num2str(batchIter), '.cache.mat');

    if exist(convPooledFeaturesCacheFile, 'file')
        % load X
        fprintf('\n  Load from cache %s \n', convPooledFeaturesCacheFile);
        load(convPooledFeaturesCacheFile);
    else
        fprintf('\n  Cache not found Creating %s \n', convPooledFeaturesCacheFile);
        [X] = loadImageSet(sampleId(startPosition:endPosition), imageDir, cnn{1}.inputWidth, cnn{1}.inputHeight); % images
        for convLayerIndex = 1 : 2
            fprintf('\n   L%u  (%u X %u X %u) -> (%u X %u X %u) \n', convLayerIndex + 1, cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth, cnn{convLayerIndex}.outputHeight, cnn{convLayerIndex}.outputChannels);
            cpFeatures = convolveAndPool(X, cnn{convLayerIndex}.theta, cnn{convLayerIndex}.features, ...
                            cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputChannels, ...
                            cnn{convLayerIndex}.patchSize, cnn{convLayerIndex}.meanPatch, cnn{convLayerIndex}.poolSize, cnn{convLayerIndex}.convolutionsStepSize);
            convOut = permute(cpFeatures, [4 3 1 2]);
            X = reshape(convOut, cnn{convLayerIndex}.outputSize, endPosition - startPosition + 1);
        end
        save(convPooledFeaturesCacheFile, 'X');

    end
    [predThis, confidence] = softmaxPredict(Theta3, X, maxTopPredictions);
    pred = [pred; predThis];
    
end



% -------------- debug info ------------
 %a = [sampleId, softmaxY(:), pred, (pred(:, 1) == softmaxY(:))]
 
 
 %confidence'
 
% -------------- debug info ------------


acc = zeros(size(softmaxY));
% accumulate predictions over maxTopPredictions
for i = 1:maxTopPredictions
    acc = acc + (pred(:, i) == softmaxY(:));
end

%acc = (pred(:, 1) == softmaxY(:)) + (pred(:, 2) == softmaxY(:)) + (pred(:, 3) == softmaxY(:));

acc = sum(acc) / size(acc, 1);
% fprintf('\nAccuracy for countries (not for coins inside): %2.3f%%\n', acc * 100);


%% -----------------------------------------------------
% show correct prediction value and amount of correct samples
%{
i = double(pred(:) == softmaxY(:));
yind = find(i == 1); % index where prediction correct

vals = softmaxY(yind)'; % value for correct prediction

uniqval = unique(vals); % unique values for correct prediction

%uniqvalamnt = size(uniqval, 1);

% show correct prediction value and amount of correct samples
for i = 1:size(uniqval, 2)
    v = uniqval(i);
    v = repmat(v, size(vals, 1), 1);
    amnt = sum(double(vals == v));
    fprintf('\n  %u - %u ', uniqval(i), amnt);
end
%}
%-----------------------------------------------------

end