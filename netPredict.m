function [prediction, mlp_confidence] = netPredict(X, cnn, Theta3, Theta4, maxTopPredictions)

%NETPREDICT Implements the test on specific dataset
%
%   X - images  N x M
%   cnn
%   softmaxTheta
%   convolutionsStepSize
%   maxTopPredictions
%
%   prediction - output prediction with size M x maxTopPredictions


numTestImages = size(X, 2); % amount of training examples

% convLayerIndex = 1;

for convLayerIndex = 1 : 2
    fprintf('\nL%u  (%u X %u X %u) -> (%u X %u X %u) \n', convLayerIndex + 1, cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputChannels, cnn{convLayerIndex}.outputWidth, cnn{convLayerIndex}.outputHeight, cnn{convLayerIndex}.outputChannels);
    cpFeatures = convolveAndPool(X, cnn{convLayerIndex}.theta, cnn{convLayerIndex}.features, ...
                    cnn{convLayerIndex}.inputHeight, cnn{convLayerIndex}.inputWidth, cnn{convLayerIndex}.inputChannels, ...
                    cnn{convLayerIndex}.patchSize, cnn{convLayerIndex}.meanPatch, cnn{convLayerIndex}.poolSize, cnn{convLayerIndex}.convolutionsStepSize);

    convOut = permute(cpFeatures, [4 3 1 2]);
    X = reshape(convOut, cnn{convLayerIndex}.outputSize, numTestImages);
    
end

[prediction, mlp_confidence] = mlpPredict(Theta3, Theta4, X, maxTopPredictions);

%% -----------------------------------------------------
% show (in)correct prediction value and amount of correct samples
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