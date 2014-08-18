%% Network configuration

% plot errors (1- plot, 0 - hide)
plotTrainError = 1;
plotValidationError = 1;
plotTestError = 1;

startTestIteration = 0; % iteration where start running prediction test 

max_class_samples = 1000; % max sample amount per output class (used for huge classes e.g. Roman Empire)

maxTestSamples = 200; % if test set is large - create subset 

% !! WHEN CHANGE batchSize - CLEAN UP / DELETE TEMP DIRECTORY (tempDir)
batchSize = 24; % batch size for mini-batch algorithm

numTrainIterFC = 100; % number of training iterations for full connected layers

numOutputClasses = 280; % 276

imgW = 400; % image width, ( width >= height )
imgH = 200; % image height

cnn = cell(1, 1); % for convolution layers L2

% L2
cnn{1}.inputWidth = imgW;
cnn{1}.inputHeight = imgH;
cnn{1}.inputChannels = 1;
cnn{1}.features = 100;
cnn{1}.patchSize = 4;
cnn{1}.poolSize = 6;
cnn{1}.numPatches = 100000;
cnn{1}.inputVisibleSize = cnn{1}.patchSize * cnn{1}.patchSize * cnn{1}.inputChannels;
cnn{1}.convolutionsStepSize = 50;
cnn{1}.saeSparsityParam = 0.1;   % desired average activation of the hidden units.
cnn{1}.saeLambda = 0.003;     % weight decay for SAE (sparse auto-encoders)       
cnn{1}.saeBeta = 3;            % weight of sparsity penalty term       

cnn{1}.outputWidth = floor((cnn{1}.inputWidth - cnn{1}.patchSize + 1) / cnn{1}.poolSize);
cnn{1}.outputHeight = floor((cnn{1}.inputHeight - cnn{1}.patchSize + 1) / cnn{1}.poolSize);
cnn{1}.outputChannels = cnn{1}.features;
cnn{1}.outputSize = cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels;

% L3
cnn{2}.inputWidth = cnn{1}.outputWidth;
cnn{2}.inputHeight = cnn{1}.outputHeight;
cnn{2}.inputChannels = cnn{1}.outputChannels;
cnn{2}.features = 100;
cnn{2}.patchSize = 4;
cnn{2}.poolSize = 4;
cnn{2}.numPatches = 100000;
cnn{2}.inputVisibleSize = cnn{2}.patchSize * cnn{2}.patchSize * cnn{2}.inputChannels;
cnn{2}.convolutionsStepSize = 50;
cnn{2}.saeSparsityParam = 0.1;   % desired average activation of the hidden units.
cnn{2}.saeLambda = 0.003;     % weight decay for SAE (sparse auto-encoders)       
cnn{2}.saeBeta = 3;            % weight of sparsity penalty term       

cnn{2}.outputWidth = floor((cnn{2}.inputWidth - cnn{2}.patchSize + 1) / cnn{2}.poolSize);
cnn{2}.outputHeight = floor((cnn{2}.inputHeight - cnn{2}.patchSize + 1) / cnn{2}.poolSize);
cnn{2}.outputChannels = cnn{2}.features;
cnn{2}.outputSize = cnn{2}.outputWidth * cnn{2}.outputHeight * cnn{2}.outputChannels;


%{
% L4
cnn{3}.inputWidth = cnn{2}.outputWidth;
cnn{3}.inputHeight = cnn{2}.outputHeight;
cnn{3}.inputChannels = cnn{2}.outputChannels;
cnn{3}.features = 100;
cnn{3}.patchSize = 3;
cnn{3}.poolSize = 3;
cnn{3}.numPatches = 100000;
cnn{3}.inputVisibleSize = cnn{3}.patchSize * cnn{3}.patchSize * cnn{3}.inputChannels;
cnn{3}.convolutionsStepSize = 50;
cnn{3}.saeSparsityParam = 0.1;   % desired average activation of the hidden units.
cnn{3}.saeLambda = 0.003;     % weight decay for SAE (sparse auto-encoders)       
cnn{3}.saeBeta = 3;            % weight of sparsity penalty term       

cnn{3}.outputWidth = floor((cnn{3}.inputWidth - cnn{3}.patchSize + 1) / cnn{3}.poolSize);
cnn{3}.outputHeight = floor((cnn{3}.inputHeight - cnn{3}.patchSize + 1) / cnn{3}.poolSize);
cnn{3}.outputChannels = cnn{3}.features;
cnn{3}.outputSize = cnn{3}.outputWidth * cnn{3}.outputHeight * cnn{3}.outputChannels;


L5
cnn{4}.inputWidth = cnn{3}.outputWidth;
cnn{4}.inputHeight = cnn{3}.outputHeight;
cnn{4}.inputChannels = cnn{3}.outputChannels;
cnn{4}.features = 100;
cnn{4}.patchSize = 3;
cnn{4}.poolSize = 2;
cnn{4}.numPatches = 100000;
cnn{4}.inputVisibleSize = cnn{4}.patchSize * cnn{4}.patchSize * cnn{4}.inputChannels;
cnn{4}.convolutionsStepSize = 50;
cnn{4}.saeSparsityParam = 0.1;   % desired average activation of the hidden units.
cnn{4}.saeLambda = 0.003;     % weight decay for SAE (sparse auto-encoders)       
cnn{4}.saeBeta = 3;            % weight of sparsity penalty term       

cnn{4}.outputWidth = floor((cnn{4}.inputWidth - cnn{4}.patchSize + 1) / cnn{4}.poolSize);
cnn{4}.outputHeight = floor((cnn{4}.inputHeight - cnn{4}.patchSize + 1) / cnn{4}.poolSize);
cnn{4}.outputChannels = cnn{4}.features;
cnn{4}.outputSize = cnn{4}.outputWidth * cnn{4}.outputHeight * cnn{4}.outputChannels;
%}

% L FC 1 (first full connected layer)
inputSizeFCL1 = cnn{2}.outputSize; 

% L FC 2 (second full connected layer)
inputSizeFCL2 = 1000;

mlpLambda = 1e-3; % weight decay for L3


addpath ./libs/         % load libs
addpath ./libs/minFunc/

%  Use minFunc to minimize cost functions
saeOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
saeOptions.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
saeOptions.display = 'on';

mlpOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
mlpOptions.maxIter = 1; % update minFunc confugs for mini batch 
mlpOptions.display = 'off';