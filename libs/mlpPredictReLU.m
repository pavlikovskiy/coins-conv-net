function [prediction, mlp_confidence] = mlpPredictReLU(Theta1, Theta2, X, maxTopPredictions)
%MLPPREDICT Predict the label of an input given a trained neural network
%   prediction = MLPPREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%   prediction -  M x maxTopPredictions

m = size(X, 2); % amount of examples

h1 = relu(Theta1' * [ones(1, m); X]);
h2 = relu(Theta2' * [ones(1, m); h1]);

% -----debug info------------------
%mlp_debug_output = h2';
%mlp_confidence = sum(mlp_debug_output, 2);
mlp_confidence = sum(h2, 1);
% -----------------------


[dummy, thisPred] = max(h2, [], 1);

prediction = zeros(m, maxTopPredictions); % M x maxTopPredictions

prediction(:, 1) = thisPred';

if maxTopPredictions > 1
    for j = 2:maxTopPredictions

        for i = 1:m
            h2(thisPred(i), i) = NaN; % reset previous max
        end
        [nop, thisPred] = max(h2, [], 1);
        prediction(:, j) = thisPred';
    end
end

% =========================================================================

end