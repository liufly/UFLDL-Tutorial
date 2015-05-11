function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
% Create an average filter matrix whose cells are given
% the value 1 / (poolDim * poolDim) to implement mean pooling
averageFilter = zeros(poolDim, poolDim);
averageFilter = averageFilter + 1 / (poolDim * poolDim);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        % Obtain the feature (filterDim x filterDim) needed during the 
        % convolution
        filter = Wc(:, :, filterNum);
        % Flip the feature matrix because of the definition of convolution, 
        % as explained later
        filter = rot90(squeeze(filter),2);
        % Obtain the image
        im = squeeze(images(:, :, imageNum));
        % Convolve "filter" with "im", adding the result to convolvedImage
        % be sure to do a 'valid' convolution
        convolvedImage = conv2(im, filter, 'valid');
        % Add the bias unit
        convolvedImage = convolvedImage + bc(filterNum, 1);
        % Apply sigmoid function
        convolvedImage = sigmoid(convolvedImage);
        % Store convolved image
        activations(:, :, filterNum, imageNum) = convolvedImage;
        % Apply mean pooling filter
        pooledImage = conv2(convolvedImage, averageFilter, 'valid');
        % Obtain the coordinates of the non-overlapping regions of the image
        arr = 1: poolDim: size(pooledImage, 1);
        % Extract the non-overlapping regions and assign them to
        % the pooled features
        activationsPooled(:, :, filterNum, imageNum) = pooledImage(arr, arr);   
    end
end   

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
% Compute fully connected layer
z = bsxfun(@plus, Wd * activationsPooled, bd);
e = exp(z);
% Softmax
probs = bsxfun(@rdivide, e, sum(e, 1));

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
% Compute cross entory error
index = sub2ind(size(probs), labels', [1 : numImages]);
ceCost = -sum(log(probs(index)));
% Compute cost
cost = ceCost / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
% Compute softmax error
error_d = full(sparse(labels', 1 : numImages, 1)) - probs;
error_d = -error_d / numImages;

% Propagate error_d through the subsampling and convolutional layer
error_c = zeros(convDim, convDim, numFilters, numImages);
for imageNum = 1:numImages
    % Obtain the error for the current image
    error = error_d(:, imageNum);
    % Transform the error into the same size as pooled images
    error_pooledImages = reshape(Wd' * error, size(arr, 2), ...
        size(arr, 2), numFilters);
    for filterNum = 1:numFilters
        % Obtain the error for the current filter
        error_pooledImage = error_pooledImages(:, :, filterNum);
        % Upsample
        delta_pool = (1 / (poolDim ^ 2)) * kron(error_pooledImage, ...
            ones(poolDim, poolDim));
        % Back propagate through convolutional layer
        % Compute sigmoid derivative
        fz = activations(:, :, filterNum, imageNum);
        fzPrime = fz .* (1 - fz);
        % Compute error for this subsampling and convolutional layer
        delta_pool = delta_pool .* fzPrime;
        % Store error
        error_c(:, :, filterNum, imageNum) = delta_pool;
        
    end
end

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
% Softmax layer
Wd_grad = error_d * activationsPooled';
bd_grad = sum(error_d, 2);

% Subsampling and convolutional layer
for imageNum = 1:numImages
    for filterNum = 1:numFilters
        % Obtain error computed in step 1c
        delta = error_c(:, :, filterNum, imageNum);     
        % Update gradients
        Wc_grad(:, :, filterNum) = Wc_grad(:, :, filterNum) + ...
            conv2(squeeze(images(:, :, imageNum)), ...
            rot90(squeeze(delta), 2), 'valid');
        bc_grad(filterNum) = bc_grad(filterNum) + sum(delta(:));
    end
end

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
