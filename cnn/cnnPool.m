function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
% Create an average filter matrix whose cells are given
% the value 1 / (poolDim * poolDim) to implement mean pooling
averageFilter = zeros(poolDim, poolDim);
averageFilter = averageFilter + 1 / (poolDim * poolDim);

for imageNum = 1:numImages
  for filterNum = 1:numFilters
      % Obtain the convolved image
      convolvedImage = convolvedFeatures(:, :, filterNum, imageNum);
      % Apply mean pooling filter
      pooledImage = conv2(convolvedImage, averageFilter, 'valid');
      % Obtain the coordinates of the non-overlapping regions of the image
      arr = 1: poolDim: size(pooledImage, 1);
      % Extract the non-overlapping regions and assign them to
      % the pooled features
      pooledFeatures(:, :, filterNum, imageNum) = pooledImage(arr, arr);
  end
end
end

