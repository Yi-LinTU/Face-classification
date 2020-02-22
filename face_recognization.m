clear;

faceDatasetPath = fullfile('D:', 'Code', 'CVR', 'CroppedYale');
imds = imageDatastore(faceDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


% figure;
% perm = randperm(2470,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end


labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)

numTrainFiles = 35;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


layers = [
    imageInputLayer([192 168 1])
    
    %-------------------------------------------------
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
    convolution2dLayer(3,128,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
%     convolution2dLayer(3,256,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     %-------------------------------------------------
%     convolution2dLayer(3,512,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
    %-------------------------------------------------
    
    

    fullyConnectedLayer(38)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'ValidationPatience',5,...
    'Momentum',0.9,...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

% test = imread('D:\Code\CVR\PRE\example\test.jpg');
% test = rgb2gray(test);
% 
% testPred = classify(net, test)




