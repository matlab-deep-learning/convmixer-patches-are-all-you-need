%% Train ConvMixer network with CIFAR-10 dataset.
%
% This example shows how to build a ConvMixer network architecture and
% train it with the CIFAR-10 dataset. The training uses the Fixed Weight
% decay ADAM algorithm.

% Copyright 2021 The MathWorks, Inc.

%% Download and load the CIFAR-10 dataset [1]

datadir = tempdir; 
downloadCIFARData(datadir);

[XTrain,YTrain,XValidation,YValidation] = loadCIFARData(datadir);

%% Build minibatchqueue objects for the training and validation sets

imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');

% Increase minibatch size for improved performance.
minibatchSize = 32;

mbqTrain = minibatchqueue(augimdsTrain,...
    MiniBatchFcn=@getTrainMinibatch,...
    MiniBatchFormat=["SSCB",""], MiniBatchSize=minibatchSize);

xVal = arrayDatastore( single(XValidation), 'IterationDimension', 4 );
valCat = categories(YValidation);

validationMinibatchSize = 32;

mbqTest = minibatchqueue(xVal,1, ...
    'MiniBatchSize',validationMinibatchSize, ...
    'MiniBatchFcn',@getValMinibatch, ...
    'MiniBatchFormat','SSCB');

%% Load a pretrained network or train a network from scratch 

loadNetwork = true;

if loadNetwork
    load("cifar10_convmixer256-8-9_200_adamw");
else
    % Instantiate a ConvMixer network and train from scratch
    lg = convMixerLayers(InputSize=imageSize,NumClasses=10,HiddenDimension=256,Depth=8,PatchSize=1);
    dlnet = dlnetwork(lg);
    
    numEpochs = 200;
    
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
    
    iteration = 0;
    
    start = tic;
    
    accfun = dlaccelerate(@modelGradients);
    
    averageGrad   = dlupdate(@(x)0*x, dlnet.Learnables);
    averageSqGrad = dlupdate(@(x)0*x, dlnet.Learnables);
    
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1.e-8;
    alpha_t = 0.01;
    weightDecay = 0.0001;
    
    for epoch=1:numEpochs
    
        reset(mbqTrain);
        shuffle(mbqTrain);
    
        while hasdata(mbqTrain)
    
            iteration = iteration + 1;
            
            [X,Y] = next(mbqTrain);
            
            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelGradients function and update the network state.
            [gradients,state,loss] = dlfeval(accfun,dlnet,X,Y);
            dlnet.State = state;
            
            loss = double(extractdata(gather(loss)));
    
            % Determine learning rate for time-based decay learning rate
            % schedule.
            eta_t = getScheduleMultiplier(epoch,numEpochs);
            
            % Update the network parameters using the ADAM optimizer with fixed
            % weight decay.
            updateFcn = @(dlnet,gradient,avgGrad,avgSqGrad) ...
                adamFixedWeightDecay(dlnet,gradient,avgGrad,avgSqGrad,...
                    eta_t,alpha_t,beta1,beta2,epsilon,iteration,weightDecay);
            [dlnet,averageGrad,averageSqGrad] = dlupdate(updateFcn,dlnet,gradients,averageGrad,averageSqGrad);
    
            % Display the training progress.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,loss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", loss = " + num2str(loss))
            drawnow
        end
    
        if epoch == 1 || (mod(epoch,10) == 0 && epoch < numEpochs)
            predictions = modelPredictions(dlnet,mbqTest,valCat);
            validationAccuracy = mean(predictions == YValidation);
    
            fprintf("Epoch %d - validation accuracy: %.4f\n", epoch, validationAccuracy );
        end
    end
end

%% Evaluate trained network

predictions = modelPredictions(dlnet,mbqTest,valCat);
validationAccuracy = mean(predictions == YValidation);
fprintf("End of training - validation accuracy: %.4f\n", validationAccuracy );


function [X,T] = getTrainMinibatch(X,T)
X = transformImages(X);
T = onehotencode(cat(2,T{:}),1);
end

function X = getValMinibatch(X)
X = transformImages(X);
end

function X = transformImages(X)
X = single(cat(4,X{:})) / 255;
end

function [gradients,state,loss] = modelGradients(dlnet,X,T)
[dlYPred,state] = forward(dlnet,X);

dlYPred = softmax(dlYPred);
loss = crossentropy(dlYPred,T);
gradients = dlgradient(loss,dlnet.Learnables);
end

function lr = getScheduleMultiplier(epoch,numEpochs)
% Cosine learning rate schedule from https://arxiv.org/abs/1608.03983v5

lr_min = 0;
lr_max = 1;

Ti = numEpochs;
restart_epoch = mod(epoch-1, Ti);

lr = lr_min + 0.5 * (lr_max-lr_min) * (1+cos(pi * restart_epoch/Ti));
end

function [parameter,averageGrad,averageSqGrad] = adamFixedWeightDecay(parameter,gradients,averageGrad,averageSqGrad,eta_t,learnRate,beta1,beta2,epsilon,iter,weightDecay)
% Fixed weight decay Adam optimizer from https://openreview.net/forum?id=rk6qdGgCZ
averageGradOld   = beta1*averageGrad;
averageSqGradOld = beta2*averageSqGrad;

averageGrad   = averageGradOld   + (1-beta1)*gradients;
averageSqGrad = averageSqGradOld + (1-beta2)*(gradients.^2);

averageGradHat   = averageGrad   / (1-beta1^iter); 
averageSqGradHat = averageSqGrad / (1-beta2^iter); 

parameter = parameter - eta_t * ( learnRate*averageGradHat./(sqrt(averageSqGradHat)+epsilon) + weightDecay * parameter );
end

function predictions = modelPredictions(dlnet,mbq,classes)
reset(mbq);
predictions = [];
while hasdata(mbq)
    dlXTest = next(mbq);
    dlYPred = predict(dlnet,dlXTest);
    dlYPred = softmax(dlYPred);
    
    YPred = onehotdecode(dlYPred,classes,1)';
    
    predictions = [predictions; YPred];
end
end

%% References

% [1] Krizhevsky, Alex. "Learning multiple layers of features from tiny images." (2009). https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf