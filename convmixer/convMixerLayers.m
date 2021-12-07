function lgraph = convMixerLayers(opts)
% convMixerLayers   Build ConvMixer architecture.
%
% lgraph = convMixerLayers() returns a LayerGraph object with a ConvMixer
% architecture with default options as proposed in
% https://openreview.net/forum?id=TVHS5Y4dNvM.
%
% lgraph = convMixerLayers(PARAM1=VAL1,PARAM2=VAL2,...) specifies optional
% parameter name/value pairs for creating the layer graph:
%
%       'InputSize'           - Size of the input images.
%
%       'NumClasses'          - Number of classes the network predicts.
%
%       'KernelSize'          - Size of the kernel for the depthwise
%                               convolution.
%
%       'PatchSize'           - Size of the pathes for the patch embedding
%                               layer.
%
%       'Depth'               - Number of repeated fully-convolutional
%                               blocks.
%
%       'HiddenDimension'     - Number of channels output by the patch
%                               embedding.
%
%       'ConnectOutputLayer'  - Determines whether to append a softmax and
%                               classification output layers to the
%                               returned LayerGraph object.
%
% Example:
%
%   lgraph = convMixerLayers(InputSize=[28 28 1], Depth=5, NumClasses=10)

% Copyright 2021 The MathWorks, Inc.

arguments
    opts.InputSize = [227 227 3]
    opts.NumClasses = 1000
    opts.KernelSize = 9
    opts.PatchSize = 7
    opts.Depth = 20
    opts.HiddenDimension = 1536
    opts.ConnectOutputLayer logical = false
end

input_size = opts.InputSize;
num_classes = opts.NumClasses;

kernel_size = opts.KernelSize;
patch_size = opts.PatchSize;
depth = opts.Depth;
hidden_dim = opts.HiddenDimension;
connectOutputLayers = opts.ConnectOutputLayer;

% First layer is a "path embedding". Seems to be this:
patchEmbedding = convolution2dLayer(patch_size, hidden_dim, ...
    Stride=patch_size, ...
    Name="patchEmbedding", ...
    WeightsInitializer="glorot");

% Make Layer Graph
lgraph = layerGraph();

start = [
    imageInputLayer(input_size,Normalization="none")
    patchEmbedding
    geluLayer(Name="gelu_0")
    batchNormalizationLayer(Name="batchnorm_0")
    ];
lgraph = addLayers(lgraph,start);

for i = 1:depth
    convMixer = [
        groupedConvolution2dLayer(kernel_size,1,"channel-wise",Name="depthwiseConv_"+i,Padding="same",WeightsInitializer="glorot")
        geluLayer(Name="gelu_"+(2*i-1))
        batchNormalizationLayer(Name="batchnorm_"+(2*i-1))
        additionLayer(2,Name="addition_"+i)
        convolution2dLayer([1 1],hidden_dim,Name="pointwiseConv_"+i,WeightsInitializer="glorot")
        geluLayer(Name="gelu_"+2*i)
        batchNormalizationLayer(Name="batchnorm_"+2*i)
        ];
    lgraph = addLayers(lgraph,convMixer);
    lgraph = connectLayers(lgraph,"batchnorm_"+2*(i-1),"depthwiseConv_"+i);
    lgraph = connectLayers(lgraph,"batchnorm_"+2*(i-1),"addition_"+i+"/in2");
end

gapFc = [
    globalAveragePooling2dLayer(Name="GAP")
    fullyConnectedLayer(num_classes)
    ];
lgraph = addLayers(lgraph,gapFc);
lgraph = connectLayers(lgraph,"batchnorm_"+2*depth,"GAP");

if connectOutputLayers
    lgraph = addLayers(lgraph, softmaxLayer('Name','softmax'));
    lgraph = addLayers(lgraph, classificationLayer('Name','classification'));
    lgraph = connectLayers(lgraph,'fc','softmax');
    lgraph = connectLayers(lgraph,'softmax','classification');
end
end
