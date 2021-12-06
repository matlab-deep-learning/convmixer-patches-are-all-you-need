classdef geluLayer < nnet.layer.Layer
% geluLayer   GELU layer.
%
% gLayer = geluLayer() returns a geluLayer object.
%
% gLayer = geluLayer(PARAM1=VAL1,PARAM2=VAL2,...) specifies optional
% parameter name/value pairs for creating the layer graph:
% 
%       'Mode'           - Size of the input images. Options are 'fast
%
%       'Name'           - Name of the layer.
%
%    See https://paperswithcode.com/method/gelu for details.
%
% Example:
% 
%   gLayer = geluLayer()

%    Copyright 2021 The MathWorks, Inc.

    properties(SetAccess='private')
        Mode
    end

    methods
        function obj = geluLayer(opts)
            arguments
                opts.Mode string {mustBeMember(opts.Mode,["fast", "exact"])} = "fast";
                opts.Name string {mustBeText} = "gelu";
            end
            obj.Name = opts.Name;
            obj.Mode = opts.Mode;
        end

        function y = predict(obj,x)
            switch obj.Mode
                case "exact"
                    y = x/2.*(1+erf(x/sqrt(2)));
                case "fast"
                    y = x/2.*(1+tanh(sqrt(2/pi)*(x+0.044715*x.^3)));
                otherwise
                    error Unknown
            end
        end
    end
end