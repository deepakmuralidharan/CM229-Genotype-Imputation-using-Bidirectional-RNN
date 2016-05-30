function [U,s,V] = svt(X,lambda,varargin)
% SVT Singular value thresholding
%
% INPUT:
%   X - p1-by-p2 matrix
%   lambda - threshold
%   k - number of singular values to try
%
% Output:
%   U - left singular vectors
%   S - thresholded singular values
%   V - right singular vectors
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu)

% input parsing rule
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('lambda', @(x) x>=0);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', 1, @isnumeric);
% parse inputs
argin.parse(X,lambda,varargin{:});
%pentype = argin.Results.penalty;
%penparam = argin.Results.penparam;

if (isinf(lambda))
    U=[]; V=[]; s=[];
    return;
end

[U,S,V] = svd(X,0);
s = diag(S);
if (lambda>0)
%    s = s(s > lambda) - lambda;  
%     lsq_sparsereg(eye(length(s)),s,lambda,'penalty',pentype,...
%         'penparam',penparam);
    idx = find(s>lambda,1,'last');
    s = s(1:idx) - lambda;
    U = U(:,1:idx);
    V = V(:,1:idx);
end
    
end