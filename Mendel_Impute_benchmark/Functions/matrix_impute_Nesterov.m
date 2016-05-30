function [Y,stats] = matrix_impute_Nesterov(X,lambda,varargin)
% MATRIX_IMPUTE Impute a set of matrices using Nesterov method
%
% INPUT:
%   X - p1-by-p2-by-n data matrices for imputation; missing values are nan
%
% Output:
%   Z - the imputation matrix
%   stats - algorithmic statistics

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('lambda', @isnumeric);
argin.addParamValue('delta', 1, @(x) isnumeric(x) && x>0);
argin.addParamValue('Display', 'iter', @(x) ischar(x));
argin.addParamValue('MaxIter', 1000, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-5, @(x) isnumeric(x) && x>0);
argin.addParamValue('Y0', [], @(x) isnumeric(x) || isempty(x));
argin.parse(X,lambda,varargin{:});
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
ridgedelta = argin.Results.delta;
TolFun = argin.Results.TolFun;
Y0 = argin.Results.Y0;

% check dimensions and retrieve mising entry information
[p1,p2,n] = size(X);
Wts = n-sum(isnan(X),3);
ObsIdx = find(Wts>0);
Wts = Wts(ObsIdx);
Xavg = mean(X,3);
Xavg = Xavg(ObsIdx);

% initialize
alpha_old = 0; alpha = 1;
if (isempty(Y0))
    Y = zeros(p1,p2);
else
    Y = Y0;
end

% main loop
Y_old = Y;
objval = inf;
for iter=1:MaxIter

    % current search point
    S = Y+(alpha_old-1)/alpha*(Y-Y_old);
    resid = Xavg-S(ObsIdx);
    lossS = sum(Wts.*resid.^2)/2;
    lossD1S = -Wts.*resid;
    
    % line search
    Y_old = Y;
    objval_old = objval;
    for l=1:50
        A = S;
        A(ObsIdx) = A(ObsIdx)-ridgedelta*(lossD1S);
        [U,s,V] = svt(A,ridgedelta*lambda);
        if (isempty(s))
            stats.maxlambda = svds(A,1);
            stats.rank = length(s);
            stats.objval = sum(Wts.*Xavg.^2)/2;
            stats.iterations = iter;
            return;
        end
        Y = bsxfun(@times,U,s')*V';
        resid = Xavg-Y(ObsIdx);
        % objective value
        objval = sum(Wts.*resid.^2)/2 + lambda*sum(s);
        % surrogate value
        YminusS = Y - S;
        surval = lossS + sum(lossD1S.*YminusS(ObsIdx)) ...
            + norm(YminusS,'fro')^2/2/ridgedelta ...
            + lambda*sum(s);
        % line search stopping rule
        if (objval<=surval)
            break;
        else
            ridgedelta = ridgedelta/2;
        end
    end
    
    % force descent
    if (objval<=objval_old+1e-8) % descent
        % stopping rule
        if (abs(objval_old-objval)<TolFun*(abs(objval_old)+1))
            break;
        end
    else % no descent
        Y = Y_old;
        objval = objval_old;
    end
    
    % display
    if (~strcmpi(Display,'off'))
        display(['iter ' num2str(iter) ', objval=' num2str(objval)]);
    end

    % update alpha constants
    alpha_old = alpha;
    alpha = (1+sqrt(4+alpha_old^2))/2;

end

% collect algorithmic statistics
stats.iterations = iter;
stats.rank = length(s);
stats.objval = objval;

end