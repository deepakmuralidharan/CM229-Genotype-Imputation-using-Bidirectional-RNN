function [imputed_window, nstats] = tune_and_impute_Nesterov(X_tr, rho, gridpts, impute_ix, tuning_window, tune_ix, tuning_sample, tol)
% TUNE_AND_IMPUTE_NESTEROV Impute a set of matrices using Nesterov method
%
% INPUT:
%   X_tr - p1-by-p2 data matrices for imputation; missing values are nan
%   rho - factor to decrease regularization parameter by
%   gridpts - number of lambda grid points to try.
%   impute_ix - the row indices of X to be imputed.
%   tuning_window - the submatrix used to pick the regularization parameter
%                   lambda
%   tune_ix   - the row indices of X corresponding to tuning_window.
%   tuning_sample - the indices of entries in the tuning_window used for
%                   selecting lambda.
%
% Output:
%   imputed_window - the imputated matrix (this is not discretized!)
%   nstats - algorithmic statistics

rho_tol = 1e-4;

%% find max lambda for nuclear norm regularization
[dummy,stats] = matrix_impute_Nesterov(X_tr,inf);
maxlambda = stats.maxlambda;
%disp(maxlambda);

%% optimization on grid pts with warm start

lambdas = maxlambda * (rho.^(0:(gridpts-1)));

nLambda = length(lambdas);
miss_rate = zeros(nLambda,1);
Zi = cell(gridpts,1);
ranks = zeros(gridpts,1);
objval = zeros(gridpts,1);
iterations = zeros(gridpts,1);

% First complete a fixed number of grid points.
for i=1:gridpts
    lambda = lambdas(i);
    if (i==1)
        Y0 = [];
    else
        Y0 = Z;
    end
    [Z,stats] = matrix_impute_Nesterov(X_tr,lambda,'Y0',Y0,'Display','off','TolFun',tol);
    ranks(i) = stats.rank;
    objval(i) = stats.objval;
    iterations(i) = stats.iterations;
    Zi{i} = Z;
    Ziv = Zi{i}(tune_ix,:);
    ziv = round(max(min(Ziv(tuning_sample),2),0));
    miss_rate(i) = nnz(tuning_window(tuning_sample) ~= ziv)/length(tuning_sample);
end

ix_best = find(miss_rate == min(miss_rate),1,'first');

% If the miss rate did not bottom out in the first grid points, keep
% relaxing the regularization parameter.
if (ix_best == gridpts)
    while miss_rate(i) <= miss_rate(i-1) - rho_tol*(miss_rate(i-1) + 1)
        lambda = rho*lambda;
        Y0 = Z;
        [Z, stats] = matrix_impute_Nesterov(X_tr,lambda,'Y0',Y0,'Display','off','TolFun',tol);
        i = i + 1;
        lambdas(i) = lambda;
        ranks(i) = stats.rank;
        objval(i) = stats.objval;
        iterations(i) = stats.iterations;
        Zi{i} = Z;
        Ziv = Zi{i}(tune_ix,:);
        ziv = round(max(min(Ziv(tuning_sample),2),0));
        miss_rate(i) = nnz(tuning_window(tuning_sample) ~= ziv)/length(tuning_sample);
    end
    ix_best = find(miss_rate == min(miss_rate),1,'first');
end

Xi = Zi{ix_best};

nstats.rank = ranks(ix_best);
nstats.miss_rate = miss_rate(ix_best);
nstats.lambda = lambdas(ix_best);
nstats.objval = objval(ix_best);
nstats.iterations = iterations(ix_best);

imputed_window = Xi(impute_ix,:);