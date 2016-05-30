function [Z, stats] = Mendel_IMPUTE(filename, w, varargin)
% MENDEL_IMPUTE Impute a contiguous stretch of matrices
% using Nesterov method.
%
%   Z = MENDEL_IMPUTE(filename, w) returns an imputed
%      matrix Z (not discretized!) that results from imputing the middle 
%      third of a sliding window of width w. The input file is a p-by-n 
%      matrix where p is the number of SNPs and n is the number of
%      subjects. 'filename' is read top-to-bottom. The entries in the file
%      should be coded as {0,1,2}, i.e. a dosage model with respect to a
%      reference allele.
%
%   Z = MENDEL_IMPUTE(input_file, w, 'param', value,...)
%      specifies optional parameters and values. Valid parameters and their
%      default values are:
%      'seed' - seed to feed into RandStream {12345}
%      'max_iter' - Maximum number of windows {2500}
%      'p_tune' - fraction to hold out for tuning {0.1}
%      'tol' - Tolerance for Nesterov Method {1e-4}
%      'gridpts' - Minimum number of regularization parameter points {11}
%      'rho' {0.5}
%
%   [Z, stats] = MENDEL_IMPUTE(...) also returns
%      algorithmic statistics from the imputation procedure.
%      stats.cputime
%      stats.ranks
%      stats.miss_rate
%      stats.lambdas
%      stats.objval
%      stats.iterations
%
%   See also matrix_impute_Nesterov and tune_and_impute_Nesterov.

%% Set algorithm parameters from input or by using defaults.
params = inputParser;
params.addParamValue('seed',12345,@isscalar);
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('p_tune',0.1,@isscalar);
params.addParamValue('max_iter',6000,@(x) isscalar(x) & x > 0);
params.addParamValue('gridpts',11,@isscalar);
params.addParamValue('rho',0.5,@isscalar);
params.parse(varargin{:});

%% Copy from params object.
seed = params.Results.seed;
tol = params.Results.tol;
p_tune = params.Results.p_tune;
max_iter = params.Results.max_iter;
gridpts = params.Results.gridpts;
rho = params.Results.rho;

%% Initialize solution matrix and diagnostic variables.
[nSNPs, nSamples] = dim(filename);
nWindows = floor(nSNPs/w);
Z = zeros(nSNPs, nSamples);
%Z = zeros(min(w*max_iter, nSNPs), nSamples); 
stats.ranks = zeros(nWindows,1);
stats.miss_rate = zeros(nWindows,1);
stats.lambdas = zeros(nWindows,1);
stats.objval = zeros(nWindows,1);
stats.iterations = zeros(nWindows,1);

%% Set seed
stream = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(stream);

fid = fopen(filename);
assert(fid ~= -1, ['Error opening file ', filename]);
%% Impute the first subwindow.
tic;
[X, nSNPs] = block_read(fid,3*w);
impute_ix = 1:w;
tune_ix = setdiff(1:(3*w), impute_ix); % Row indices corresponding to one-third training blocks.
fprintf('Imputing SNPs %d:%d\n', 1, w);

%% determine hold out sets for parameter tuning and validation.
tuning_window = X(tune_ix,:);
[X_tr, tuning_sample] = block_mask(X, p_tune, tune_ix);

%% Impute
[imputed_window, nstats] = tune_and_impute_Nesterov(X_tr, rho, gridpts, impute_ix, tuning_window, tune_ix, tuning_sample, tol);
Z(impute_ix,:) = imputed_window;

%% Update algorithm statistics
stats.ranks(1) = nstats.rank;
stats.miss_rate(1) = nstats.miss_rate;
stats.lambdas(1) = nstats.lambda;
stats.objval(1) = nstats.objval;
stats.iterations(1) = nstats.iterations;

%% Impute the second subwindow.
impute_ix = (w+1):2*w;
tune_ix = setdiff(1:(3*w), impute_ix); % Row indices corresponding to one-third training blocks.
iter = 2;
fprintf('Imputing SNPs %d:%d\n', (iter-1)*w+1, iter*w);
X_last = X;
% Need to check if nSNPs == w.

%% determine hold out sets for parameter tuning and validation.
tuning_window = X(tune_ix,:);
[X_tr, tuning_sample] = block_mask(X, p_tune, tune_ix);

%% Impute
[imputed_window, nstats] = tune_and_impute_Nesterov(X_tr, rho, gridpts, impute_ix, tuning_window, tune_ix, tuning_sample, tol);
Z((iter-1)*w+1:iter*w,:) = imputed_window;

%% Update algorithm statistics
stats.ranks(2) = nstats.rank;
stats.miss_rate(2) = nstats.miss_rate;
stats.lambdas(2) = nstats.lambda;
stats.objval(2) = nstats.objval;
stats.iterations(2) = nstats.iterations;

%% Main loop to do imputations
for iter = 3:max_iter
    [dX, nSNPs] = block_read(fid,w);
    if (nSNPs < w)
        break
    end
    X = [X_last((w+1):(3*w),:); dX]; % Move window over by 1 subwindow.
    fprintf('Imputing SNPs %d:%d\n', (iter-1)*w+1, iter*w);
    X_last = X;
    
%% determine hold out sets for parameter tuning and validation.   
    tuning_window = X(tune_ix,:);
    [X_tr, tuning_sample] = block_mask(X, p_tune, tune_ix);
    
%% Impute
    [imputed_window, nstats] = tune_and_impute_Nesterov(X_tr, rho, gridpts, impute_ix, tuning_window, tune_ix, tuning_sample, tol);
    Z((iter-1)*w+1:iter*w,:) = imputed_window;
    
%% Update algorithm statistics
    stats.ranks(iter) = nstats.rank;
    stats.miss_rate(iter) = nstats.miss_rate;
    stats.lambdas(iter) = nstats.lambda;
    stats.objval(iter) = nstats.objval;
    stats.iterations(iter) = nstats.iterations;
end

%% Impute last block.
last_ix = 3*w + nSNPs;
impute_ix = (2*w+1):last_ix;
tune_ix = setdiff(1:last_ix, impute_ix); % Row indices corresponding to one-third training blocks.
fprintf('Imputing SNPs %d:%d\n', iter*w+1, iter*w+nSNPs);
if (nSNPs > 0)
   X = [X_last; dX]; 
end
    
%% determine hold out sets for parameter tuning and validation.
tuning_window = X(tune_ix,:);
[X_tr, tuning_sample] = block_mask(X, p_tune, tune_ix);

%% Impute
[imputed_window, nstats] = tune_and_impute_Nesterov(X_tr, rho, gridpts, impute_ix, tuning_window, tune_ix, tuning_sample, tol);
Z((iter-1)*w+1:iter*w+nSNPs,:) = imputed_window;
Z = Z(1:iter*w+nSNPs,:);

%% Update algorithm statistics
stats.ranks(end) = nstats.rank;
stats.miss_rate(end) = nstats.miss_rate;
stats.lambdas(end) = nstats.lambda;
stats.objval(end) = nstats.objval;
stats.iterations(end) = nstats.iterations;

%% Clean up.
stats.cputime = toc;

fclose(fid);

end

function [Xm,hold_out] = block_mask(X,p,rows_ix)
    target_block = X(rows_ix,:);
    hold_out_pool = find(isnan(target_block) == 0);
    nHold_out = floor(p*length(hold_out_pool));
    hold_out = randsample(hold_out_pool, nHold_out);
    Xm = X;
    tblock = target_block;
    tblock(hold_out) = NaN;
    Xm(rows_ix,:) = tblock;
end

function [X, nSNPs] = block_read(fid,n)
    nSNPs = n;
    for i = 1:n
        tline = fgets(fid);
        if (~ischar(tline))
            nSNPs = i-1;
            if (i == 1)
                X = NaN;
            else
                X(i:n,:) =  NaN;
            end
            break
        end
        X(i,:) = str2num(tline);
    end
end

function [nRow, nCol] =  dim(filename)
    fid = fopen(filename);
    assert(fid ~= -1, ['Error opening file ', filename]);
    [X, dummy] = block_read(fid,1);
    fclose(fid);
    nCol = size(X,2);

    command = sprintf('wc -l %s', filename);
    [status, result] = system(command);
    assert(status == 0, ['Error opening file ', filename]);
    nRow = sscanf(result,'%d');
end
