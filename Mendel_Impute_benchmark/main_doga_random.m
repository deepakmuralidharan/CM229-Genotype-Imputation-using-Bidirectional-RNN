clear; clc;
D=importdata('/Users/deepakmuralidharan/Documents/Bidirectional-LSTM/data/Diploidrandom.txt');
%D=randi([0 2],1092,500);
D = D';
n = 1895;
ndx = randperm(numel(D), n);
[row,col] = ind2sub(size(D), ndx);

M = D;
Masked_M = D;
for i = 1:length(row)
    Masked_M(row(i),col(i)) = NaN;
end

dlmwrite('data/Masked_Mprime.txt',Masked_M');

w=16;
tic
Z = Mendel_IMPUTE('data/Masked_Mprime.txt', w);
toc
dlmwrite('data/Imputed_prime.txt',Z');


%x = 1:26;
%stem(x,sum(errors));
%xlabel('SNP position');
%ylabel('Number of mismatches (out of 92)');
%title('SNP position vs Mismatches [Mendel Impute on Diploid Data]');