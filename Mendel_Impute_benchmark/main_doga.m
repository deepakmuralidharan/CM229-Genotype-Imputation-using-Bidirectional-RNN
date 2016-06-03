clear; clc;
D=importdata('/Users/deepakmuralidharan/Documents/Bidirectional-LSTM/data/diploid_random.txt');
%D=randi([0 2],1092,500);
D = D';
M=D(:,1:26);
Masked_M=M; 
%Masked_M(2084:2184,15)=NaN;
w=8;
corrs = [];
for i=1:26
i
Masked_M=M; 
Masked_M(51:250,i)=NaN; %%imputing :)
dlmwrite('Data/Masked_Mprime',Masked_M');
Z = Mendel_IMPUTE('Data/Masked_Mprime', w);
Z1=round(Z);
Z1=Z1';
%isplay(Z1(:,i));
corr_i = corr(Z1(51:250,i),M(51:250,i))^2;
display(corr_i);
corrs = [corrs corr_i];
end

%x = 1:26;
%stem(x,sum(errors));
%xlabel('SNP position');
%ylabel('Number of mismatches (out of 92)');
%title('SNP position vs Mismatches [Mendel Impute on Diploid Data]');