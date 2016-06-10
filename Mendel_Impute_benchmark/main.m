
clc;
close all;
clear all;

D=importdata('/Users/deepakmuralidharan/Documents/Bidirectional-LSTM/data/geno_loc_new_diploid.txt');
%D=randi([0 2],1092,500);
M=D(:,1:50);
Masked_M=M;
errors = []
%Masked_M(2084:2184,15)=NaN;
w=16;
for i=1:50
i
Masked_M=M; 
Masked_M(1001:1092,i)=NaN; %%imputing :)
dlmwrite('Data/Masked_Mprime',Masked_M');
Z = Mendel_IMPUTE('Data/Masked_Mprime', w);
Z1=round(Z);
Z1=Z1';
error_i=(Z1(1001:1092,i)~=M(1001:1092,i));
errors=[errors error_i];
end

x = 1:50;
y = sum(errors);
stem(x,y);