function Xnew=orthTucker(X)
% orthTucker orthogonalize a Tucker tensor
% Xnew=orthTucker(X)
% Input:
%   X: a Tucker tensor (ttensor)
%
% Output: 
%   Xnew: orthogonalized tensor (ttensor)
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Oct. 26, 2023.

core=X.core;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};


[Q1,R1]=qr(U1,0);
[Q2,R2]=qr(U2,0);
[Q3,R3]=qr(U3,0);


Xnew=ttensor(ttm(core, {R1, R2, R3}),{Q1,Q2,Q3});
