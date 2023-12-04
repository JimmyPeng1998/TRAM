function s=calcInitial_Variety(eucGrad,g,A_Omega)
% calcInitial_Variety compute the initial guess by exact line-search on
% Tucker tensor varieties
% s=calcInitial_Variety(eucGrad,g,A_Omega)
% Input:
%   eucGrad: Euclidean gradient (ttensor)
%   g: tangent cone variables
%   A_Omega: training set (sptensor)
%
% Output: 
%   s: initial stepsize
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Oct. 26, 2023.

g_tc=ttensor(g.Y_tilde,{g.U1_tilde,g.U2_tilde,g.U3_tilde});

vals=getValsAtIndex(g_tc,A_Omega.subs);

s=-(vals'*eucGrad.vals)/(vals'*vals);


