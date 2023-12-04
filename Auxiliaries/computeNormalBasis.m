function [tildeU1,tildeU2,tildeU3,temp1,temp2,temp3]=computeNormalBasis(X,r,eucGrad,basis)
% computeNormalBasis compute the normal basis Uk1
% [tildeU1,tildeU2,tildeU3,temp1,temp2,temp3]=computeNormalBasis(X,r,eucGrad,basis)
% Input:
%   X: an iterate (ttensor)
%   r: rank parameter (larger than the rank of X)
%   eucGrad: Euclidean gradient (ttensor)
%   basis: approach generating bases ('random' or 'orth', 'random' by
%   default) 
%
% Output: 
%   tildeU1, tildeU2, tildeU3: selected bases
%   temp1, temp2, temp3: precomputed Aneqk (only applicable to 'orth')
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Oct. 27, 2023.

dims=size(X);
core=X.core;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
underliner=size(core);

temp1=[];
temp2=[];
temp3=[];
if strcmp(basis,'orth')==1
    [temp1,temp2,temp3]=computeAneqk(uint32(eucGrad.subs'),eucGrad.vals,U1',U2',U3');
    temp1=reshape(temp1, [dims(1) underliner(2)*underliner(3)]);
    temp2=reshape(temp2, [dims(2) underliner(1)*underliner(3)]);
    temp3=reshape(temp3, [dims(3) underliner(1)*underliner(2)]);

    [U12,~,~]=svd(temp1-U1*(U1'*temp1),'econ');
    [U22,~,~]=svd(temp2-U2*(U2'*temp2),'econ');
    [U32,~,~]=svd(temp3-U3*(U3'*temp3),'econ');
    
    tildeU1=U12(:,1:(r(1)-underliner(1)));
    tildeU2=U22(:,1:(r(2)-underliner(2)));
    tildeU3=U32(:,1:(r(3)-underliner(3)));
else
    tildeU1_temp=[U1 randn(dims(1),r(1)-underliner(1))];
    [Q,~]=qr(tildeU1_temp,0);
    tildeU1=Q(:,underliner(1)+1:end);
    
    tildeU2_temp=[U2 randn(dims(2),r(2)-underliner(2))];
    [Q,~]=qr(tildeU2_temp,0);
    tildeU2=Q(:,underliner(2)+1:end);
    
    tildeU3_temp=[U3 randn(dims(3),r(3)-underliner(3))];
    [Q,~]=qr(tildeU3_temp,0);
    tildeU3=Q(:,underliner(3)+1:end);
end
