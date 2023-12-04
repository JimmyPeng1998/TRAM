function g=computeApproxProj(X,r,eucGrad,tildeU1,tildeU2,tildeU3,temp1,temp2,temp3)
% computeApproxProj compute the approximate projection of the gradient onto
% Tucker tensor varieties
% s=computeApproxProj(eucGrad,g,A_Omega)
% Input:
%   X: an iterate (ttensor)
%   r: rank parameter (larger than the rank of X)
%   eucGrad: Euclidean gradient (ttensor)
%   tildeU1, tildeU2, tildeU3: selected bases
%   temp1, temp2, temp3: precomputed Aneqk (can be empty)
%
% Output: 
%   g: approximate projection
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Oct. 27, 2023.

core=X.core;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
underliner=size(core);
dims=size(X);

mergeU1=[U1 tildeU1];
mergeU2=[U2 tildeU2];
mergeU3=[U3 tildeU3];

% Core tensor
Core=tenzeros(underliner+r);


% Factor matrices
S1_inv = pinv( double( tenmat( core, 1 ) ));
S2_inv = pinv( double( tenmat( core, 2 ) ));
S3_inv = pinv( double( tenmat( core, 3 ) ));

if isempty(temp1)
    [temp1,temp2,temp3]=computeAneqk(uint32(eucGrad.subs'),eucGrad.vals,U1',U2',U3');
    temp1=reshape(temp1, [dims(1) underliner(2)*underliner(3)])*S1_inv;
    temp2=reshape(temp2, [dims(2) underliner(1)*underliner(3)])*S2_inv;
    temp3=reshape(temp3, [dims(3) underliner(1)*underliner(2)])*S3_inv;
    
else
    temp1=temp1*S1_inv;
    temp2=temp2*S2_inv;
    temp3=temp3*S3_inv;
end



temp=computeAk1d(uint32(eucGrad.subs'),eucGrad.vals,mergeU1',mergeU2',mergeU3');
temp=reshape(temp,[r(1) r(2) r(3)]);

U12=temp1-mergeU1*(mergeU1'*temp1);
U22=temp2-mergeU2*(mergeU2'*temp2);
U32=temp3-mergeU3*(mergeU3'*temp3);



Core(1:r(1),1:r(2),1:r(3))=temp;
Core(1:underliner(1),1:underliner(2),r(3)+1:end)=core;
Core(1:underliner(1),r(2)+1:end,1:underliner(3))=core;
Core(r(1)+1:end,1:underliner(2),1:underliner(3))=core;




% Results
g=struct('Y_tilde', Core, 'U1_tilde', [mergeU1 U12],...
    'U2_tilde', [mergeU2 U22],...
    'U3_tilde', [mergeU3 U32]);