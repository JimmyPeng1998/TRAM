function stats_RI=rankIncreasing(X,eucGrad,normgradient,r,normaltol,basis)
% rankIncreasing applying rank-increasing procedure
% stats_RI=rankIncreasing(X,eucGrad,normgradient,r,normaltol,basis)
% Input:
%   X: an iterate (ttensor)
%   eucGrad: Euclidean gradient (ttensor)
%   normgradient: norm of Riemannian gradient
%   r: rank parameter
%   normaltol: threshold
%   basis: approach generating bases ('random' or 'orth', 'random' by
%   default) 
%
% Output: 
%   stats_RI.canIncrease: whether or not increase the rank
%   stats_RI.Xnew: New tensor after rank increasing
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Nov. 02, 2023.

underliner=size(X.core);

stats_RI.canIncrease=false;
if underliner(1)==r(1) || underliner(2)==r(2) || underliner(3)==r(3)
    stats_RI.canIncrease=false;
else
    [tildeU1,tildeU2,tildeU3,~,~,~]=computeNormalBasis(X,underliner+[1 1 1],eucGrad,basis);
    
    temp=computeAk1d(uint32(eucGrad.subs'),eucGrad.vals,tildeU1',tildeU2',tildeU3');
    
    normalVec=ttensor(tensor(temp,[1 1 1]),{tildeU1,tildeU2,tildeU3});
    
    if abs(temp)>normgradient*normaltol
        stats_RI.canIncrease=true;
        
        
        vals=getValsAtIndex(normalVec,eucGrad.subs);
% 
        s=-(vals'*eucGrad.vals)/(vals'*vals);
%         s=0.1;
        
        coreTensor=tenzeros(underliner+[1 1 1]);
        coreTensor(1:underliner(1),1:underliner(2),1:underliner(3))=X.core;
        coreTensor(1+underliner(1),1+underliner(2),1+underliner(3))=s*temp;
        
        
        stats_RI.Xnew=ttensor(coreTensor,{[X.U{1} tildeU1],[X.U{2} tildeU2],[X.U{3} tildeU3]});
    end
end

