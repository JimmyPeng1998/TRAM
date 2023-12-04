function stats_RD=rankDecreasing(X,Delta)
% rankDecreasing applying rank-decreasing procedure
% stats_RD=rankDecreasing(X,Delta)
% Input:
%   X: an iterate (ttensor)
%   Delta: threshold
%
% Output: 
%   stats_RD.canDecrease: whether or not decrease the rank
%   stats_RD.r1, stats_RD.r2, stats_RD.r3: Tucker rank after truncation
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Nov. 02, 2023.


underliner=size(X.core);
temp1=double(tenmat(X.core,1));
[~,S1,~]=svd(temp1,'econ');
sigma1=diag(S1);

temp2=double(tenmat(X.core,2));
[~,S2,~]=svd(temp2,'econ');
sigma2=diag(S2);

temp3=double(tenmat(X.core,3));
[~,S3,~]=svd(temp3,'econ');
sigma3=diag(S3);

if  sigma1(end)/sigma1(1)<Delta ||...
    sigma2(end)/sigma2(1)<Delta ||...
    sigma3(end)/sigma3(1)<Delta

    stats_RD.canDecrease=true;
    if sigma1(end)<Delta*sigma1(1)
        stats_RD.r1=find(sigma1<Delta*sigma1(1),1)-1;
    else
        stats_RD.r1=underliner(1);
    end
    if sigma2(end)<Delta*sigma2(1)
        stats_RD.r2=find(sigma2<Delta*sigma2(1),1)-1;
    else
        stats_RD.r2=underliner(2);
    end
    if sigma3(end)<Delta*sigma3(1)
        stats_RD.r3=find(sigma3<Delta*sigma3(1),1)-1;
    else
        stats_RD.r3=underliner(3);
    end
else
    stats_RD.canDecrease=false;
end
    