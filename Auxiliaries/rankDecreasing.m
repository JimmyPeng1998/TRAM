function stats_RD=rankDecreasing(tildeX,Delta)

underliner=size(tildeX.core);
temp1=double(tenmat(tildeX.core,1));
[~,S1,~]=svd(temp1,'econ');
sigma1=diag(S1);

temp2=double(tenmat(tildeX.core,2));
[~,S2,~]=svd(temp2,'econ');
sigma2=diag(S2);

temp3=double(tenmat(tildeX.core,3));
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
    