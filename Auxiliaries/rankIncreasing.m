function stats_RI=rankIncreasing(tildeX,eucGrad,normgradient,r,normaltol,basis)


underliner=size(tildeX.core);

stats_RI.canIncrease=false;
if underliner(1)==r(1) || underliner(2)==r(2) || underliner(3)==r(3)
    stats_RI.canIncrease=false;
else
    [tildeU1,tildeU2,tildeU3,~,~,~]=computeNormalBasis(tildeX,underliner+[1 1 1],eucGrad,basis);
    
    temp=computeAk1d(uint32(eucGrad.subs'),eucGrad.vals,tildeU1',tildeU2',tildeU3');
    
    normalVec=ttensor(tensor(temp,[1 1 1]),{tildeU1,tildeU2,tildeU3});
    
    if abs(temp)>normgradient*normaltol
        stats_RI.canIncrease=true;
        
        
        vals=getValsAtIndex(normalVec,eucGrad.subs);
% 
        s=-(vals'*eucGrad.vals)/(vals'*vals);
%         s=0.1;
        
        coreTensor=tenzeros(underliner+[1 1 1]);
        coreTensor(1:underliner(1),1:underliner(2),1:underliner(3))=tildeX.core;
        coreTensor(1+underliner(1),1+underliner(2),1+underliner(3))=s*temp;
        
        
        stats_RI.Xnew=ttensor(coreTensor,{[tildeX.U{1} tildeU1],[tildeX.U{2} tildeU2],[tildeX.U{3} tildeU3]});
    end
end

