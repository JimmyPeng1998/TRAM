function Xnew=retractionVariety(X,t,g,r)


[U1,R1]=qr(g.U1_tilde,0);
[U2,R2]=qr(g.U2_tilde,0);
[U3,R3]=qr(g.U3_tilde,0);
underliner=size(X.core);

core=t*g.Y_tilde;
if underliner(1)==1 || underliner(2)==1 || underliner(3)==1
    core(1:underliner(1),1:underliner(2),1:underliner(3))=...
        tensor(double(core(1:underliner(1),1:underliner(2),1:underliner(3))),underliner)+X.core;
else
    core(1:underliner(1),1:underliner(2),1:underliner(3))=...
        core(1:underliner(1),1:underliner(2),1:underliner(3))+X.core;
end


core = ttm( core, {R1, R2, R3});


HO = hosvd(core, r);

Q1 = U1 * HO.U{1};
Q2 = U2 * HO.U{2};
Q3 = U3 * HO.U{3};

Xnew = ttensor( HO.core, {Q1, Q2, Q3});