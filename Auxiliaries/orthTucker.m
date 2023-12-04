function Xnew=orthTucker(X)


core=X.core;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};


[Q1,R1]=qr(U1,0);
[Q2,R2]=qr(U2,0);
[Q3,R3]=qr(U3,0);


Xnew=ttensor(ttm(core, {R1, R2, R3}),{Q1,Q2,Q3});
