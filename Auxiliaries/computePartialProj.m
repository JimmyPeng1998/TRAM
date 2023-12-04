function [g,info]=computePartialProj(X,r,eucGrad,tildeU1,tildeU2,tildeU3,temp1,temp2,temp3)


core=X.core;
U1=X.U{1};
U2=X.U{2};
U3=X.U{3};
dims=size(X);
underliner=size(X.core);

mergeU1=[U1 tildeU1];
mergeU2=[U2 tildeU2];
mergeU3=[U3 tildeU3];

S1_inv = pinv( double( tenmat( core, 1 ) ));
S2_inv = pinv( double( tenmat( core, 2 ) ));
S3_inv = pinv( double( tenmat( core, 3 ) ));

% if isempty(temp1)
%     temp1=double(tenmat(ttm(eucGrad,{U2',U3'},[2 3]),1))*S1_inv;
%     temp2=double(tenmat(ttm(eucGrad,{U1',U3'},[1 3]),2))*S2_inv;
%     temp3=double(tenmat(ttm(eucGrad,{U1',U2'},[1 2]),3))*S3_inv;
% else
%     temp1=temp1*S1_inv;
%     temp2=temp2*S2_inv;
%     temp3=temp3*S3_inv;
% end

[temp,temp1,temp2,temp3]=computeAllProjection(uint32(eucGrad.subs'),eucGrad.vals,...
    U1',U2',U3',mergeU1',mergeU2',mergeU3');
% if isempty(temp1)
%     [temp1,temp2,temp3]=computeAneqk(uint32(eucGrad.subs'),eucGrad.vals,U1',U2',U3');
temp1=reshape(temp1, [dims(1) underliner(2)*underliner(3)])*S1_inv;
temp2=reshape(temp2, [dims(2) underliner(1)*underliner(3)])*S2_inv;
temp3=reshape(temp3, [dims(3) underliner(1)*underliner(2)])*S3_inv;

% else
%     temp1=temp1*S1_inv;
%     temp2=temp2*S2_inv;
%     temp3=temp3*S3_inv;
% end

U12=temp1-U1*(U1'*temp1);
U22=temp2-U2*(U2'*temp2);
U32=temp3-U3*(U3'*temp3);


% Gcore=ttm(eucGrad,{mergeU1',mergeU2',mergeU3'},[1 2 3]);
% temp=computeAk1d(uint32(eucGrad.subs'),eucGrad.vals,mergeU1',mergeU2',mergeU3');
temp=tensor(reshape(temp,[r(1) r(2) r(3)]),[r(1) r(2) r(3)]);





% ProjG=ttensor(Gcore,{mergeU1,mergeU2,mergeU3});
ProjG=ttensor(temp,{mergeU1,mergeU2,mergeU3});
Proj1=ttensor(core,{U12,U2,U3});
Proj2=ttensor(core,{U1,U22,U3});
Proj3=ttensor(core,{U1,U2,U32});

% tic
% normProjG=norm(ProjG)
% toc
% tic
normProjG=norm(temp);
% toc
% tic
% normProj1=norm(Proj1)
% toc
% tic
UtU=U12'*U12;
GGt=double(tenmat(core,1))*double(tenmat(core,1))';
normProj1=sqrt(GGt(:)'*UtU(:));
% toc
% tic
% normProj2=norm(Proj2)
% toc
% tic
UtU=U22'*U22;
GGt=double(tenmat(core,2))*double(tenmat(core,2))';
normProj2=sqrt(GGt(:)'*UtU(:));
% toc
% tic
% normProj3=norm(Proj3)
% toc
% tic
UtU=U32'*U32;
GGt=double(tenmat(core,3))*double(tenmat(core,3))';
normProj3=sqrt(GGt(:)'*UtU(:));
% toc

if normProjG>=max([normProj1 normProj2 normProj3])
    info=0;
    g=struct('Y_tilde', temp, 'U1_tilde', mergeU1,...
        'U2_tilde', mergeU2,...
        'U3_tilde', mergeU3);
elseif normProj1>=max([normProj2 normProj3])
    info=1;
    g=struct('Y_tilde', core, 'U1_tilde', U12,...
        'U2_tilde', U2,...
        'U3_tilde', U3);
elseif normProj2>=normProj3
    info=2;
    g=struct('Y_tilde', core, 'U1_tilde', U1,...
        'U2_tilde', U22,...
        'U3_tilde', U3);
else
    info=3;
    g=struct('Y_tilde', core, 'U1_tilde', U1,...
        'U2_tilde', U2,...
        'U3_tilde', U32);
end






