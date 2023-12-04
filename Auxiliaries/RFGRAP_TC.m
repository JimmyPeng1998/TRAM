function stats = RFGRAP_TC ( A_Omega, X, A_Gamma, r, opts )
% RFGRAP_TC retraction-free gradient-related approximate projection method
% for tensor completion 
% stats = RFGRAP_TC ( A_Omega, X, A_Gamma, r, opts )
% Input:
%   A_Omega: training set (sptensor)
%   X: initial guess (ttensor)
%   A_Gamma: test set to evaluate the recovery performance (sptensor)
%   r: rank parameter
%   Opts: user-defined options
%       - maxiter: maximum iteration (1000)
%       - verbose: verbosity (1)
%       - tol: tolerance of training error (1e-6)
%       - difftol: relative difference of training error (1e-8)
%       - basis: generation of Uk1 ('random')
%       - lastit: the last iterate (false)
%
% Output: 
%   stats.errorOmega: training error ||P_Omega X-P_Omega A|| / ||P_Omega A||
%   stats.errorGamma: test error ||P_Gamma X-P_Gamma A|| / ||P_Gamma A||
%   stats.duration: time elapsed (second)
%   stats.conv: convergence (true or false)
%   stats.X_RFGRAP: the last iterate
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Dec. 04, 2023.


if ~isfield( opts, 'maxiter');  opts.maxiter = 1000;      end
if ~isfield( opts, 'verbose');  opts.verbose = 1;         end
if ~isfield( opts, 'tol');      opts.tol = 1e-6;          end
if ~isfield( opts, 'difftol');  opts.difftol = 1e-8;      end
if ~isfield( opts, 'basis');    opts.basis = 'random';    end
if ~isfield( opts, 'lastit');   opts.lastit = false;      end

conv=false;
fprintf('RFGRAP_TC starts.\n')
%% Initial errors
tic

PAOmega=norm(A_Omega);
PAGamma=norm(A_Gamma);
errorOmega(1)=sqrt(2*calcFunction(A_Omega,X))/PAOmega;
errorGamma(1)=sqrt(2*calcFunction(A_Gamma,X))/PAGamma;

duration(1)=toc;

if opts.verbose == 1
    fprintf("Iter 0: training error %.4e, test error %.4e\n",errorOmega(1),errorGamma(1))
end




%% Line search
for t=1:opts.maxiter
    tic
    underliner=size(X.core);
    % Euclidean gradients
    eucGrad=calcGradient(A_Omega,X);
    negaEucGrad=-eucGrad;
    normEucGrad=norm(eucGrad);
    
    % Approximate projection
    [tildeU1,tildeU2,tildeU3,temp1,temp2,temp3]=computeNormalBasis(X,r,negaEucGrad,opts.basis);
    [g,info(t+1)]=computePartialProj(X,r,negaEucGrad,tildeU1,tildeU2,tildeU3,temp1,temp2,temp3);
    g_tc=ttensor(g.Y_tilde,{g.U1_tilde,g.U2_tilde,g.U3_tilde});
    
    
    
    s=calcInitial_Variety(eucGrad,g,A_Omega);
    Xnew=g_tc;
    switch info(t+1)
        case 0
            Xnew.core=s*Xnew.core;
            if underliner(1)==1 || underliner(2)==1 || underliner(3)==1
                Xnew.core(1:underliner(1),1:underliner(2),1:underliner(3))=...
                    tensor(double(Xnew.core(1:underliner(1),1:underliner(2),1:underliner(3))),underliner)+X.core;
            else
                Xnew.core(1:underliner(1),1:underliner(2),1:underliner(3))=...
                    Xnew.core(1:underliner(1),1:underliner(2),1:underliner(3))+X.core;
            end
        case 1
            Xnew.U{1}=X.U{1}+s*Xnew.U{1};
            [Q1,R1]=qr(Xnew.U{1},0);
            Xnew.U{1}=Q1;
            Xnew.core=ttm(Xnew.core, R1, 1);
        case 2
            Xnew.U{2}=X.U{2}+s*Xnew.U{2};
            [Q2,R2]=qr(Xnew.U{2},0);
            Xnew.U{2}=Q2;
            Xnew.core=ttm(Xnew.core, R2, 2);
        case 3
            Xnew.U{3}=X.U{3}+s*Xnew.U{3};
            [Q3,R3]=qr(Xnew.U{3},0);
            Xnew.U{3}=Q3;
            Xnew.core=ttm(Xnew.core, R3, 3);
    end
    
    
    duration(t+1)=toc;
    
    
    % Evaluate training and test errors
    errorOmega(t+1)=sqrt(2*calcFunction(A_Omega,Xnew))/PAOmega;
    errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
    
    
    
    if opts.verbose == 1
        fprintf("Iter %d: training error %.4e, test error %.4e, Projection: %d\n",...
            t,errorOmega(t+1),errorGamma(t+1),info(t+1))
    end
    
    
    
    
    if abs(errorOmega(t+1)-errorOmega(t))/errorOmega(t)  < opts.difftol
        fprintf('RFGRAP stagnates after %d steps.\n',t)
        break
    end
    
    if errorOmega(t+1)  < opts.tol
        fprintf('RFGRAP converges after %d steps.\n',t)
        conv=true;
        break
    end
    
    
    
    X=Xnew;
    
end



stats = struct('errorOmega',errorOmega,'errorGamma',errorGamma,'duration',cumsum(duration),...
    'conv',conv);

if opts.lastit==true
    stats.X_RFGRAP=Xnew;
end

