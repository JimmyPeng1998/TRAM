function stats = TRAM_TC ( A_Omega, X, A_Gamma, r, opts)
% TRAM_TC gradient-related approximate projection method for tensor
% completion 
% stats = TRAM_TC ( A_Omega, X, A_Gamma, r, opts )
% Input:
%   A_Omega: training set (sptensor)
%   X: initial guess (ttensor)
%   A_Gamma: test set to evaluate the recovery performance (sptensor)
%   r: rank parameter
%   Opts: user-defined options
%       - maxiter: maximum iteration (1000)
%       - maxiterRGD: maximum iteration of RGD (5)
%       - verbose: verbosity (1)
%       - tol: tolerance of training error (1e-6)
%       - Delta: tolerance of sigular values (0.1)
%       - gradtol: tolerance of Riemannian gradient (eps)
%       - normaltol: norm of normal vector / norm of Riemannian gradient
%       norm (0.1)
%       - difftol: relative difference of training error (1e-8)
%       - basis: generation of Uk1 ('random')
%       - lastit: the last iterate (false)
%       - iterseq: history of iterates (false)
%
% Output: 
%   stats.errorOmega: training error ||P_Omega X-P_Omega A|| / ||P_Omega A||
%   stats.errorGamma: test error ||P_Gamma X-P_Gamma A|| / ||P_Gamma A||
%   stats.duration: time elapsed (second)
%   stats.conv: convergence (true or false)
%   stats.X_GRAP: the last iterate
%
% Reference: Low-rank optimization on Tucker tensor varieties,
%    Bin Gao, Renfeng Peng, Ya-xiang Yuan, https://arxiv.org/abs/2311.18324
%
% Original author: Renfeng Peng, Dec. 04, 2023.


%% Set parameters
if ~isfield( opts, 'maxiter');      opts.maxiter = 1000;     end
if ~isfield( opts, 'maxiterRGD');   opts.maxiterRGD = 5;     end
if ~isfield( opts, 'verbose');      opts.verbose = 1;        end
if ~isfield( opts, 'tol');          opts.tol = 1e-9;         end
if ~isfield( opts, 'Delta');        opts.Delta = 0.1;        end
if ~isfield( opts, 'gradtol');      opts.gradtol = eps;      end
if ~isfield( opts, 'normaltol');    opts.normaltol = 0.1;    end
if ~isfield( opts, 'difftol');      opts.difftol = 1e-8;     end
if ~isfield( opts, 'basis');        opts.basis = 'random';   end
if ~isfield( opts, 'lastit');       opts.lastit = false;     end
if ~isfield( opts, 'iterseq');      opts.iterseq = false;    end


gradtol_init=0.1;
gradtol=gradtol_init;

fprintf('TRAM_TC starts.\n')

%% Initial errors
tic

PAOmega=norm(A_Omega);
PAGamma=norm(A_Gamma);
func=calcFunction(A_Omega,X);
errorOmega(1)=sqrt(2*func)/PAOmega;
errorGamma(1)=sqrt(2*calcFunction(A_Gamma,X))/PAGamma;

duration(1)=toc;
rank(1,:)=size(X.core);
iteratesX{1}=X;


if opts.verbose == 1
    fprintf("Iter 0: training error %.4e, test error %.4e\n",errorOmega(1),errorGamma(1))
end



%% TRAM
flag=false; % to avoid redundant computation of gradient

t=1;
while t<=opts.maxiter
        conv=false;
        
        for iter=1:opts.maxiterRGD
            tic
            if flag==false
                eucGrad=calcGradient(A_Omega,X);
                xi = calcProjection( X, eucGrad );
                eta = uminusFactorized(xi);
                normgradient = sqrt(innerProduct( X, xi, xi ));
            else
                flag=false;
            end
            
            if normgradient < gradtol && iter>1 
                duration(t)=duration(t)+toc;
                conv=true;
                break
            end
            
            s=calcInitial( eucGrad, X, eta );
            Xnew = retraction( X, eta, s );
            duration(t+1)=toc;
            
            
            % Evaluate training and test errors
            func=calcFunction(A_Omega,Xnew);
            errorOmega(t+1)=sqrt(2*func)/PAOmega;
            errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
            rank(t+1,:)=size(Xnew.core);
            iteratesX{t+1}=X;
            t=t+1;
            
            
            
            if opts.verbose ==1
                fprintf("Iter %d: training error %.4e, test error %.4e\n",...
                    t-1,errorOmega(t),errorGamma(t))
            end
            
            
            
            Xold=X;
            etaold=eta;
            sold=s;
            X=Xnew;
        end
        
        tildeX=Xnew;
    
    
    
    
    if abs(errorOmega(t)-errorOmega(t-1))/errorOmega(t-1)  < opts.difftol
        fprintf('TRAM_TC stagnates after %d steps.\n',t)
        break
    end
    
    if errorOmega(t)  < opts.tol
        fprintf('TRAM_TC converges after %d steps with errorOmega<%.4e.\n',t,opts.tol)
        break
    end
    
    
    
    if conv==true % Rank increasing?
        tic
        stats_RI=rankIncreasing(tildeX,eucGrad,normgradient,r,opts.normaltol,opts.basis);
        
        if stats_RI.canIncrease==false && gradtol<opts.gradtol
            fprintf("TRAM_TC converges after %d steps with gradnorm<%.4e.\n", t, opts.gradtol)
            duration(t)=duration(t)+toc;
            break
        elseif stats_RI.canIncrease==true
            Xnew=stats_RI.Xnew;
            newr=size(Xnew.core);
            fprintf("Rank increasing to r=[%d %d %d].\n",newr(1),newr(2),newr(3))
            gradtol=gradtol_init;
            flag=false;
            
            func=calcFunction(A_Omega,Xnew);
            duration(t+1)=toc;
            errorOmega(t+1)=sqrt(2*func)/PAOmega;
            errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
            rank(t+1,:)=size(Xnew.core);
            iteratesX{t+1}=X;
            t=t+1;
            
            
            if opts.verbose==1
                fprintf("Iter %d: training error %.4e, test error %.4e\n",...
                    t-1,errorOmega(t),errorGamma(t))
            end
        else
            while normgradient < gradtol
                gradtol=0.5*gradtol;
            end
            Xnew=tildeX;
            flag=true;
            duration(t)=duration(t)+toc;
        end
    else % Rank decreasing
        tic
        stats_RD=rankDecreasing(tildeX,opts.Delta);
        
        if stats_RD.canDecrease==true
            flag=false;
            H0=hosvd(tildeX.core,[stats_RD.r1 stats_RD.r2 stats_RD.r3]);
            Xnew=ttensor(H0.core,{tildeX.U{1}*H0.U{1},tildeX.U{2}*H0.U{2},tildeX.U{3}*H0.U{3}});
            fprintf("Rank decreasing to r=[%d %d %d].\n",stats_RD.r1,stats_RD.r2,stats_RD.r3)
            func=calcFunction(A_Omega,Xnew);
            duration(t+1)=toc;
            errorOmega(t+1)=sqrt(2*func)/PAOmega;
            errorGamma(t+1)=sqrt(2*calcFunction(A_Gamma,Xnew))/PAGamma;
            rank(t+1,:)=size(Xnew.core);
            iteratesX{t+1}=X;
            t=t+1;
            gradtol=gradtol_init;
            
            if opts.verbose==1
                fprintf("Iter %d: training error %.4e, test error %.4e\n",...
                    t-1,errorOmega(t),errorGamma(t))
            end
            
        else
            Xnew=tildeX;
            flag=false;
            duration(t)=duration(t)+toc;
        end
    end
    
    
    X=Xnew;
end


stats = struct('errorOmega',errorOmega,...
    'errorGamma',errorGamma,...
    'duration',cumsum(duration),...
    'rank',rank);

if opts.lastit==true
    stats.X_TRAM=Xnew;
end

if opts.iterseq==true
    stats.iteratesX=iteratesX;
end
