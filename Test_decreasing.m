clear
clc

%% Creating problem
rank=3;
n = [100 100 100];
r1 = rank;
r2 = rank;
r3 = rank;
r = [r1, r2, r3];
workingr=r+[1 1 1]; % workingr>r
p=0.05;
A = makeRandnTensor( n, r );

% create the sampling set
subs = makeOmegaSet( n, round(p*prod(n)) );
vals = getValsAtIndex(A, subs);
A_Omega = sptensor( subs, vals, n, 0);
normPAOmega=norm(A_Omega);

% create the test set to compare:
subs_Test = makeOmegaSet( n, round(p*prod(n)) );
vals_Test = getValsAtIndex(A, subs_Test);
A_Gamma = sptensor( subs_Test, vals_Test, n, 0);
normPAGamma=norm(A_Gamma);


%% Initial setting
r_init = workingr;
X_init = makeRandnTensor( n, r_init );
maxIter=100;


%% Run GRAP_TC & RFGRAP_TC & TRAM
opts = struct( 'maxiter', maxIter,'gradtol', eps, 'tol', 1e-12,...
    'basis', 'random', 'verbose', 1, 'lastit', false );
stats_GRAP = GRAP_TC(A_Omega, X_init, A_Gamma, workingr, opts);

opts = struct( 'maxiter', maxIter,'gradtol',eps, 'tol', 1e-12,...
    'basis', 'random', 'verbose', 1, 'lastit', false );
stats_RFGRAP = RFGRAP_TC(A_Omega, X_init, A_Gamma, workingr, opts);

opts = struct( 'maxiter', maxIter, 'maxiterRGD', 5,...
    'tol', 1e-12, 'gradtol', eps, 'verbose', 1, 'lastit', false);
stats_TRAM = TRAM_TC( A_Omega, X_init, A_Gamma, workingr, opts);


%% Results
figure()
lwidth=6;
semilogy(stats_GRAP.duration,stats_GRAP.errorGamma,'LineWidth',lwidth)
hold on
semilogy(stats_RFGRAP.duration,stats_RFGRAP.errorGamma,'LineWidth',lwidth)
semilogy(stats_TRAM.duration,stats_TRAM.errorGamma,'LineWidth',lwidth)
legend('GRAP', 'RFGRAP', 'TRAM')
title('Test on rank decreasing')
xlabel('Time (s)')
ylabel('Test error')
set(gca,'fontsize',20)