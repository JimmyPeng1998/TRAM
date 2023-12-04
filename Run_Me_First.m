clear
clc


%% Compiling mex
fprintf("Compiling mex files.\n")
addpath tensor_toolbox-v3.4/
cd Auxiliaries/
addpath(pwd)
mex calcFunction_mex.c
mex calcGradient_mex.c
mex calcInitial_mex.c
mex calcProjection_mex.c
mex computeAk1d.c
mex computeAllProjection.c
mex computeAneqk.c
mex getValsAtIndex_mex.c


fprintf("Successfully installed!\n")
cd ..