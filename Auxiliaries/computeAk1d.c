#include "mex.h"

/*=================================================================
% function temp=computeAk1d(subs,vals,U1',U2',U3')
% This function computes A\times_{k=1}^3 Uk
% subs: 3-by-|Omega| uint32; vals: |Omega|-by-1 double, Uk: nk-by-rk double
%
% Original author: Renfeng Peng, Oct 27th, 2023.
 *=================================================================*/


/* GLOBAL VARIABLE DEFINITION FOR THE NUMBER OF DIMENSIONS */
const mwSize d = 3;


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double* eucGrad;
    double* U1;
    double* U2;
    double* U3;

    double* Aproj;              /* output tensor */

    uint32_T* index;
    mwSize sizeOmega;
    const mwSize* dims;
    mwSize r1, r2, r3;
    mwSize n1, n2, n3;

    /* GET THE INDEX ARRAY */
    /* ------------------- */

    index = (uint32_T*) mxGetPr( prhs[0] );
    sizeOmega = mxGetN( prhs[0] );

    if ( mxGetM(prhs[0]) != d )
        mexErrMsgIdAndTxt( "arrayProduct:Dimensions",
                           "Error in first input. This function currently only works for 3D tensors.");


    /* GET THE VALUES OF EUCLID. DERIV*/
    /* ------------------- */

    eucGrad = mxGetPr( prhs[1] );
    if ( mxGetM( prhs[1] ) != sizeOmega ||
         mxGetN( prhs[1] ) != 1 )
        mexErrMsgIdAndTxt( "arrayProduct:Dimensions",
                           "Error in second input. Must be a (number of nonzeros) column vector.");

    /* GET THE U FACTORS       */
    /* ----------------------- */

    U1 = mxGetPr( prhs[2] );
    dims = mxGetDimensions( prhs[2] );
    r1 = dims[0];
    n1 = dims[1];

    U2 = mxGetPr( prhs[3] );
    dims = mxGetDimensions( prhs[3] );
    r2 = dims[0];
    n2 = dims[1];

    U3 = mxGetPr( prhs[4] );
    dims = mxGetDimensions( prhs[4] );
    r3 = dims[0];
    n3 = dims[1];

    /* create the output vector (vectorized result tensor) */
    plhs[0] = mxCreateDoubleMatrix( r1*r2*r3, 1, mxREAL );

    /* get a pointer to the real data in the output matrix */
//    result = mxGetPr( plhs[0] );
//    result1 = mxGetPr( plhs[1] );
//    result2 = mxGetPr( plhs[2] );
//    result3 = mxGetPr( plhs[3] );
    Aproj = mxGetPr( plhs[0] );

    /* Compute Aneqk for k=1,2,3 */

    mwIndex p, q, r, ind;
    mwIndex i, j, k;


    for(ind=0; ind < sizeOmega; ++ind )
    {
        // get the indices
        i = index[ d*ind ] - 1;
        j = index[ d*ind + 1] - 1;
        k = index[ d*ind + 2] - 1;



        for(p=0; p<r1; p++)
            for(q=0; q<r2; q++)
                for(r=0; r<r3; r++)
                    Aproj[p+r1*(q+r2*r)]+=eucGrad[ind]*U1[p+i*r1]*U2[q+j*r2]*U3[r+k*r3];


    }
    
    

    
    
}
