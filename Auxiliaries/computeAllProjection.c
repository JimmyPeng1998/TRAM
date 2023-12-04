#include "mex.h"

/*=================================================================
% function [temp,temp1,temp2,temp3]=computeAllProjection(subs,vals,U1',U2',U3')
% This function computes A_{\neq k}=(A\times_{j\neq k}Uk')_(k)
% subs: 3-by-|Omega| uint32; vals: |Omega|-by-1 double, Uk: nk-by-rk double
%
% Original author: Renfeng Peng, Oct 27th, 2023.
 *=================================================================*/

/* HELPER FUNCTIONS FOR LINEAR ARRAY ACCESS */
#define map2(i,j) i+n*j
#define map3(i,j,k) i+r1*j+k12*k

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
    double* mergeU1;
    double* mergeU2;
    double* mergeU3;


    double* Aproj;              /* output tensor */
    double* Aneq1;              /* output tensor */
    double* Aneq2;              /* output tensor */
    double* Aneq3;              /* output tensor */

    uint32_T* index;
    mwSize sizeOmega;
    const mwSize* dims;
    mwSize r1, r2, r3;
    mwSize n1, n2, n3;
    mwSize mr1, mr2, mr3;

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
    
    
    mergeU1 = mxGetPr( prhs[5] );
    dims = mxGetDimensions( prhs[5] );
    mr1 = dims[0];
    
    mergeU2 = mxGetPr( prhs[6] );
    dims = mxGetDimensions( prhs[6] );
    mr2 = dims[0];
    
    mergeU3 = mxGetPr( prhs[7] );
    dims = mxGetDimensions( prhs[7] );
    mr3 = dims[0];
    

    /* create the output vector (vectorized result tensor) */
    plhs[0] = mxCreateDoubleMatrix( mr1*mr2*mr3, 1, mxREAL );
    plhs[1] = mxCreateDoubleMatrix( n1*r2*r3, 1, mxREAL );
    plhs[2] = mxCreateDoubleMatrix( n2*r1*r3, 1, mxREAL );
    plhs[3] = mxCreateDoubleMatrix( n3*r1*r2, 1, mxREAL );

    /* get a pointer to the real data in the output matrix */
//    result = mxGetPr( plhs[0] );
//    result1 = mxGetPr( plhs[1] );
//    result2 = mxGetPr( plhs[2] );
//    result3 = mxGetPr( plhs[3] );

    Aproj = mxGetPr( plhs[0] );
    Aneq1 = mxGetPr( plhs[1] );
    Aneq2 = mxGetPr( plhs[2] );
    Aneq3 = mxGetPr( plhs[3] );

    /* Compute Aneqk for k=1,2,3 */

    mwIndex p, q, r, ind;
    mwIndex i, j, k;

    mwIndex k12;


    for(ind=0; ind < sizeOmega; ++ind )
    {
        // get the indices
        i = index[ d*ind ] - 1;
        j = index[ d*ind + 1] - 1;
        k = index[ d*ind + 2] - 1;



        for(q=0; q<r2; q++)
            for(r=0; r<r3; r++)
                Aneq1[i+n1*(q+r2*r)]+=eucGrad[ind]*U2[q+j*r2]*U3[r+k*r3];


        for(p=0; p<mr1; p++)
        {
            for(r=0; r<r3; r++)
                if(p<r1)
                    Aneq2[j+n2*(p+r1*r)]+=eucGrad[ind]*U1[p+i*r1]*U3[r+k*r3];


            for(q=0; q<mr2; q++)
            {
                if(q<r2 && p<r1)
                    Aneq3[k+n3*(p+r1*q)]+=eucGrad[ind]*U1[p+i*r1]*U2[q+j*r2];
                for(r=0; r<mr3; r++)
                    Aproj[p+mr1*(q+mr2*r)]+=eucGrad[ind]*mergeU1[p+i*mr1]*mergeU2[q+j*mr2]*mergeU3[r+k*mr3];
            }
        }


    }
    
    

    
    
}
