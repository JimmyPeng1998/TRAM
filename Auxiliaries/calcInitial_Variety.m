function s=calcInitial_Variety(eucGrad,g,A_Omega)


g_tc=ttensor(g.Y_tilde,{g.U1_tilde,g.U2_tilde,g.U3_tilde});

vals=getValsAtIndex(g_tc,A_Omega.subs);

s=-(vals'*eucGrad.vals)/(vals'*vals);


