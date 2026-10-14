"""Independent symbolic checks for public service-productivity heterogeneity.

Run from any directory. Prints checks and illustrative branch candidates only.
No claim of calibration, global equilibrium selection, or novelty is made.
"""
import sympy as s

a,b,k,r,e,c1,c2,C,S,K,R,alpha=s.symbols('a b k r e c1 c2 C S K R alpha', positive=True)
x=s.symbols('x', real=True)
r1,r2=r+e,r
rb=(r1+r2)/2
u=b*e**2
F=2*k+u
g=b*e*(r1+r2)
D=e*a+c2-c1
H=4*k+u
beta1=k+b*r1*e
beta2=k-b*r2*e
A1=r1*(a-b*r2*S)-c1-C
A2=r2*(a-b*r1*S)-c2-C
X=r2*S+e*x
W=a*X-b*X**2/2-c1*x-c2*(S-x)-k*(x**2+(S-x)**2)-C*S
xW=S/2+(D-g*S/2)/H
xM=S/2+D/(4*F)
xT=S/2+(D-g*S)/(2*F)
yM=(A1+A2)/2-F*S
yT=A1-2*beta1*xT
WM=s.expand(W.subs(x,xM))
WT=s.expand(W.subs(x,xT))
diff=(2*g*S-D)*((2*F+3*u)*D-2*u*g*S)/(32*F**2)
checks={
 'efficient_foc':s.diff(W,x).subs(x,xW),
 'efficient_curvature':s.diff(W,x,2)+H,
 'price_agreement_M':yM-(A1-2*F*xM),
 'price_agreement_T':yT-(A2-2*beta2*(S-xT)),
 'welfare_rank_factorization':WM-WT-diff,
 'welfare_rank_derivative':s.diff(WM-WT,S)-g*(D*(F+2*u)-2*u*g*S)/(8*F**2),
 'homogeneous_loss':(WM-WT).subs(e,0)+(c2-c1)**2/(32*k),
}
L=a*rb-(c1+c2)/2-C
J=b*r1*r2+F
dM=D/(4*F)
wM1=L-b*rb*e*dM
wM2=b*rb**2+k
constM=D*dM-H*dM**2/2
checks['WM_full_release']=WM.subs(S,K)-(constM+wM1*K-wM2*K**2/2)
checks['M_revenue']=yM*S-(L*S-J*S**2)
KM=(wM1+(alpha-1)*L)/(R+wM2+2*(alpha-1)*J)
checks['M_investment_foc']=s.diff(WM.subs(S,K)+(alpha-1)*yM.subs(S,K)*K-R*K**2/2,K).subs(K,KM)

# T is non-manipulative bidding along residual-allocation values, not IC or FB.
lT=beta2/F
dT=D/(2*F)
rhoT=r2+e*lT
cT=c1*lT+c2*(1-lT)
wT1=a*rhoT-cT-C-b*rhoT*e*dT-2*k*dT*(2*lT-1)
wT2=b*rhoT**2+2*k*(lT**2+(1-lT)**2)
yT0=L-g*D/(2*F)
yT1=b*r1*r2+2*beta1*beta2/F
KT=(wT1+(alpha-1)*yT0)/(R+wT2+2*(alpha-1)*yT1)
checks['T_price']=yT-(yT0-yT1*S)
checks['T_investment_foc']=s.diff(WT.subs(S,K)+(alpha-1)*yT.subs(S,K)*K-R*K**2/2,K).subs(K,KT)
checks['alpha1_capacity_sign_numerator']=(wM1-wT1)-(wM2-wT2)*KT.subs(alpha,1)-g*(D*(F+2*u)-2*u*g*KT.subs(alpha,1))/(8*F**2)
checks['homogeneous_alpha1_capacity']=(KM-KT).subs({alpha:1,e:0})
checks['constant_cost_interior_price']=(a-b*X).subs(x,xW).subs(k,0)-(c1-c2)/e
# Main extension: congestion is COMMON PER RESOURCE, hence total cost C*S.
# Unequal flight resource weights are already absorbed into rho_i=d_i/h_i,
# c_i=t_i/h_i, and k_i=kappa_i/h_i**2. No per-flight C*x_i/h_i is used.
h1,h2,d1,d2,t1,t2,kap1,kap2,k1,k2=s.symbols('h1 h2 d1 d2 t1 t2 kap1 kap2 k1 k2', positive=True)
Wgeneral=a*X-b*X**2/2-c1*x-c2*(S-x)-k1*x**2-k2*(S-x)**2-C*S
xWgeneral=(e*(a-b*r2*S)+c2-c1+2*k2*S)/(b*e**2+2*(k1+k2))
checks['general_efficient_foc']=s.diff(Wgeneral,x).subs(x,xWgeneral)
checks['general_to_special_case']=xWgeneral.subs({k1:k,k2:k})-xW
checks['flight_to_resource_cost_mapping']=(t1*(x/h1)+kap1*(x/h1)**2)-(t1/h1*x+kap1/h1**2*x**2)
checks['flight_to_service_mapping']=d1*(x/h1)+d2*((S-x)/h2)-(d1/h1*x+d2/h2*(S-x))
B,gamma=s.symbols('B gamma', nonnegative=True)
Cof=B*S/K+gamma*S**2/(2*K**2)
AM=alpha*(a*rb-(c1+c2)/2)-b*rb*e*D/(4*F)
DM=wM2+2*(alpha-1)*J
FM=(WM+(alpha-1)*yM*S).subs(C,Cof)
checks['state_objective']=FM-(constM+AM*S-DM*S**2/2-alpha*B*S**2/K-alpha*gamma*S**3/(2*K**2))
checks['state_capacity_derivative']=s.diff(FM,K)-alpha*(B*S**2/K**2+gamma*S**3/K**3)
for name, expression in checks.items():
    assert s.factor(expression)==0, (name,s.factor(expression))
print(f'{len(checks)} symbolic checks passed')

params={a:10,b:1,k:s.Rational(1,5),r:1,e:s.Rational(1,10),c1:2,c2:2,C:s.Rational(1,10),S:3,R:s.Rational(3,2)}
def num(v,extra=None):
    return float(v.subs(params | (extra or {})))
def report_at_release(release):
    sub={S:release}
    im=num(xM,sub); it=num(xT,sub); iw=num(xW,sub)
    b1end=A1-2*beta2*xM-2*beta1*S
    b2end=A2-2*beta1*(S-xM)-2*beta2*S
    return dict(S=float(release),xM=im,xT=it,xW=iw,yM=num(yM,sub),yT=num(yT,sub),
                XM=num(X.subs(x,xM),sub),XT=num(X.subs(x,xT),sub),XW=num(X.subs(x,xW),sub),
                WM=num(WM,sub),WT=num(WT,sub),WW=num(W.subs(x,xW),sub),gain=num(diff,sub),
                bidder1_end_bid=num(b1end,sub),bidder2_end_bid=num(b2end,sub),
                MV1_end=num(A1-2*beta1*S,sub),MV2_end=num(A2-2*beta2*S,sub))
print('fixed_release',report_at_release(s.Integer(3)))
for av in (1,2):
    km=s.cancel(KM.subs(params | {alpha:av}))
    kt=s.cancel(KT.subs(params | {alpha:av}))
    print('investment',dict(alpha=av,KM=float(km),KT=float(kt),
          at_M=report_at_release(km),at_T=report_at_release(kt)))
