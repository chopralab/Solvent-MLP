%rwf=irc.rwf
%oldchk=ts.chk
%chk=irc.chk
%nprocshared=20
%mem=50GB
#p irc=(calcfc,MaxCycle=100,MaxPoints=50,ReCalcFC=(Predictor=10,Corrector=5))
scrf=(solvent=o-Xylene)
wb97xd gen 
Geom=AllCheck Guess=Read

C O N H 0
6-311G(d,p)
****
I 0
@../iodine.gbs
****

