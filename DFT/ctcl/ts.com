%rwf=ts.rwf
%nosave
%oldchk=../vac/ts.chk
%chk=ts.chk
%nprocshared=20
%mem=50GB
#p opt=(calcfc,ts,tight,noeigentest) freq wb97xd gen scrf=(solvent=CarbonTetraChloride) Geom=AllCheck Guess=Read

C O N H 0
6-311G(d,p)
****
I 0
@../iodine.gbs
****

