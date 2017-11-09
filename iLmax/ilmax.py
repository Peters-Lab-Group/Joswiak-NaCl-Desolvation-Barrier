###############################################################################
## DESCRIPTION
###############################################################################

# Code by Ryan Mullen. Slightly modified by Mark Joswiak.
## Modifications ##
# The acceleration terms are altered so that, if the acceleration option is included
# then each qi has it's own associated coefficient cai
##########################

# Script to maximize the likelihood for trial coordinates, q, that are linear
# combinations of order parameters, qi, with acceleration parameters for EACH 
# coordinate (if the acceleration option is selected
#
#  q = Sum( ci * qi + cai * (ci * qdotdoti) ) with i = 1 ... M
#
# The reaction coordinate model includes q & the force on the trial coordinate,
# which is proportional to q's acceleration, a. To optimize the reaction 
# coordinate also with respect to the transmission coeffecient, we include its 
# velocity
#
#  r = c0 + Sum( ci * qi + cai * (ci * qdotdoti) ) 
#		  + cv * Sum( ci * qdoti + cai * ci * qdotdotdoti)
#
# with j being the jerk (qdotdotdot), the third derivative of q with respect to time. c0, 
# cv are additional constants optimized to maximize the likelihood. All
# constants are combined in the array Coeff
#
#  Coeff = [c0 c1 c2 ... cM-1 cM ca1 ca2 ... caM-1 caM cv]

###############################################################################
## IMPORT MODULES
###############################################################################

import numpy as np
import sys
import os
import logging
import argparse
import math

#logging.basicConfig(stream=sys.stderr,
#logging.basicConfig(filename='info.log', 
#										format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig()

###############################################################################
## ARGUMENT LIST
###############################################################################
#Parse command-line options
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
description=
"""DESCRIPTION:
This script performs either original likelihood maximization (LMax) or inertial
likelihood maximization (iLMax) depending on the supplied inputs.

ORIGINAL LMAX
LMax requires DMAX, IFILE (default=q.txt) and RESULTSFILE (default=results.dat).

If DMAX is greater than 1, then linear combinations of upto DMAX order 
parameters are also evaluted. For example, with DMAX=3, the following trial 
coordinates will be evaluated

    D=1: c0 + c1 * qi                      with i = 1 ... M
    D=2: c0 + c1 * qi + c2 * qj            with i = 1 ... M-1, j = i+1 ... M
    D=3: c0 + c1 * qi + c2 * qj + c3 * qk  with i = 1 ... M-2, j = i+1 ... M-1,
                                                k = j+1 ... M

INERTIAL LMAX
iLMax requires DMAX, IFILE, RESULTSFILE and VELFILE (default=dt.txt).

The reaction coordinate model can optionally include the velocity v of the 
trial coordinate q in order to optimize the reaction coordinate also with 
respect to the transmission coeffecient. Trial coordinates are of the form

    r = c0 + q + cv * v

with q = Sum(ci * qi) for i = 1 ... D and v is the first time derivative of q.

ACCELERATION
The reaction coordinate model can optionally include the acceleration a of the
trial coordinate q. The acceleration is proportional to the force on q. Trial
coordinates have the form

    r  = c0 + Sum( ci * qi + cai * (ci * qdotdoti) ) 
		  + cv * Sum( ci * qdoti + cai * ci * qdotdotdoti)

with i = 1 ... D and j is the jerk (qdotdotdot), the third derivative of
q with respect to time. 

For example,
    python ilmax.py -d 2              LMax for all qi and pairs qi,qj in q.txt
    python ilmax.py -d 1 -i 0.dat -a  LMax for all qi in 0.dat,
                                      a from 0.forward.dat & 0.back.dat
    python ilmax.py -d 1 -v           iLMax for all qi in q.txt, v from dt.txt
    python ilmax.py -d 1 -v -a        iLMax for all qi in q.txt, v from dt.txt, 
                                      a from q.forward.txt & q.back.txt
                                      j from dt.forward.txt & dt.back.txt

Each of these examples requires results.dat.
""")
parser.add_argument('--Dmax', '-d', type=int, required=True,
								help="""the maximum number of order parameters to include in a 
trial coordinate""")
parser.add_argument('--iFile', '-i', default='q.txt',
								help="""IFILE contains the order parameters computed for each 
shooting point. IFILE has the following format: 

#sp q1 q2 .. qM
0   .. ..    ..
1   .. ..    ..
.         .
.          .
N   .. ..    ..

where q1 through qM are the order parameters to be 
tested by LMax. This option is only required if the 
IFILE is named something other than q.txt.""")
parser.add_argument('--ResultsFile', '-r', default='results.dat',
								help="""RESULTSFILE contains the aimless shooting outcomes for 
each trajectory. RESULTSFILE has the following format:

#sp hAb hAf hBb hBf
0   .   .   .   .
1   .   .   .   .
.
.
N   .   .   .   .

where hA=1 if a configuration is in basin A and 0 
otherwise. Likewise for hB. The subscripts b and f stand
for 'back' and 'forward', respectively. For example, an 
A-->B trajectory will have the entry '1 0 0 1'. The back
outcomes are used to verify basins A and B don't overlap
and are then discarded. The forward outcomes are used in
LMax. This option is only required if RESULTSFILE is 
named something other than results.dat.""")
parser.add_argument('--VelFile', '-v', nargs='?', const='dt.txt', default='',
								help="""VELFILE contains the order parameters computed for a 
configuration a short time (e.g., +1 timestep) from the 
shooting point. VELFILE has the same format as IFILE. If
-v is provided without an argument, the default filename
is dt.txt.""")
parser.add_argument('--AccelExt', '-a', nargs='?', 
								const=['forward', 'back'], default=[],
								help="""ACCELEXT is a python list of filename modifiers. This 
option requires two additional files for LMax and four 
additional files for iLMax. All files have the same 
format as IFILE. If -a is provided without an argument, 
the default modifiers are 'forward' and 'back'. With -a 
and IFILE=q.txt, files q.forward.txt and q.back.txt 
contain order parameters for configurations a short NVE
time forward and backward, respectively, from the
shooting point. With -a and -v, files dt.forward.txt 
and dt.back.txt are also required.""")
args = parser.parse_args(sys.argv[1:])
logging.info('Dmax=' + str(args.Dmax))
logging.info('iFile=' + args.iFile)
logging.info('VelFile=' + args.VelFile)
logging.info('AccelExt=' + str(args.AccelExt))
logging.info('ResultsFile=' + args.ResultsFile)

FileList = [args.iFile]
#Velocity
if args.VelFile: FileList.append(args.VelFile)
V = True if args.VelFile else False
#Acceleration
AccelFileList = []
A = True if args.AccelExt else False
for File in FileList:
	File = File.rsplit('.',1)
	for dir in args.AccelExt:
		AccelFileList.append(File[0] + '.' + dir + '.' + File[1])
#Test command-line options
for File in FileList + AccelFileList + [args.ResultsFile]:
	if not os.path.isfile(File):
		logging.error("Input file " + File + " does not exist.")
		sys.exit(2)

###############################################################################
## DEFINE FUNCTIONS
###############################################################################

def Comb(N, M):
	"""Calculate the number of combinations choosing M from N
	Input:
		N: integer, total number of options
		M: integer, total number of selections
	Output: 
		Result: integer, number of combinations N!/(N-M)!/M!
"""
	Result=1
	for C in range(N-M+1,N+1):
		Result *= C
	for C in range(1,M+1):
		Result /= C
 
	return Result

def LComb(NList, M):
	"""List the combinations choosing M items from NList
	Input:
		NList: list of options
		M: total number of selections
	Output:
		Result: tuple of tuples, N!/(N-M)!/M! combinations
"""
	N = len(NList)
	NCM = Comb(N, M)
	Result = [0] * NCM
	i = range(M)
	for c in range(NCM):
		Result[c] = tuple([NList[j] for j in i])

		for m1 in range(M-1,-1,-1):
			if i[m1] < N-M+m1:
				i[m1] += 1
				for m2 in range(m1+1,M):
					i[m2] = i[m2-1] + 1
				break

	return tuple(Result)

def RxnCoord(Coeff, Z):
	"""Calculate reaction coordinate realizations
	Input:
		Coeff: (1+D+A*D+V) array, coefficients for combining order parameters
		Z: (4,N,D) array, dimensionless order parameter values 
	Output:
		R: (N) array, reaction coordinate values
"""
	#identify global variables
	global A, V #boolean variables
	D = Z.shape[2]

	C0 = Coeff[0]  #Additive constant
	C  = Coeff[1:D+1] #Coeff for combining order parameters
	if A:							## correctly pick out the ca vector, depends if V is true or not
		if V:
			CA = Coeff[D+1:-1]
		else:
			CA = Coeff[D+1:]
	else:
		CA = 0.
##	CA = Coeff[D+1] if A else 0. #Coeff for combining acceleration
	CV = Coeff[-1]  if V else 0. #Coeff for combining velocity

	
	R = (C0 + (np.dot(Z[0,:],C) + np.dot(Z[2,:],C*CA)) + 
			 CV * (np.dot(Z[1,:],C) + np.dot(Z[3,:],C*CA)))
			 


##	R = (C0 + (np.dot(Z[0,:],C) + CA * np.dot(Z[2,:],C)) + 
##			 CV * (np.dot(Z[1,:],C) + CA * np.dot(Z[3,:],C)))

	return R

def Committor(R):
	"""Calculate committor realizations
	Input:
		R: (N) array, reaction coordinate values
	Output:
		pB: (N) array, committor realizations
"""
	pB = [0.5 + 0.5 * math.erf(r) for r in R]
	pB = np.array(pB)
	return pB

def LogLikelihood(pB, hB):
	"""Calculate the log likelihood
	Input:
		pB: (N) array, committor realizations
		hB: (N) array of boolean values, whether trajectory relaxed to B
	Output:
		lnL: float, log likelihood
"""
	lnL = np.sum(np.log(pB[hB])) + np.sum(np.log(1. - pB[~hB]))
	return -lnL

def Gradient(Coeff, Z, hB):
	"""Calculate the gradient of lnL wrt to Coeff
	Input:
		Coeff: (3+D) array, coefficients for combining order parameters
		Z: (4,N,D) array, dimensionless order parameter values 
		hB: (N) array of boolean values, whether trajectory relaxed to basin B
	Output:
		dCoeff: (3+D) array, gradient for coefficients
"""
	#identify global variables
	global A, V #boolean variables
	D = Z.shape[2]

	#Partial derivative of lnL wrt to pB
	R = RxnCoord(Coeff, Z)
	pB = Committor(R)
	ddpB_lnL = np.array([1./p if h else -1./(1.-p) for (h,p) in zip(hB,pB)])

	#Partial derivative of pB wrt to R
	irpi = 1 / np.sqrt(np.pi)
	ddr_pB = np.array([irpi * math.exp(-r**2) for r in R])

	#Partial derivatives of R wrt to Coeff
	C0 = Coeff[0]  #Additive constant
	C  = Coeff[1:D+1] #Coeff for combining order parameters
	if A:								## correctly pick out CA depending on if doing velocity too or not
		if V:
			CA = Coeff[D+1:-1]
		else:
			CA = Coeff[D+1:]
	else:
		CA = 0.
##	CA = Coeff[D+1] if A else 0. #Coeff for combining acceleration
	CV = Coeff[-1]  if V else 0. #Coeff for combining velocity
	ddCoeff_r = np.ones((1+D+A*D+V,N)) #ddC0_r			## increase array from previous version to get CA vector
##	ddCoeff_r = np.ones((1+D+A+V,N)) #ddC0_r
	ddCoeff_r[1:D+1] = ((z[0] + z[2] * CA) + CV * (z[1] + z[3] * CA)).T #ddC_r
##	ddCoeff_r[1:D+1] = ((z[0] + CA * z[2]) + CV * (z[1] + CA * z[3])).T #ddC_r
	logging.debug("Z.shape=" + str(Z.shape))
	logging.debug("C=" + str(C))
	(q,v,a,j) = np.dot(Z,C)
	if A:
		if V:
			ddCoeff_r[D+1:-1] = (z[2] * C + CV * (z[3] * C)).T
		else:
			ddCoeff_r[D+1:] = (z[2] * C + CV * (z[3] * C)).T
##		ddCoeff_r[D+1] = a + CV * j #ddCA_r					 

	if V:
		ddCoeff_r[-1]  = np.dot(z[1,:],C) + np.dot(z[3,:],C*CA) 
##		ddCoeff_r[-1]  = v + CA * j #ddCV_r

	#Return derivatives of lnL wrt to Coeff
	dCoeff = np.sum(ddpB_lnL * ddr_pB * ddCoeff_r, axis=1)
	return dCoeff

def LineSearch(Coeff, Z, hB, Dir, dx, FracTol,
								Accel = 1.5, MaxInc = 10., MaxIter = 10000):
	"""Performs a line search along direction Dir.
	Input:
		Coeff: (3+D) array, coefficients for reaction coordinate
		Z: (4,N,D) array, dimensionless order parameter values 
		hB: (N) array of boolean values, whether trajectory relaxed to basin B
		Dir: (3+D) array, gradient of lnL wrt coefficients
		dx: initial step amount
		FracTol: fractional lnL tolerance
		Accel: acceleration factor
		MaxInc: the maximum increase in lnL for bracketing
		MaxIter: maximum number of iteration steps
	Output:
		lnLMin: value of lnL at minimum along Dir
		CoeffMin: minimum lnL (3+D) position array along Dir
"""
	#start the iteration counter
	Iter = 0

	#find the normalized direction
	NormDir = Dir / np.sqrt(np.sum(Dir * Dir))

	#take the first two steps and compute values
	Dists = [0., dx]
	lnLs = [LogLikelihood(Committor(RxnCoord(Coeff + NormDir * x, Z)),hB) 
						for x in Dists]
	
	#if the second point is not downhill, back
	#off and take a shorter step until we find one
	while lnLs[1] > lnLs[0]:
		Iter += 1
		dx = dx * 0.5
		Dists[1] = dx
		lnLs[1] = LogLikelihood(Committor(
							RxnCoord(Coeff + NormDir * Dists[-1], Z)),hB)

	#find a third point
	Dists = Dists + [2. * dx]
	lnLs = lnLs + [LogLikelihood(Committor(
								RxnCoord(Coeff + NormDir * Dists[-1], Z)),hB)]
	
	#keep stepping forward until the third point is higher;
	#then we have bracketed a minimum
	while lnLs[2] < lnLs[1]:
		Iter += 1
	
		#find a fourth point and evaluate
		Dists = Dists + [Dists[-1] + dx]
		lnLs = lnLs + [LogLikelihood(Committor(
									RxnCoord(Coeff + NormDir * Dists[-1], Z)),hB)]

		#check if we increased too much; if so, back off
		if (lnLs[3] - lnLs[0]) > MaxInc * (lnLs[0] - lnLs[2]):
			lnLs = lnLs[:3]
			Dists = Dists[:3]
			dx = dx * 0.5
		else:
			#shift all of the points over
			lnLs = lnLs[-3:]
			Dists = Dists[-3:]
			dx = dx * Accel
	
	#we've bracketed a minimum; now we want to find it to high
	#accuracy
	OldlnL3 = 1.e300
	while True:
		Iter += 1
		if Iter > MaxIter:
			print "Warning: maximum number of iterations reached in line search."
			break
	
		#store distances for ease of code-reading
		d0, d1, d2 = Dists
		lnL0, lnL1, lnL2 = lnLs

		#use a parobolic approximation to estimate the location
		#of the minimum
		d10 = d0 - d1
		d12 = d2 - d1
		Num = d12*d12*(lnL0-lnL1) - d10*d10*(lnL2-lnL1)
		Dem = d12*(lnL0-lnL1) - d10*(lnL2-lnL1)
		if Dem == 0:
			#parabolic extrapolation won't work; set new dist = 0 
			d3 = 0
		else:
			#location of parabolic minimum
			d3 = d1 + 0.5 * Num / Dem
			
		#compute the new potential 
		lnL3 = LogLikelihood(Committor(RxnCoord(Coeff + NormDir * d3, Z)),hB) 
		
		#sometimes the parabolic approximation can fail;
		#check if d3 is out of range < d0 or > d2 or the new lnL is higher
		if d3 < d0 or d3 > d2 or lnL3 > lnL0 or lnL3 > lnL1 or lnL3 > lnL2:
			#instead, just compute the new distance by bisecting two
			#of the existing points along the line search
			if abs(d2 - d1) > abs(d0 - d1):
				d3 = 0.5 * (d2 + d1)
			else:
				d3 = 0.5 * (d0 + d1)
			lnL3 = LogLikelihood(Committor(RxnCoord(Coeff + NormDir * d3, Z)),hB) 
	
		#decide which three points to keep; we want to keep
		#the three that are closest to the minimum
		if d3 < d1:
			if lnL3 < lnL1:
				#get rid of point 2
				Dists, lnLs = [d0, d3, d1], [lnL0, lnL3, lnL1]
			else:
				#get rid of point 0
				Dists, lnLs = [d3, d1, d2], [lnL3, lnL1, lnL2]
		else:
			if lnL3 < lnL1:
				#get rid of point 0
				Dists, lnLs = [d1, d3, d2], [lnL1, lnL3, lnL2]
			else:
				#get rid of point 2
				Dists, lnLs = [d0, d1, d3], [lnL0, lnL1, lnL3]
	
		#check how much we've changed
		if abs(OldlnL3 - lnL3) < FracTol * abs(lnL3):
			#the fractional change is less than the tolerance,
			#so we are done and can exit the loop
			break
		OldlnL3 = lnL3

	#return the coeff at the minimum (point 1)
	CoeffMin = Coeff + NormDir * Dists[1]
	lnLMin = lnLs[1]

	return lnLMin, CoeffMin

	
def ConjugateGradient(Coeff, Z, hB, dx, FracTolLS, FracTolCG):
	"""Performs a conjugate gradient search.
	Input:
		Coeff: (3+D) array, coefficients for reaction coordinate
		Z: (4,N,D) array, dimensionless order parameter values 
		hB: (N) array of boolean values, whether trajectory relaxed to basin B
		dx: initial step amount
		FracTolLS: fractional tolerance for line search
		FracTolCG: fractional tolerance for conjugate gradient
	Output:
		lnL: value of lnL at minimum
		Coeff: minimum (3+D) coefficient array 
"""
	R = RxnCoord(Coeff, Z)
	pB = Committor(R)
	lnL = LogLikelihood(pB, hB)
	Grad = Gradient(Coeff, Z, hB)
	Dir = Grad
	while True:
		OldlnL = lnL
		lnL, Coeff = LineSearch(Coeff, Z, hB, Dir, dx, FracTolLS)
		logging.info("iteration " + str(lnL) + " c: " + str(Coeff))
		#check how much we've changed
		if abs(OldlnL - lnL) < FracTolCG * abs(lnL):
				#the fractional change is less than the tolerance,
				#so we are done and can exit the loop
				break
		OldGrad = Grad
		Grad = Gradient(Coeff, Z, hB)
		gamma = ((Grad - OldGrad) * Grad).sum() / np.square(OldGrad).sum()
		Dir = Grad + gamma * Dir

	return lnL, Coeff

###############################################################################
## SET VARIABLES
###############################################################################

FracTolLS = 1.e-8
FracTolCG = 1.e-10
dx = 0.001

# FracTolLS = fractional energy tolerance for line search algorithm
# FracTolCG = fractional energy tolerance for congruant gradient allgorithm
# dx = starting step size for line search algorithm

###############################################################################
## MAIN CODE
###############################################################################

dt = 2.0
dt_sq = dt*dt
print 'THIS VERSION PERFORMS OPTIMIZATION WITH CA AS A VECTOR\n'

print 'MAKE SURE THE CORRECT DT IS SET BASED ON THE NVE TIMESTEP'
print 'timestep =',dt,' fs'

#initialize data collection
lnL = {}
Coeff = {}
Type = np.array(['q', 'v', 'a', 'j']) #coordinate,velocity,acceleration,jerk

#read in Aimless Shooting results
f = open(args.ResultsFile, 'r')
data = np.loadtxt(f, skiprows=1, usecols=(1,2,3,4))
hA = np.array(data[:,0:3:2], int)
hB = np.array(data[:,1:4:2], int)
f.close()
#verify basin results
if (2 in hA[:,0] + hB[:,0]) or (2 in hA[:,1] + hB[:,1]):
	print "YOUR BASIN DEFINITIONS OVERLAP"
	exit()
accepted = int((hA[:,0] * hB[:,1] + hA[:,1] * hB[:,0]).sum())
#only keep forward results
hA = np.array(hA[:,1],bool)
hB = np.array(hB[:,1],bool)
nA = hA.sum()
nB = hB.sum()
print "accepted %d of %d" % (accepted, nA + nB)
print "n(A)=%d  n(B)=%d" % (nA, nB)

#read in coordinate data
f = open(args.iFile, "r")
QName = f.readline().split()[1:]
data = np.loadtxt(f)[:,1:]
N, M = data.shape
if N != nA + nB: 
	logging.error("iFile and ResultsFile have different numbers of lines")
	sys.exit(2)
#declare Q 
Q = np.zeros((4,N,M))
Q[0] = data.copy()
f.close()
logging.info("Read input file.")
logging.debug("N=" + str(N))
logging.debug("M=" + str(M))
#calc formatting parameters MLen & QNLen
MLen=0
while ( M >= 10 ** MLen ): 
	MLen += 1
QNLen=0
for m in range(M):
	if (len(QName[m]) > QNLen):
		QNLen = len(QName[m])

#read add'l timestep data
#acceleration
if A:
	for File in AccelFileList[:2]:
		f = open(File, 'r')
		AName = f.readline().split()[1:]
		if QName != AName:
			logging.error("iFile and " + File + " headers do not match")
			sys.exit(2)
		Q[2] += np.loadtxt(f)[:,1:]
		f.close()
	Q[2] = (Q[2] - 2 * Q[0]) / (dt_sq) 		## central finite difference to get 2nd derivative 
#velocity
if V:
	f = open(args.VelFile, 'r')
	VName = f.readline().split()[1:]
	Q[1] = np.loadtxt(f)[:,1:]				## op values 1 timestep past shooting point
	f.close()
	logging.info("Read velocity file")
	if QName != VName: 
		logging.error("iFile and VelFile headers do not match")
		sys.exit(2)
	#jerk
	if A:
		for File in AccelFileList[2:]:
			f = open(File, 'r')
			JName = f.readline().split()[1:]
			if QName != JName:
				logging.error("iFile and " + File + " headers do not match")
				sys.exit(2)
			Q[3] += np.loadtxt(f)[:,1:]		## op values 2 timesteps past shooting point
			f.close()
		Q[3] = (Q[3] - 2 * Q[1]) / (dt_sq)			#0.0625
		Q[3] = (Q[3] - Q[2]) / dt					## finite difference accelerations to get jerk
	Q[1] = (Q[1] - Q[0]) / dt

#non-dimensionalize & report Q's
Z = np.zeros_like(Q)
print "\nNON-DIMENSIONALIZE ORDER PARAMETERS"
print " %*s: %-*s %10s %10s %10s" % (MLen, "#", QNLen, "Name", 
			"Min", "Max", "Range")
Qmax = Q[0].max(axis=0)
Qmin = Q[0].min(axis=0)
Qspan = Qmax - Qmin
for m in range(M):
	print " %*d: %-*s %10.4g %10.4g %10.4g" % (MLen, m+1, 
				QNLen, QName[m], Qmin[m], Qmax[m], Qspan[m])
Z = Q.copy()
Z[0] = Q[0]-Qmin
Z = Z / Qspan
print "\nREDUCED VARIABLES: Z = (Q-Qmin)/(Qmax-Qmin)"
print "SAMPLED Z's IN [0, 1], Q = Z(Qmax-Qmin)+Qmin"
print "BIC = LN(N)/2 = %.3f" % (0.5 * np.log(N))

#loop over dimensionality of rxn coord
for D in range(1,args.Dmax+1):
	#print header
	print "\nD=%d LMAX" % D
	header = ''
	for d in range(D):
		header += " %*s" % (MLen, '#')
	header += ":"
	for d in range(D):
		header += " %-*s" % (QNLen, 'Name')
	header += " %10s" % "-lnL"
	for d in range(1+D):
		header += " %9s%d" % ('c', d)
	if A:
		for d in range(D):														### ADDED
			header += " %9s%1d" % ('ca',d+1)										### ADDED
##		header += " %10s" % 'ca'
	if V:
		header += " %10s" % 'cv'
	print header

	#loop over all combinations
	for c in LComb(range(M),D):
		mi = np.array(c, int)
		logging.info("c=" + str(c))
		z = Z[:,:,mi]
		if D == 1:
		#get initial values by random numbers
			A, V = False, False #want to optimize trial coordinate with a,v first
			pB = np.zeros(N)
			#1st guess
			while 0. in pB[hB] or 1. in pB[~hB]:
				Coeff[c] = np.random.uniform(-2., 2., 1+D)
				pB = Committor(RxnCoord(Coeff[c],z))
			lnL[c] = LogLikelihood(Committor(RxnCoord(Coeff[c], z)), hB)
			logging.info("initial " + str(lnL[c]) + " c: " + str(Coeff[c]))
			#improve on 1st guess if possible
			zavg = np.mean(z[0,~hB],axis=0)
			for i in range(16*D*D):
				TempCoeff = np.random.uniform(-2., 2., 1+D)
				TempCoeff[0] = - np.dot(zavg,TempCoeff[1:D+1])
				pB = Committor(RxnCoord(TempCoeff,z))
				if 0. in pB[hB] or 1. in pB[~hB]:
					continue
				TemplnL = LogLikelihood(Committor(RxnCoord(TempCoeff, z)), hB)
				if TemplnL < lnL[c]:
					lnL[c] = TemplnL
					Coeff[c] = TempCoeff
					logging.info("improvement " + str(lnL[c]) + " c: " + str(Coeff[c]))
			lnL[c], Coeff[c] = ConjugateGradient(Coeff[c], z, hB, dx, 
																					 1., 1.)
			A = True if args.AccelExt else False #turn a,v back on
			V = True if args.VelFile else False
			if A: Coeff[c] = np.append(Coeff[c], np.zeros(D))
##			if A: Coeff[c] = np.append(Coeff[c], np.zeros(1))
			if V: Coeff[c] = np.append(Coeff[c], np.zeros(1))
		else:
		#use best sub-model
			Coeff[c] = np.zeros(1+D+A*D+V)
##			Coeff[c] = np.zeros(1+D+A+V)
			cbest = c[:-1]
			for c1 in LComb(c, D-1):
				if lnL[c1] < lnL[cbest]:
					cbest = c1
			Coeff[c][0] = Coeff[cbest][0]
			for m in c:
				if m in cbest:
					Coeff[c][c.index(m) + 1] = Coeff[cbest][cbest.index(m) + 1]
			logging.debug("cbest: " + str(cbest))
			if A:
				for m in c:
					if m in cbest:
						Coeff[c][c.index(m) + 1 + D] = Coeff[cbest][cbest.index(m) + 1 + D - 1]
##				Coeff[c][D+1] = Coeff[cbest][D]
			if V:
				Coeff[c][-1] = Coeff[cbest][-1]
			lnL[c] = LogLikelihood(Committor(RxnCoord(Coeff[c], z)), hB)
			logging.info("initial " + str(lnL[c]) + " c: " + str(Coeff[c]))

		
		#find the minimum
		lnL[c], Coeff[c] = ConjugateGradient(Coeff[c], z, hB, dx, 
																				 FracTolLS, FracTolCG)
		logging.info("final " + str(lnL[c]) + " c: " + str(Coeff[c]) + "\n")

		#print results
		results = ''
		for m in mi:
			results += " %*s" % (MLen, m+1)
		results += ":"
		for m in mi:
			results += " %-*s" % (QNLen, QName[m])
		results += " %10.0f" % lnL[c]
		C = Coeff[c][0] - np.sum(Coeff[c][1:D+1] * Qmin[mi] / Qspan[mi])
		results += " %10.4g" % C
		for C in Coeff[c][1:D+1] / Qspan[mi]:
			results += " %10.4g" % C
		if A:
			if V:														### ADDED
				for C in Coeff[c][D+1:-1]:								### ADDED
					results += " %10.4g" % C							### ADDED
			else:														### ADDED
				for C in Coeff[c][D+1:]:								### ADDED
					results += " %10.4g" % C							### ADDED
##			results += " %10.4g" % Coeff[c][D+1]
		if V:
			results += " %10.4g" % Coeff[c][-1]
		print results
