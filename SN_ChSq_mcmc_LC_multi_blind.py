import numpy as np
import emcee
import pyfits
import glob
from scipy.integrate import quad
import sys

import math
# import astropy
# import matplotlib.pyplot as plt
# from collections import OrderedDict
# from astropy import units as u

def Ez( z, Omega_M, Omega_L, w_o, w_a):

	# scale factor
	a = 1 / (1+z)

	# equation of state of Dark Energy w(z) (CPL)
	w_z = w_o + w_a * ( z * a )

	# E(z)
	Ez = (Omega_L * math.pow( (1+z), (3*(1+w_z)) ) ) +	(Omega_M * math.pow( (1+z), 3) )

	Ez = math.sqrt( Ez )
	Ez = 1.0 / Ez

	return Ez


def Luminosity_Distance( z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a):


	# factor in units
	Dc = Comoving_Distance( z_cmb, Omega_M, Omega_L, w_o, w_a)

	# calculate the luminosity distance
	Dl = Dc * (1.0+z_hel)

	return Dl

def Comoving_Distance( z, Omega_M, Omega_L, w_o, w_a):

	# Speed of light, in km / s
	cLight = 299792.458

	# Hubble's constant, in (km / s) / Mpc
	H_o = 70.0

	# integrate E(z) to get comoving distance
	Dc, error = quad( Ez, 0, z, args=(Omega_M, Omega_L, w_o, w_a)  )

	# factor in units
	Dc = Dc* (cLight/H_o)

	return Dc


def distance_modulus( z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a   ):

	# get the luminosity distance
	d_L = Luminosity_Distance( z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a)

	# convert to distance modulus
	mu = 25 + 5.0 *  np.log10( d_L  )

	return mu


# defines a prior.  just sets acceptable ranges
def lnprior(theta):
    my_Om0, my_w0, alpha, beta, M_1_B, Delta_M = theta

    if  0.0 < my_Om0 < 1.0 and -2.0 < my_w0 < -0.0:
        return 0.0
    return -np.inf


# lnprob - this just combines prior with likelihood
def lnprob(theta, zhel, zcmb, mb, x1, color, thirdvar, Ceta,blind_values):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Ceta, blind_values)


# defines likelihood.  has to be ln likelihood
def lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Cmu, blind_values):
	
	

	# fold in blinding
	blind_index = 0
	for blind_index in range(len(blind_values)):
		theta[blind_index] = theta[blind_index] * blind_values[blind_index]

	# unpack parameters once they've been blinded
	my_Om0, my_w0, alpha, beta, M_1_B, Delta_M = theta


	"""# assemble covariance matrix
	Cmu = np.zeros_like(Ceta[::3, ::3])

	for i, coef1 in enumerate([1., alpha, -beta]):
		for j, coef2 in enumerate([1., alpha, -beta]):
			Cmu += (coef1 * coef2) * Ceta[i::3, j::3]

	# Add diagonal term from Eq. 13
	sigma = np.loadtxt('covmat/sigma_mu.txt')
	sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
	Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2"""

	# calculate observation
	mod = mb - (M_1_B - alpha * x1 + beta * color)

	for i in range(0, len(zcmb)):
		if thirdvar[i] > 10:
			mod[i] = mod[i] - Delta_M

	# calculate theory
	mod_theory = []

	for i in range(0, len(zcmb)):
		mod_i = distance_modulus(zhel[i], zcmb[i], my_Om0, (1.0-my_Om0), my_w0, 0.0)
		mod_theory = np.append(mod_theory,mod_i)


	# calculate Chi-squared

	Delta = mod - mod_theory

	inv_CM = np.linalg.pinv(Cmu)

	ChSq = np.dot(Delta, (np.dot(inv_CM, Delta)))

	# ***** write parameters ******


    t = time.gmtime(time.time())
    date = '%4d%02d%02d' % (t[0], t[1], t[2])

	param_file_name = 'my_params_JLA_FlatwCDM_CPL_uncorrected_PV_blind_%s.txt'%date

	chain_path = 'Chains/'
	chain_path_file = chain_path + param_file_name
	f_handle = open(chain_path_file, 'a')
	stringOut = str(my_Om0) +  ',' + str(my_w0)  + ',' + str(
		alpha) + ',' + str(beta) + ',' + str(M_1_B) + ',' + str(Delta_M) + '\n'

	f_handle.write(stringOut)
	f_handle.close()

	param_file_name = 'ChSqFile_JLA_FlatwCDM_CPL_uncorrected_PV_blind_%s.txt'%date
	chain_path = 'ChSq_Chains/'
	chain_path_file = chain_path + param_file_name
	f_handle = open(chain_path_file, 'a')
	stringOut = str(ChSq) + '\n'
	f_handle.write(stringOut)
	f_handle.close()


	return -0.5 * ChSq


# ****** load eta covariance matrix, now as C_total ****************

#Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('covmat/C*.fits')])
#Ceta = pyfits.getdata('C_eta_20160610.fits')


date_Cmu = '20160915'
Cmu = pyfits.getdata('C_total_%s.fits')%date_Cmu

# ****** load JLA, now from fits file ****************

FileName = 'DES_20160914.fits'

data = Table.read(FileName)

zcmb = data['zcmb']
zhel = data['zhel']
mb = data['mb']
x1 = data['x1']
color = data['color']
thirdvar = data['3rdvar']


# ****** load uncorrected redshifts - use instead ****************
# comment to turn on or off
#FileName = 'z_JLA_uncorrected_Bonnie.txt'
#DataBlock = np.genfromtxt(FileName , skip_header=1,  delimiter = ' ')

#zhel =DataBlock[:,0]
#zcmb =DataBlock[:,1]


# best fit values from Betoule paper
alpha = 0.141
beta = 3.101
M_1_B = -19.05
Delta_M = -0.07

H0 = 70.0
my_Om0 = 0.3
Ode0 = 0.7
my_w0 = -1.0
wa = 0.0

startValues = [ my_Om0, my_w0, alpha, beta, M_1_B, Delta_M]

# how many parameters to fit
ndim = len(startValues)

# how many walkers
nwalkers = 200
nSteps = 1000


# *** make blinding values ***

N_params_to_blind = 2 # how many parameters we want to blind.  Put the blind parameters at the start of thd parameter array

blind_mean = 1.0
blind_standard_deviation = 0.3

blind_values = []
blind_index = 0

# seed blind

sd = 'random seed to replace'
np.random.seed(sd)

for blind_index in range(N_params_to_blind):
	blind_values = np.append(blind_values, np.random.normal(blind_mean, blind_standard_deviation)  )


# save the blind values.  Maybe add a run or date string so they don't get overwritten
np.save('blind_values.npy', blind_values)


startValues = [ my_Om0, my_w0, alpha, beta, M_1_B, Delta_M]

pos = [startValues + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

# setup the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zhel, zcmb, mb, x1, color, thirdvar, Ceta, blind_values), threads=4)
# run the sampler
# how many steps (will have nSteps*nwalkers of samples)
sampler.run_mcmc(pos, nSteps)
