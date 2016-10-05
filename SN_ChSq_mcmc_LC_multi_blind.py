# pylint: disable=W0312

import numpy as np
import emcee
import astropy.io.fits as fits
from astropy.table import Table

#import glob

import scipy
from scipy.integrate import quad
import os

import time

import math

from optparse import OptionParser

from chainconsumer import ChainConsumer

def Ez( z, Omega_M, Omega_L, w_o, w_a):

    # scale factor
    a = 1 / (1+z)

    # equation of state of Dark Energy w(z) (CPL)
    w_z = w_o + w_a * ( z * a )

    # E(z)
    Ez = (Omega_L * math.pow( (1+z), (3*(1+w_z)) ) ) +    (Omega_M * math.pow( (1+z), 3) )

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
def lnprob(theta, zhel, zcmb, mb, x1, color, thirdvar, Cmu, blind_values):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Cmu, blind_values)


# defines likelihood.  has to be ln likelihood
def lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Cmu, blind_values):
    
    

    # fold in blinding
    blind_index = 0
    for blind_index in range(len(blind_values)):
        theta[blind_index] = theta[blind_index] * blind_values[blind_index]

    # unpack parameters once they've been blinded
    my_Om0, my_w0, alpha, beta, M_1_B, Delta_M = theta

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

    #print 'ok so far'
    #inv_CM = scipy.linalg.inv(Cmu) #my np.linalg crashes python BZ
    inv_CM = np.linalg.pinv(Cmu)

    ChSq = np.dot(Delta, (np.dot(inv_CM, Delta)))

    # ***** write parameters ******


    param_file_name = 'my_params_JLA_FlatwCDM_CPL_uncorrected_PV_blind_%s.txt'%date

    if not os.path.isdir(options.chains):
        os.mkdir(options.chains)
        os.mkdir(options.chains + '/ChSq')

    chain_path_file = options.chains + '/' + param_file_name
    f_handle = open(chain_path_file, 'a')
    stringOut = str(my_Om0) +  ',' + str(my_w0)  + ',' + str(alpha) + ',' + str(beta) + ',' + str(M_1_B) + ',' + str(Delta_M) + '\n'

    f_handle.write(stringOut)
    f_handle.close()

    param_file_name = 'ChSqFile_JLA_FlatwCDM_CPL_uncorrected_PV_blind_%s.txt'%date
    #chain_path = 'ChSq_Chains/'
    chain_path_file = options.chains + '/ChSq/' + param_file_name
    f_handle = open(chain_path_file, 'a')
    stringOut = str(ChSq) + '\n'
    f_handle.write(stringOut)
    f_handle.close()


    return -0.5 * ChSq


if __name__ == '__main__':
        
    t = time.gmtime(time.time())
    date = '%4d%02d%02d' % (t[0], t[1], t[2])

    parser = OptionParser()

    parser.add_option('-l', '--lcfits', dest='lcfits', #default= ''
              help='fits file containing all SN data including light curve fits')

    parser.add_option('-C', '--covmat', dest='covmat', #default = ''
              help='total SN magnitude covariance matrix (C_total*.fits from Covariance code)')

    parser.add_option('-c', '--chains', dest='chains', default='Chains_%s'%date,
              help='directory to store chains')
    
    parser.add_option('-w', '--nwalkers', dest='nwalkers', default=200,
              help='number of walkers')
    
    parser.add_option('-s', '--nSteps', dest='nSteps', default=1000,
              help='number of walkers')
    
    #parser.add_option('-p', '--plot', action='store_true', default=False,
    #          help='plot walkers and contours with ChainConsumer')
    
    (options, args) = parser.parse_args()
    
    Cmu = fits.getdata(options.covmat)

    data = Table.read(options.lcfits)

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
    nSteps = options.nSteps
    nwalkers = options.nwalkers
    
    # *** make blinding values ***
    
    N_params_to_blind = 2 # how many parameters we want to blind.  Put the blind parameters at the start of thd parameter array

    blind_mean = 1.0
    blind_standard_deviation = 0.3
    
    blind_values = []
    blind_index = 0
    
    # seed blind
    
    sd = 42 #random seed to replace
    np.random.seed(sd)
    
    for blind_index in range(N_params_to_blind):
        blind_values = np.append(blind_values, np.random.normal(blind_mean, blind_standard_deviation)  )

    # save the blind values.  Maybe add a run or date string so they don't get overwritten
    np.save('blind_values.npy', blind_values)

    
    #startValues = [ my_Om0, my_w0, alpha, beta, M_1_B, Delta_M]

    pos = [startValues + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

    # setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zhel, zcmb, mb, x1, color, thirdvar, Cmu, blind_values), threads=4)
    # run the sampler
    # how many steps (will have nSteps*nwalkers of samples)
    sampler.run_mcmc(pos, nSteps)
