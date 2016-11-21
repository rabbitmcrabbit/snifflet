import os
import sys
import warnings
import cPickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.io import loadmat as loadmat_original
from scipy.optimize import fmin_l_bfgs_b, fmin_ncg, line_search
from scipy.stats import linregress, spearmanr, wilcoxon
from scipy.stats.mstats import mquantiles
from scipy.ndimage.interpolation import affine_transform

import progress
from datatypes import PrintableBunch as Bunch
from autocr import AutoCR, cached

try:
    __IPYTHON__
    from IPython.core.debugger import Tracer
    tracer = Tracer()
except NameError:
    pass

sys.path.append('/users-archive/neil/mop')

np.seterr(all='ignore')

na = np.newaxis
A = np.array
conc = np.concatenate
in1d = np.lib.in1d
join = os.path.join
from numpy import log, exp, diag, nanmean, nanvar, shape
from numpy.linalg import norm, svd, inv
from numpy.random import permutation as perm


"""
================
Helper functions
================
"""

def pval_to_stars( p ):
    s = ''
    if p < 0.05:
        s += '*'
    if p < 0.01:
        s += '*'
    if p < 0.001:
        s += '*'
    return s

def off_diag_elements( x ):
    return conc([ x[i, i+1:] for i in range(len(x)) ])

def zero_mean( x ):
    return x - np.mean(x)


def nanmedian(y):
    return np.median(y[~np.isnan(y)])

def nancorrcoef( *a ):
    if len(a) == 1:
        x = a[0]
        N = len(x)
        cc = np.empty( (N, N) )
        for i in range(N):
            for j in range(i+1, N):
                y = x[[i,j], :]
                y = y[:, np.all(~np.isnan(y), axis=0)]
                if np.size(y) > 0:
                    cc[j, i] = cc[i, j] = np.corrcoef( y )[0, 1]
                else:
                    cc[j, i] = cc[i, j] = np.nan
        cc[range(N), range(N)] = 1.
        return cc
    else:
        y = A([ a ])
        return nancorrcoef(y)

def nancov( *a ):
    if len(a) == 1:
        x = a[0]
        N = len(x)
        c = np.empty( (N, N) )
        for i in range(N):
            for j in range(i, N):
                y = x[[i,j], :]
                y = y[:, np.all(~np.isnan(y), axis=0)]
                if np.size(y) > 0:
                    c[j, i] = c[i, j] = np.cov( y )[0, 1]
                else:
                    c[j, i] = c[i, j] = np.nan
        return c
    else:
        y = A([ a ])
        return nancov(y)

def nonnans( *a ):
    if len(a) == 1:
        x = a[0]
        return x[~np.isnan(x)]
    else:
        to_keep = np.all( np.isfinite(A(a)), axis=0 )
        return [ x[to_keep] for x in a ]

def loadmat(f):
    return loadmat_original( f, struct_as_record=False, squeeze_me=True )

def udiag(A):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    idx = np.triu_indices(n, k=1)
    return A[idx]

# soft threshold function
def F(x):
    return log( 1 + exp(x) ) / log(2)
def Finv(x):
    return log( exp(x * log(2)) - 1 )
def dF(x):
    return exp(x) / (1 + exp(x)) / log(2)


def resid_variance( R_tm ):
    R_tm = R_tm - np.nanmean( R_tm, axis=0 )[na, :]
    return np.nanvar( R_tm )


def stderr(x):
    return x.std() / np.sqrt(len(x))

def stderr_in_var(x, N_reps=100):
    if len(x) < 2:
        return np.nan
    return A([ x[ np.random.randint(0, len(x), size=len(x)) ].var() for _ in range(N_reps) ]).std()

def spearmanr_stderr( x, y, N_reps=100 ):
    x, y = all_sa[idx], all_prev_sa[idx]
    svals = np.zeros( N_reps )
    for i in range( N_reps ):
        tok = np.random.choice( len(x), len(x) )
        svals[i] = spearmanr( x[tok], y[tok] )[0]
    return np.std( svals )

def udiag(A):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    idx = np.triu_indices(n, k=1)
    return A[idx]

def S_Ellipse( S, S_ref=None, stderr=True, **kw ):
    # ellipse centre
    mu = np.mean( S, axis=1 )
    # ellipse shape
    if stderr:
        C = np.cov( S ) / S.shape[1]
    else:
        C = np.cov( S )
    v0, v1 = np.sqrt( C[0, 0] ), np.sqrt( C[1, 1] )
    angle = np.arccos( np.linalg.eigh( C )[1][:, 0][0] ) * 180/np.pi
    # offset
    if S_ref is not None:
        mu_ref = np.mean( S_ref, axis=1 )
        if stderr:
            C_ref = np.cov( S_ref ) / S_ref.shape[1]
        else:
            C_ref = np.cov( S_ref )
        v0_ref, v1_ref = np.sqrt( C_ref[0, 0] ), np.sqrt( C_ref[1, 1] )
        mu = mu - mu_ref
        v0 = np.sqrt( v0 ** 2 + v0_ref ** 2 )
        v1 = np.sqrt( v1 ** 2 + v1_ref ** 2 )            
    return Ellipse(mu, v0*2, v1*2, angle, **kw )


def nice_spines( ax, show_bottom=True, show_left=True ):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    if not show_bottom:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
    if not show_left:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])



def announce( c, attr ):
    s = '%s:  %s' % ( attr, c.__repr__() )
    print '='*len(s)
    print s

class AlreadyDoneException( Exception ):
    pass
