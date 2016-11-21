
# coding: utf-8

# # setup

# In[1]:

get_ipython().magic(u'matplotlib inline')

from matplotlib import pyplot as plt
from IPython.display import clear_output

max_fig_w = 28
max_fig_h = 12

plt.rcParams['figure.facecolor'] = (1, 1, 1)
plt.rcParams['axes.facecolor'] = (1, 1, 1)
plt.rcParams['xtick.direction'] = 'out'                                                                              
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Droid Sans']
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.max_open_warning'] = 40
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['figure.max_open_warning'] = 40


# In[2]:

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import ranksums, fisher_exact, ks_2samp, chi2_contingency
from scipy.signal import correlate

from helpers import *
import glm
import core
import get_data
from get_data import recordings, cells, trials

clear_output()


# In[3]:

odours = A([ 
        u'2_4_dimethylacetophenone',
        u'benzaldehyde',
        u'menthone',
        u'acetophenone',        
        u'4_methylacetophenone',
        u'ethyl_tiglate',
        u'2_hydroxyacetophenone' ])


# In[4]:

for r in recordings:
    try:
        del r.sniffs
    except AttributeError:
        pass


# # main

# In[7]:

# parameters
kw = {}
kw = {'binsize_ms':5, 'D_warp_inh':30, 'D_warp_total':120, 'D_spike_history':0}


# In[15]:

c_idx = 1
odour_name = '2_hydroxyacetophenone'
inh_speed = 'slow'
warp_mode = 'halfinh'


# In[9]:

def get_warp_data( c_idx, odour_name, inh_speed ):

    # get basic data
    c = cells[c_idx]
    data = c.recording.data
    trials_to_keep = (
            (data.odour_name__a == odour_name) &
            (data.odour_conc__a > 0) &
            A([ len(sn)>0 for sn in data.full_sniff_cycles__ai ]) )
    data = data.filter_trials( trials_to_keep )

    # ensure there is an inhalation during the stimulus presentation
    first_exists = A([ data.stim__at[a, data.inhalations__ai[a][-1][0]] > 0 for a in range(data.A) ])
    data = data.filter_trials( first_exists )
    # get first sniff
    inhalations__ai = []
    for a in range(data.A):
        inh__i = data.inhalations__ai[a]
        try:
            inhalations__ai.append([[inh for inh in inh__i if inh[0] >= data.stim_t0__a[a]][0]])
        except IndexError:
            tracer()
    data.inhalations__ai = inhalations__ai
    # only the inhalations of the right speed
    is_fast = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] < 100)
    is_slow = lambda i: (i[1] - i[0] >= 100) and (i[1] - i[0] <= 300)
    is_both = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] <= 300)
    if inh_speed == 'slow':
        data = data.filter_inhalations( is_slow )
    elif inh_speed == 'fast':
        data = data.filter_inhalations( is_fast )
    elif inh_speed == 'both':
        data = data.filter_inhalations( is_both )
    else:
        raise ValueError('unknown inh_speed: %s' % inh_speed)
    # must be a sniff
    to_keep__a = A([ len(inh) > 0 for inh in data.inhalations__ai ])
    data = data.filter_trials( to_keep__a )
    # number of sniffs
    if data.N_trials > 0:
        N_sniffs = sum([len(inh) for inh in data.inhalations__ai])
    else:
        return None, None
    # find the full sniff cycles
    cycle_times = np.zeros( (data.A, 2), dtype=int )
    for a in range(data.A):
        t0 = data.inhalations__ai[a][0][0]
        if t0 == data.full_sniff_cycles__ai[a][-1][-1]:
            t1 = data.T
        else:
            t1 = [cy for cy in data.full_sniff_cycles__ai[a] if cy[0] == t0][0][1]
        cycle_times[a, :] = (t0, t1)
    cycle_smps = cycle_times / 5
    # keep the first sniff data only
    to_keep__t = np.zeros( data.A * 1100, dtype=bool )
    for a in range(data.A):
        t0 = (a*1100) + cycle_smps[a, 0]
        t1 = (a*1100) + cycle_smps[a, 1]
        to_keep__t[ t0:t1 ] = True
    # done
    return data, to_keep__t


# In[20]:

#def calc_cv_warp( c_idx, odour_name, inh_speed, warp_mode, check=False ):

# parse warp mode
c = cells[c_idx]
if warp_mode == 'halfinh':
    attr = 'cv_warphalfinf_%s_%s' % (inh_speed, odour_name)
else:
    raise ValueError('unknown warp_mode: %s' % warp_mode)

# get the cv slices
cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
c.load_attribute(cv_slices_attr)
cv_slices = getattr(c, cv_slices_attr)

# get the data
data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )


# In[160]:

reload(core)


# In[157]:

XY = data.get_halfwarp_XY_data()
XY.Y__nt[ :, ~to_keep__t ] = np.nan


# In[176]:

T = self.T 
T_ds = T / binsize_ms
T_total = T_ds * binsize_ms
N_cells = self.N_cells 
N_trials = self.N_trials 
# spikes 
spikes__nat = self.spikes__nat[ :, :, :T_total ]
Y__nat = spikes__nat.reshape(
        (N_cells, N_trials, T_ds, binsize_ms) ).sum(axis=-1)
Y__nat = Y__nat.astype(float) 
Y__nt = np.reshape( Y__nat, (N_cells, N_trials * T_ds) )
# helper functions
def delta(i, max_T=1000):
    z = np.zeros(max_T, dtype=float)
    z[i] = 1
    return z
def dilation_matrix( dilation_factor, D, max_T=1000 ):
    return A([ 
            affine_transform( delta(i, max_T), A([dilation_factor]), order=1 ) 
            for i in range(D) ]).T
# warp matrix
warp__atd = np.zeros( (N_trials, T_ds, D_warp_total) )
for a in range( self.N_trials ):
    # run through inhalations
    inhs = self.inhalations__ai[a]
    for i, inh in enumerate( inhs ):
        # time markers for this inhalation (in samples)
        t0_smp, t1_smp = np.round(A(inh) / 5.).astype(int)
        inh_dur_smp = ( inh[1] - inh[0] ) / float(binsize_ms)
        # find which sniff cycle this is
        try:
            this_cycle = [cy for cy in self.full_sniff_cycles__ai[a] if cy[0] == inh[0]][0]
            t2_smp = np.round(this_cycle[1] / 5.).astype(int)
        except IndexError:
            if inh[0] >= self.full_sniff_cycles__ai[a][-1][1]:
                t2_smp = self.T / binsize_ms
                pass
            else:
                raise IndexError('inhalation does not appear in sniff cycles')
        sniff_dur_smp = t2_smp - t1_smp
        # build the dilation matrix for this sniff
        max_T = max([ sniff_dur_smp, D_warp_total ])
        dilation_factor_1 = D_warp_inh / float(inh_dur_smp)
        dilation_factor_2 = (D_warp_total - D_warp_inh) / float(sniff_dur_smp - inh_dur_smp)
        D1__td = dilation_matrix( dilation_factor_1, D_warp_inh, max_T=max_T )
        D2__td = dilation_matrix( dilation_factor_2, D_warp_total - D_warp_inh, max_T=max_T )
        offset = (D1__td.sum(axis=1) == 0).nonzero()[0][0]
        D2__td = np.roll( D2__td, offset, axis=0 )
        this_D__td = np.hstack([D1__td, D2__td])
        this_D__td /= this_D__td.sum(axis=1)[:, None]
        this_D__td[ ~np.isfinite(this_D__td) ] = 0.
        warp__atd[ a, t0_smp : t0_smp + sniff_dur_smp, : ] = this_D__td[ :sniff_dur_smp, : ]
X_warp__td = warp__atd.reshape((N_trials*T_ds, D_warp_total))
# create spike history regressor
X_history = self._get_X_history( Y__nat, binsize_ms, **kw )
# return
results = Bunch({
    'X_stim__td' : X_warp__td,
    'Y__nt' : Y__nt, 
    'binsize_ms' : binsize_ms,
    'D_warp_total' : D_warp_total,
    'D_warp_inh' : D_warp_inh })
results.update( **X_history )
return results


# In[177]:

imshow( this_D__td[ :sniff_dur_smp, : ] )


# In[169]:

imshow(D2__td.T)


# In[158]:

# fit
LL_training, LL_testing = [], []
for s in progress.dots(cv_slices, 'fitting cv'):
    m = c.get_warp_asd_model(XY, testing_proportion=0.2)
    m.training_slices = s.training_slices
    m.testing_slices = s.testing_slices
    m.solve(verbose=False)
    LL_training.append( m.posterior.LL_training_per_observation * 200 )
    LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
LL_training = A(LL_training)
LL_testing = A(LL_testing)
# save
results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})


# In[148]:

(results.LL_training - c.cv_warp_slow_2_hydroxyacetophenone.LL_training).mean()


# In[150]:

(results.LL_testing - c.cv_warp_slow_2_hydroxyacetophenone.LL_testing)


# In[144]:

c.load_attribute('cv_warp_slow_2_hydroxyacetophenone')


# In[146]:

c.cv_warp_slow_2_hydroxyacetophenone.LL_training

