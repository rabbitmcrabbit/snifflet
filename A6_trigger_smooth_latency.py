from matplotlib import pyplot as plt
from IPython.display import clear_output
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

for r in recordings:
    try:
        del r.sniffs
    except AttributeError:
        pass

# odours to fit
odours = np.unique( conc([ r.data.odour_name__a for r in recordings ]))
odours = conc([ ['baseline', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6'], odours ])

# trigger smoothing values
trigger_smoother = cPickle.load(open('data/trigger_smoother.pickle'))
durations = trigger_smoother['durations']
w__i = trigger_smoother['weights']

def calc_latency( c_idx, check=False ):

    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_smooth_slow_latency'
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        return False

    # load source models
    for o in odours:
        a = 'trigger_smooth_slow_%s' % o
        if c.can_load_attribute(a):
            c.load_attribute(a, overwrite=False)

    attrs = [k for k in c.__dict__.keys() if k.startswith('trigger_smooth_slow_')]

    c.latency = Bunch()
    for a in progress.dots(attrs):
        # get the model
        odour = a.split('trigger_smooth_slow_')[1]
        if odour == 'baseline':
            continue
        m = getattr( c, a ).first
        # draw random samples, to calculate error
        X = np.tensordot(w__i, m.X__itd, axes=[0, 0])
        del m.X__itd
        samples = X.dot( np.random.multivariate_normal(m.k, m.Lambda, size=1000).T )
        # calculate
        c.latency[odour] = Bunch()
        c.latency[odour].max_samples = np.argmax( samples, axis=0 ) * 5
        c.latency[odour].min_samples = np.argmin( samples, axis=0 ) * 5
        c.latency[odour].max = np.argmax( m.mu__it[0] ) * 5
        c.latency[odour].min = np.argmin( m.mu__it[0] ) * 5
        c.latency[odour].max_err = np.diff(mquantiles(c.latency[odour].max_samples, [0.25, 0.75]))[0]
        c.latency[odour].min_err = np.diff(mquantiles(c.latency[odour].min_samples, [0.25, 0.75]))[0]
        
        # latency to significant deviation from baseline
        m_baseline = getattr( c, a ).pre
        dlogmu__t = m.logmu__t - m_baseline.logmu__t
        dlogmu_std__t = np.sqrt(m.logmu_std__t**2 + m_baseline.logmu_std__t**2)
        N_std__t = dlogmu__t / dlogmu_std__t
        deviation_idxs = (np.abs(N_std__t) > 3).nonzero()[0]
        if len(deviation_idxs) == 0:
            c.latency[odour].significance = np.nan
            c.latency[odour].significance_sign = 0
        else:
            c.latency[odour].significance = deviation_idxs[0] * 5
            c.latency[odour].significance_sign = np.sign(N_std__t[deviation_idxs[0]])

        # check if we're done
        if c.can_load_attribute(attr):
            print ' - already done'
            return

    # save model
    setattr( c, attr, c.latency )
    c.save_attribute(attr, overwrite=False)




"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    jobs.append( (calc_latency, c_idx) )

# check if jobs are done
checked_jobs = []
for job in progress.dots( jobs, 'checking if jobs are done' ):
    func = job[0]
    args = tuple([ a for a in job[1:] ])
    if not func( *args, check=True ):
        checked_jobs.append(job)
print '%d jobs to do out of %d' % (len(checked_jobs), len(jobs))
jobs = checked_jobs


if __name__ == '__main__':
    # run
    print ' '
    print ' '
    randomised_jobs = [ jobs[i] for i in np.random.permutation(len(jobs)) ]
    for job in progress.numbers( randomised_jobs ):
        func = job[0]
        args = tuple([ a for a in job[1:] ])
        if not func( *args, check=True ):
            func( *args )
        print ' '
        print ' '
