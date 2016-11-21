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

def smooth_trigger( c_idx, odour, check=False ):

    # check if it is done
    c = cells[c_idx]
    source_attr = 'trigger_slow_%s' % odour
    attr = 'trigger_smooth_slow_%s' % odour 
    if check:
        # does the source exist
        if not c.can_load_attribute(source_attr):
            return True
        # is it done already
        if c.can_load_attribute(attr):
            return True
        return False

    # load source model
    c.load_attribute(source_attr, overwrite=False)
    a = getattr(c, source_attr)

    # set up the durations
    if odour == 'baseline':
        models = [a]
    else:
        models = [a.pre, a.first]
    for m in models:
        m.durations = durations
        m.mu__t = w__i.dot(m.mu__it)
        m.logmu__t = w__i.dot(m.logmu__it)
        m.logmu_std__t = np.sqrt( w__i.dot(m.logmu_std__it**2) )

    # check if we're done
    if c.can_load_attribute(attr):
        print ' - already done'
        return

    # save model
    setattr(c, attr, a)
    c.save_attribute(attr, overwrite=False)





"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    for o in odours:
        jobs.append( (smooth_trigger, c_idx, o) )

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
