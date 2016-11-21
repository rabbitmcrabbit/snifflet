from helpers import *
import glm
import core
from get_data import recordings, cells
from IPython.display import clear_output

for r in recordings:
    try:
        del r.sniffs
    except AttributeError:
        pass

# parameters
kw = {}
kw['slow'] = {'binsize_ms':5, 'D_warp_inh':30, 'D_warp_total':120, 'D_spike_history':0}
kw['fast'] = {'binsize_ms':5, 'D_warp_inh':12, 'D_warp_total':60, 'D_spike_history':0}
kw['both'] = {'binsize_ms':5, 'D_warp_inh':30, 'D_warp_total':120, 'D_spike_history':0}


def fit_baseline( c_idx, inh_speed, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'plain_%s_baseline' % (inh_speed)
    if check:
        # is it done already
        return c.can_load_attribute( attr )
    # collect data
    data = c.recording.data
    # there must be some stimulus on this trial
    data = data.filter_trials( data.stim__at.sum(axis=1) > 0 )
    # remove stimulus-driven component
    data_pre = data.copy()
    data_pre.inhalations__ai = []
    for a in range(data.A):
        inh__i = data.inhalations__ai[a]
        try:
            data_pre.inhalations__ai.append([inh for inh in inh__i if inh[0] < data.stim_t0__a[a]])
        except IndexError:
            tracer()
    # only the inhalations of the right speed
    is_fast = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] < 100)
    is_slow = lambda i: (i[1] - i[0] >= 100) and (i[1] - i[0] <= 300)
    is_both = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] <= 300)
    if inh_speed == 'slow':
        data_pre = data_pre.filter_inhalations(is_slow)
    elif inh_speed == 'fast':
        data_pre = data_pre.filter_inhalations(is_fast)
    elif inh_speed == 'both':
        data_pre = data_pre.filter_inhalations( is_both )
    else:
        raise ValueError('unknown inh_speed: %s' % inh_speed)
    # must be a sniff
    to_keep__a = A([ len(inh) > 0 for inh in data_pre.inhalations__ai ])
    data_pre = data_pre.filter_trials( to_keep__a )
    # number of sniffs
    if data_pre.N_trials > 0:
        N_sniffs_pre = sum([len(inh) for inh in data_pre.inhalations__ai])
    else:
        N_sniffs_pre = 0
    # XY data for each
    XY_pre = data_pre.get_warp_XY_data( warp_mode=None, **kw[inh_speed] )
    # nan out the irrelevant bins
    for XY in [XY_pre]:
        to_nan__t = (XY.X_stim__td == 0).all(axis=1)
        XY.Y__nt[ :, to_nan__t ] = np.nan

    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return

    # solve
    print ' - fitting baseline'
    model = c.get_warp_asd_model(XY_pre)
    model.solve(verbose=False)
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return

    # durations for WarpModel results
    if inh_speed == 'fast':
        durations = [60]
    elif inh_speed == 'slow':
        durations = [150]
    elif inh_speed == 'both':
        durations = [150]
    else:
        raise ValueError('unknown inh_speed: `%s`' % inh_speed)
    # prepare results
    results = core.WarpModel(model, durations=durations, **kw[inh_speed])
    results.N_sniffs = N_sniffs_pre
    # save these results
    setattr( c, attr, results )
    c.save_attribute( attr, overwrite=False )





"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    for inh_speed in ['slow', 'fast', 'both']:
        jobs.append( (fit_baseline, c_idx, inh_speed) )

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
