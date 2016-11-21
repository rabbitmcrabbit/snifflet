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

# odours to fit
odours = np.unique( conc([ r.data.odour_name__a for r in recordings ]))


def fit_affinity( c_idx, odour_name, inh_speed, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'plain_%s_%s' % (inh_speed, odour_name)
    if check:
        # is it done already
        if c.can_load_attribute( attr ):
            return True
        # can it be done
        N_trials = (c.recording.data.odour_name__a == odour_name).sum()
        return not (N_trials > 0)
    # filter to this odour
    data = c.recording.data
    trials_to_keep = (
            (data.odour_name__a == odour_name) &
            (data.odour_conc__a > 0) &
            A([ len(sn)>0 for sn in data.full_sniff_cycles__ai ]) )
    data = data.filter_trials( trials_to_keep )
    # run
    return fit( c, attr, data, inh_speed )

def fit_concentration( c_idx, odour_conc_idx, inh_speed, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'plain_%s_o%d' % (inh_speed, odour_conc_idx)
    if check:
        # is it done already
        if c.can_load_attribute( attr ):
            return True
        # not a concentration series
        if not c.recording.is_concentration_series:
            return True
        # there are no trials for this condition
        N_trials = (c.recording.data.odour_conc_idx__a == odour_conc_idx).sum()
        return not (N_trials > 0)
    # filter to this odour
    data = c.recording.data
    trials_to_keep = (
                (data.odour_conc_idx__a == odour_conc_idx) &
                A([ len(sn)>0 for sn in data.full_sniff_cycles__ai ]) )
    data = data.filter_trials( trials_to_keep )
    # run
    return fit( c, attr, data, inh_speed )



def fit( c, attr, data, inh_speed ):

    # ensure there is an inhalation during the stimulus presentation
    first_exists = A([ data.stim__at[a, data.inhalations__ai[a][-1][0]] > 0 for a in range(data.A) ])
    data = data.filter_trials( first_exists )

    # separate into baseline and first sniff
    data_pre = data.copy()
    data_first = data.copy()
    data_pre.inhalations__ai = []
    data_first.inhalations__ai = []
    for a in range(data.A):
        inh__i = data.inhalations__ai[a]
        try:
            data_pre.inhalations__ai.append([inh for inh in inh__i if inh[0] < data.stim_t0__a[a]])
            data_first.inhalations__ai.append([[inh for inh in inh__i if inh[0] >= data.stim_t0__a[a]][0]])
        except IndexError:
            tracer()
    # only the inhalations of the right speed
    is_fast = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] < 100)
    is_slow = lambda i: (i[1] - i[0] >= 100) and (i[1] - i[0] <= 300)
    is_both = lambda i: (i[1] - i[0] >= 20) and (i[1] - i[0] <= 300)
    if inh_speed == 'slow':
        data_pre = data_pre.filter_inhalations( is_slow )
        data_first = data_first.filter_inhalations( is_slow )
    elif inh_speed == 'fast':
        data_pre = data_pre.filter_inhalations( is_fast )
        data_first = data_first.filter_inhalations( is_fast )
    elif inh_speed == 'both':
        data_pre = data_pre.filter_inhalations( is_both )
        data_first = data_first.filter_inhalations( is_both )
    else:
        raise ValueError('unknown inh_speed: %s' % inh_speed)

    # must be a sniff
    to_keep__a = A([ len(inh) > 0 for inh in data_first.inhalations__ai ])
    data_first = data_first.filter_trials( to_keep__a )
    data_pre = data_pre.filter_trials( to_keep__a )

    # number of sniffs
    if data_first.N_trials > 0:
        N_sniffs_pre = sum([len(inh) for inh in data_pre.inhalations__ai])
        N_sniffs_first = sum([len(inh) for inh in data_first.inhalations__ai])
    else:
        N_sniffs_pre = 0
        N_sniffs_first = 0

    # XY data for each
    XY_first = data_first.get_warp_XY_data( warp_mode=None, **kw[inh_speed] )
    XY_pre = data_pre.get_warp_XY_data( warp_mode=None, **kw[inh_speed] )

    # anchor the first time bin's data to be the same for `pre` and `first`
    bin0__t = A([ XY_first.X_stim__td[:, 0], XY_pre.X_stim__td[:, 0] ]).max(axis=0)
    for XY in [XY_first, XY_pre]:
        XY.X_stim__td[:, 0] = bin0__t

    # nan out the irrelevant bins
    for XY in [XY_first, XY_pre]:
        to_nan__t = (XY.X_stim__td == 0).all(axis=1)
        XY.Y__nt[ :, to_nan__t ] = np.nan

    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return

    # solve: pre-odour inhalations
    print ' - fitting pre'
    model_pre = c.get_warp_asd_model(XY_pre)
    model_pre.solve(verbose=False)
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return

    # solve: first odour inhalations
    print ' - fitting first'
    model_first = c.get_warp_asd_model(XY_first)
    model_first.solve(verbose=False)
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
    # container
    results = Bunch()
    # prepare results: pre
    results.pre = core.WarpModel(model_pre, durations=durations, **kw[inh_speed])
    results.pre.N_sniffs = N_sniffs_pre
    # prepare results: first sniff
    results.first = core.WarpModel(model_first, durations=durations, **kw[inh_speed])
    results.first.N_sniffs = N_sniffs_first
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
    for inh_speed in ['slow']:
        for o in odours:
            jobs.append( (fit_affinity, c_idx, o, inh_speed) )
        for i in range(1, 7):
            jobs.append( (fit_concentration, c_idx, i, inh_speed) )

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
