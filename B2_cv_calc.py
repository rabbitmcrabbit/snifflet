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

trigkw = {}
trigkw['slow'] = {'binsize_ms':5, 'D_trigger':100, 'D_spike_history':0}
trigkw['both'] = {'binsize_ms':5, 'D_trigger':100, 'D_spike_history':0}

# odours to fit
odours = np.unique( conc([ r.data.odour_name__a for r in recordings ]))


def get_warp_data( c_idx, odour_name, inh_speed ):

    if odour_name == 'baseline':
        return get_baseline_warp_data( c_idx, inh_speed )
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


def get_baseline_warp_data( c_idx, inh_speed ):

    # get basic data
    c = cells[c_idx]
    data = c.recording.data
    trials_to_keep = (
            (data.stim__at.sum(axis=1) > 0) &
            (data.odour_conc__a > 0) &
            A([ len(sn)>0 for sn in data.full_sniff_cycles__ai ]) )
    data = data.filter_trials( trials_to_keep )
    # remove stimulus-driven component
    inhalations__ai = []
    for a in range(data.A):
        inh__i = data.inhalations__ai[a]
        try:
            inhalations__ai.append([inh for inh in inh__i if inh[0] < data.stim_t0__a[a]])
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
    cycle_times = np.empty( (data.A), dtype=object )
    for a in range(data.A):
        cycle_times[a] = [cy for cy in data.full_sniff_cycles__ai[a] 
                if cy[0] in A(data.inhalations__ai[a])[:, 0]]
        if len(cycle_times[a]) != len(data.inhalations__ai[a]):
            data.inhalations__ai[a] = [inh for inh in data.inhalations__ai[a] 
                    if inh[0] in A(cycle_times[a])[:, 0]]
            assert len(cycle_times[a]) == len(data.inhalations__ai[a])
    cycle_smps = A([ A(cy)/5 for cy in cycle_times ])
    # keep the full sniff cycles only
    to_keep__t = np.zeros( data.A * 1100, dtype=bool )
    for a in range(data.A):
        for cs in cycle_smps[a]:
            t0 = (a*1100) + cs[0]
            t1 = (a*1100) + cs[1]
            to_keep__t[ t0:t1 ] = True
    # done
    return data, to_keep__t



def calc_cv_warp( c_idx, odour_name, inh_speed, warp_mode, check=False ):
    # parse warp mode
    c = cells[c_idx]
    if warp_mode == 'inh':
        attr = 'cv_warp_%s_%s' % (inh_speed, odour_name)
    elif warp_mode == 'sniff':
        attr = 'cv_warpsniff_%s_%s' % (inh_speed, odour_name)
    elif warp_mode is None:
        attr = 'cv_plain_%s_%s' % (inh_speed, odour_name)
    else:
        raise ValueError('unknown warp_mode: %s' % warp_mode)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_warp_XY_data( warp_mode=warp_mode, **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_warp_asd_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)


def calc_cv_ridge_warp( c_idx, odour_name, inh_speed, warp_mode, check=False ):
    # parse warp mode
    c = cells[c_idx]
    if warp_mode == 'inh':
        attr = 'cv_ridge_warp_%s_%s' % (inh_speed, odour_name)
    elif warp_mode == 'sniff':
        attr = 'cv_ridge_warpsniff_%s_%s' % (inh_speed, odour_name)
    elif warp_mode is None:
        attr = 'cv_ridge_plain_%s_%s' % (inh_speed, odour_name)
    else:
        raise ValueError('unknown warp_mode: %s' % warp_mode)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_warp_XY_data( warp_mode=warp_mode, **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_ridge_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)


def calc_cv_const( c_idx, odour_name, inh_speed, check=False ):
    # parse warp mode
    c = cells[c_idx]
    attr = 'cv_const_%s_%s' % (inh_speed, odour_name)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_warp_XY_data( warp_mode='sniff', **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    XY.X_stim__td[ to_keep__t, 0 ] = 1.
    XY.X_stim__td = XY.X_stim__td[ :, [0] ]
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_ridge_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)

def calc_cv_trigger( c_idx, odour_name, inh_speed, N_triggers, check=False ):
    # parse warp mode
    c = cells[c_idx]
    attr = 'cv_trigger_%d_%s_%s' % (N_triggers, inh_speed, odour_name)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_trigger_XY_data( N_triggers=N_triggers, **trigkw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_trigger_asd_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)

def calc_cv_halfwarp( c_idx, odour_name, inh_speed, check=False ):
    # parse warp mode
    c = cells[c_idx]
    attr = 'cv_halfwarp_%s_%s' % (inh_speed, odour_name)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_halfwarp_XY_data( **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_warp_asd_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)

def calc_cv_doublewarp( c_idx, odour_name, inh_speed, check=False ):
    # parse warp mode
    c = cells[c_idx]
    attr = 'cv_doublewarp_%s_%s' % (inh_speed, odour_name)
    # check if it is done
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True
    # is it done already
    if c.can_load_attribute(attr):
        return 
    # get the cv slices
    cv_slices_attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    c.load_attribute(cv_slices_attr)
    cv_slices = getattr(c, cv_slices_attr)
    if cv_slices is None:
        return None
    # get the data
    data, to_keep__t = get_warp_data( c_idx, odour_name, inh_speed )
    # XY data
    XY = data.get_doublewarp_XY_data( **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan
    # fit
    LL_training, LL_testing = [], []
    for s in progress.dots(cv_slices, 'fitting cv'):
        m = c.get_warp_asd_model(XY, testing_proportion=0.2)
        m.training_slices = s.training_slices
        m.testing_slices = s.testing_slices
        m.solve(verbose=False)
        LL_training.append( m.posterior.LL_training_per_observation * 200 )
        LL_testing.append( m.posterior.LL_testing_per_observation * 200 )
        # check if done
        if c.can_load_attribute(attr):
            return
    LL_training = A(LL_training)
    LL_testing = A(LL_testing)
    # save
    results = Bunch({'LL_training':LL_training, 'LL_testing':LL_testing})
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, results )
    c.save_attribute(attr)







"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    for inh_speed in ['both']: #['slow']: #, 'fast', 'both']:
        for o in odours: #['baseline']: #odours:
            for warp_mode in ['inh', None, 'sniff']:
                jobs.append( (calc_cv_warp, c_idx, o, inh_speed, warp_mode) )
            for warp_mode in [None]:
                jobs.append( (calc_cv_ridge_warp, c_idx, o, inh_speed, warp_mode) )
            jobs.append( (calc_cv_const, c_idx, o, inh_speed) )
            jobs.append( (calc_cv_trigger, c_idx, o, inh_speed, 4) )
            #jobs.append( (calc_cv_trigger, c_idx, o, inh_speed, 10) )
            jobs.append( (calc_cv_trigger, c_idx, o, inh_speed, 2) )
            jobs.append( (calc_cv_halfwarp, c_idx, o, inh_speed) )
            jobs.append( (calc_cv_doublewarp, c_idx, o, inh_speed) )

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
