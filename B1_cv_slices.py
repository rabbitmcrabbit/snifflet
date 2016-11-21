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


def prepare_slices( c_idx, odour_name, inh_speed, N_reps=10, check=False ):

    if odour_name == 'baseline':
        return prepare_baseline_slices( c_idx, inh_speed, N_reps=N_reps, check=check )

    # check if it is done 
    c = cells[c_idx]
    attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True

    # GET DATA

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
        cv_slices = None
        if c.can_load_attribute(attr):
            return
        setattr( c, attr, cv_slices )
        c.save_attribute( attr, overwrite=False )
        return
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
    # standard warp data
    XY = data.get_warp_XY_data( **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan

    # standard warp model
    cv_slices = []
    for _ in range(10):
        m = c.get_warp_asd_model(XY, testing_proportion=0.2)
        cv_slices.append( 
                Bunch({'training_slices':m.training_slices, 'testing_slices':m.testing_slices}) )

    # save
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, cv_slices )
    c.save_attribute( attr, overwrite=False )


def prepare_baseline_slices( c_idx, inh_speed, N_reps=10, check=False ):

    odour_name = 'baseline'

    # check if it is done 
    c = cells[c_idx]
    attr = 'cv_slices_%s_%s' % (inh_speed, odour_name)
    if check:
        # is it done already
        if c.can_load_attribute(attr):
            return True
        # can it be done
        warp_attr = 'warp_%s_%s' % (inh_speed, odour_name)
        if c.can_load_attribute(warp_attr):
            return False
        return True

    # GET DATA

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
        cv_slices = None
        if c.can_load_attribute(attr):
            return
        setattr( c, attr, cv_slices )
        c.save_attribute( attr, overwrite=False )
        return
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
    # standard warp data
    XY = data.get_warp_XY_data( **kw[inh_speed] )
    XY.Y__nt[ :, ~to_keep__t ] = np.nan

    # standard warp model
    cv_slices = []
    for _ in range(10):
        m = c.get_warp_asd_model(XY, testing_proportion=0.2)
        cv_slices.append( 
                Bunch({'training_slices':m.training_slices, 'testing_slices':m.testing_slices}) )

    # save
    if c.can_load_attribute(attr):
        return
    setattr( c, attr, cv_slices )
    c.save_attribute( attr, overwrite=False )


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
            jobs.append( (prepare_slices, c_idx, o, inh_speed) )

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
