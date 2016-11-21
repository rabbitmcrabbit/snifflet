from helpers import *
import glm
import core
from get_data import recordings, cells
from IPython.display import clear_output


# parameters
kw = {'binsize_ms':5, 'D_trigger':100}

# odours to fit
odours = np.unique( np.concatenate([ 
    np.unique( r.trial_data.odour_name__a ) for r in recordings ]) )
#odours = ['2-hydroxyacetophenone', 'menthone']
#odours = ['menthone']


"""
================
helper functions
================
"""

def announce( c, attr ):
    s = '%s:  %s' % ( attr, c.__repr__() )
    print '='*len(s)
    print s

class AlreadyDoneException( Exception ):
    pass


"""
===============
specific models
===============
"""

def fit_asdN_baseline( c_idx, N_triggers, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_asdN_baseline_%02d' % N_triggers
    if check:
        return c.can_load_attribute( attr )
    # fit
    source_data = c.recording.baseline_data
    return fit_asdN( c_idx, attr, source_data, N_triggers )

def fit_asdN_all_odours_all_sniffs( c_idx, N_triggers, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_asdN_all_odours_all_sniffs_%02d' % N_triggers
    if check:
        return c.can_load_attribute( attr )
    # fit
    source_data = c.recording.trial_data
    return fit_asdN( c_idx, attr, source_data, N_triggers )

def fit_asdN_odour_all_sniffs( c_idx, odour_name, N_triggers, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_asdN_%s_all_sniffs_%02d' % (odour_name, N_triggers)
    if check:
        if c.can_load_attribute( attr ):
            return True
        trials_to_keep = c.recording.trial_data.odour_name__a == odour_name
        if trials_to_keep.sum() == 0:
            return True
        return False
    # filter to this odour
    td = c.recording.trial_data
    trials_to_keep = (
            (td.odour_name__a == odour_name) &
            A([ len(sn)>0 for sn in td.full_sniff_cycles__ai ]) )
    source_data = td.filter_trials( trials_to_keep )
    if source_data.N_trials == 0:
        return
    # fit model
    return fit_asdN( c_idx, attr, source_data, N_triggers )

def fit_asdN_odour_first_sniff( c_idx, odour_name, N_triggers, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_asdN_%s_first_sniff_%02d' % (odour_name, N_triggers)
    if check:
        if c.can_load_attribute( attr ):
            return True
        trials_to_keep = c.recording.trial_data.odour_name__a == odour_name
        if trials_to_keep.sum() == 0:
            return True
        return False
    # filter to this odour
    td = c.recording.trial_data
    trials_to_keep = (
            (td.odour_name__a == odour_name) &
            A([ len(sn)>0 for sn in td.full_sniff_cycles__ai ]) )
    source_data = td.filter_trials( trials_to_keep )
    if source_data.N_trials == 0:
        return
    # extract first sniff
    source_data.inhalations__ai = [
            [ inh[0] ] for inh in source_data.inhalations__ai ]
    source_data.full_sniff_cycles__ai = [
            [ sn[0] ] for sn in source_data.full_sniff_cycles__ai ]
    # fit model
    return fit_asdN( c_idx, attr, source_data, N_triggers )


"""
=============
main function
=============
"""

def fit_asdN( c_idx, attr, source_data, N_triggers, dry_run=False ):

    # get cell
    c = cells[c_idx]
    # announce
    announce( c, attr )
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return
    # skip if this is a dry run
    if dry_run:
        print ' - dry run; skipping'
        return
    # helper callback
    def already_done_callback():
        if c.can_load_attribute( attr ):
            print ' - already done (within callback)'
            raise AlreadyDoneException()
    # extract the source data and fit the model
    results = Bunch()
    results.update( **kw )
    # fit the model
    try:
        # extract the source data
        XY_data = source_data.get_trigger_XY_data( N_triggers=N_triggers, **kw )
        m = c.get_trigger_asd_model( XY_data )
        m.solve( verbose=True, callback=already_done_callback )
    except AlreadyDoneException:
        return
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return
    # prepare output
    p = m.posterior
    results.update(**{
        'theta':p.theta, 'v':p.v, 'k':p.k__d, 'Lambda':p.Lambda__dd, 
        'mu__t':p.mu__t, 'evidence':p.evidence, 'N_triggers':N_triggers,
        'LL':p.LL_training_per_observation }) 
    setattr( c, attr, results )
    # done
    c.save_attribute( attr, overwrite=False )

def fit_asdN_oc_first_sniff( c_idx, odour_conc_idx, N_triggers, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'trigger_asdN_o%d_first_sniff_%02d' % (odour_conc_idx, N_triggers)
    if check:
        # already done
        if c.can_load_attribute( attr ):
            return True
        # not a concentration series
        if not c.recording.is_concentration_series:
            return True
        # there are no trials for this condition
        trials_to_keep = c.recording.trial_data.odour_conc_idx__a == odour_conc_idx
        if trials_to_keep.sum() == 0:
            return True
        return False
    # filter to this odour
    td = c.recording.trial_data
    trials_to_keep = (
            (td.odour_conc_idx__a == odour_conc_idx) &
            A([ len(sn)>0 for sn in td.full_sniff_cycles__ai ]) )
    source_data = td.filter_trials( trials_to_keep )
    if source_data.N_trials == 0:
        return
    # extract first sniff
    source_data.inhalations__ai = [
            [ inh[0] ] for inh in source_data.inhalations__ai ]
    source_data.full_sniff_cycles__ai = [
            [ sn[0] ] for sn in source_data.full_sniff_cycles__ai ]
    # fit model
    return fit_asdN( c_idx, attr, source_data, N_triggers )

"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    for N in [10]:
        #jobs.append( (fit_asdN_baseline, c_idx, N) )
        #jobs.append( (fit_asdN_all_odours_all_sniffs, c_idx, N) )
        for o in odours:
            #jobs.append( (fit_asdN_odour_all_sniffs, c_idx, o, N) )
            jobs.append( (fit_asdN_odour_first_sniff, c_idx, o, N) )
        for i in range(1, 7):
            jobs.append( (fit_asdN_oc_first_sniff, c_idx, i, N) )
            pass

# check all the jobs
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
        func( *args )
        print ' '
        print ' '
