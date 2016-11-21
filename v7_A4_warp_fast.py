from helpers import *
import glm
import core
from get_data import recordings, cells
from IPython.display import clear_output


# parameters
kw = {}
kw['warp'] = {'binsize_ms':5, 'D_warp_inh':40, 'D_warp_total':120, 'D_spike_history':0}

# odours to fit
odours = np.unique( np.concatenate([ 
    np.unique( r.trial_data.odour_name__a ) for r in recordings ]) )


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

def fit_odour_first_sniff( c_idx, model, odour_name, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = '%s_fast_%s_first_sniff' % (model, odour_name)
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
            (td.odour_conc__a > 0) &
            A([ len(sn)>0 for sn in td.full_sniff_cycles__ai ]) )
    source_data = td.filter_trials( trials_to_keep )
    if source_data.N_trials == 0:
        return
    # extract first sniff
    source_data.inhalations__ai = [
            [ inh[0] ] for inh in source_data.inhalations__ai ]
    source_data.full_sniff_cycles__ai = [
            [ sn[0] ] for sn in source_data.full_sniff_cycles__ai ]
    # only fast condition
    durations = np.diff(source_data.inhalations__ai).flatten()
    source_data = source_data.filter_trials( durations < 100 )
    # fit model
    return fit( c_idx, attr, model, source_data )

def fit_oc_first_sniff( c_idx, model, odour_conc_idx, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = '%s_fast_o%d_first_sniff' % (model, odour_conc_idx)
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
    # only fast condition
    durations = np.diff(source_data.inhalations__ai).flatten()
    source_data = source_data.filter_trials( durations < 100 )
    # fit model
    return fit( c_idx, attr, model, source_data )


"""
=============
main function
=============
"""

def fit( c_idx, attr, model, source_data, dry_run=False ):

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
    results.update( **kw[model] )
    try:
        # extract the source data
        XY_data = getattr( source_data, 'get_%s_XY_data' % model )( **kw[model] )
        # fit the model
        m = getattr( c, 'get_%s_asd_model' % model )( XY_data )
        m.solve( verbose=True, callback=already_done_callback )
    # unless it has been done already
    except AlreadyDoneException:
        return
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return
    # prepare output
    p = m.posterior
    results.update( **{
        'theta':p.theta, 'v':p.v, 'k':p.k__d, 'Lambda':p.Lambda__dd, 
        'mu__t':p.mu__t, 'evidence':p.evidence, 
        'LL':p.LL_training_per_observation }) 
    setattr( c, attr, results )
    # done
    c.save_attribute( attr, overwrite=False )

"""
====
jobs
====
"""

# job list
jobs = []
for c_idx in range( len(cells) ):
    model = 'warp'
    for o in odours:
        jobs.append( (fit_odour_first_sniff, c_idx, model, o) )
        pass
    for i in range(1, 7):
        jobs.append( (fit_oc_first_sniff, c_idx, model, i) )
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
