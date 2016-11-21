from helpers import *
import glm
import core
from get_data import recordings, cells
from IPython.display import clear_output


# parameters
kw = {}
kw['trigger'] = {'binsize_ms':5, 'D_trigger':100, 'N_triggers':4}
kw['flow'] = {'binsize_ms':5, 'D_flow':120}
kw['biflow'] = {'binsize_ms':5, 'D_flow':120, 'bisplit_ms':100}
kw['warp'] = {'binsize_ms':5, 'D_warp_inh':40, 'D_warp_total':160}
kw['biwarp'] = {'binsize_ms':5, 'D_warp_inh':40, 'D_warp_total':160, 'bisplit_ms':100}

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

def fit_baseline( c_idx, model, check=False ):
    # check if it is done
    c = cells[c_idx]
    attr = 'cv_%s_asd_baseline' % model
    if check:
        return c.can_load_attribute( attr )
    # fit
    source_data = c.recording.baseline_data
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
    # load slices
    c.load_attribute('cv_slices_baseline')
    # extract the source data and fit the model
    results = Bunch()
    results.LL_training = np.empty(10)
    results.LL_testing = np.empty(10)
    # extract the source data
    
    XY_data = getattr( source_data, 'get_%s_XY_data' % model )( **kw[model] )
    try:
        for i in progress.numbers( range(10), header='-' ):
            m = getattr(c, 'get_%s_asd_model' % model )(XY_data)
            m.training_slices = c.cv_slices_baseline[i].training_slices
            m.testing_slices = c.cv_slices_baseline[i].testing_slices
            m.solve( verbose=True, callback=already_done_callback )
            results.LL_training[i] = m.posterior.LL_training_per_observation
            results.LL_testing[i] = m.posterior.LL_testing_per_observation
    # unless it has been done already
    except AlreadyDoneException:
        return
    # is it already done
    if c.can_load_attribute( attr ):
        print ' - already done'
        return
    # prepare output
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
    for model in ['trigger', 'flow', 'biflow', 'warp', 'biwarp']:
        jobs.append( (fit_baseline, c_idx, model) )

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
