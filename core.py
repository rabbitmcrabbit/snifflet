# load modules
from helpers import *
from scipy.io import loadmat as loadmat_original
import glm

def loadmat( filename, **kw ):
    return loadmat_original( 
            filename, struct_as_record=False, squeeze_me=True, **kw) 

root_dir = 'data'
computations_dir = os.path.join( root_dir, 'computations' )


"""
===================
Raw data structures
===================

These are for importing directly from the .mat files.

"""

attrs_to_rename = {
        'sniffFlow' : 'flow',
        'sniffPhase' : 'sniff_phase_old',
        'odorConc' : 'odour_conc',
        'odorName' : 'odour_name',
        'odorTimes' : 'odour_times',
        'sniffZeroTimes' : 'sniff_zero_times',
        'trialId' : 'trial_idx',
        'cellId' : 'cell_idx',
        'start' : 't0',
        'clu':'cluster',
        'laserAmp':'laser_amp',
        'laserPower':'laser_power',
        'laserTimes':'laser_times'
        }

attrs_to_delete = ['sniff_zero_times', 'sniff_phase_old']


class MatFile( AutoCR ):

    """ Superclass for loading mat files. Easier to interface. """

    def __init__( self, raw_mat, empty_copy=False, **kw ):
        # is this an empty copy
        if empty_copy:
            return
        # import
        d = self.__dict__
        d.update( **raw_mat.__dict__ )
        del self._fieldnames
        # rename fields
        """
        for k in ['unitId', 'cellId', 'uId', 'Id', 'id']:
            if k in d.keys():
                print k
                tracer()
        """
        for k in d.keys():
            if k in attrs_to_rename.keys():
                k2 = attrs_to_rename[k]
                d[k2] = d[k]
                del d[k]
        # remove any old attributes
        for k in attrs_to_delete:
            try:
                del d[k]
            except KeyError:
                pass
        # any post processing
        self.__init_post__( **kw )

    def __init_post__( self, **kw ):
        pass


class RawTrial( MatFile ):

    """ A trial """

    def __init_post__( self, **kw ):
        # bugfix with sniff_phase
        self.flow = self.flow.astype(float)
        self._fix_bad_flows()
        if len( self.flow ) > 0:
            self._refit_sniff_phase()
        else:
            pass
        # check odour name
        self.odour_name = self.odour_name.replace('-', '_')


    def _fix_bad_flows( self, show_plot=False ):
        if np.abs( np.median( self.flow ) ) > 100:
            if show_plot:
                plt.plot( self.flow )
                plt.plot( 0*self.flow, 'k-' )
                plt.plot( 0*self.flow + np.median(self.flow), 'r--' )
                plt.gca().set_title( self.trial_idx )
                plt.show()
            self.flow -= np.median( self.flow )
            #tracer()

    def _refit_sniff_phase( self, threshold=None, prefilter=True, filter_w_ms=2,
            filter_zero_mean=False, onset_scan=np.arange(-3, 3), offset_scan=np.arange(-5, 1),
            force_merge=None, force_delete=None ) :
        """ Improved sniff phase detection. """
        # find threshold crossings
        x = self.flow
        if prefilter:
            if filter_zero_mean != False:
                w = float(filter_zero_mean)
                f = np.arange(-w*3, w*3.01)
                f = np.exp((-f/w)**2)
                f -= f.mean()
                f /= f.std()
                # filter the flow
                x2 = np.convolve(x, f, 'same')
                x = self.flow - x2
            # create filter
            w = float(filter_w_ms)
            f = np.arange(-w*3, w*3.01)
            f = np.exp((-f/w)**2)
            # filter the flow
            x2 = np.convolve(x, f, 'same')
            x2 *= np.std(x) / np.std(x2)
            x = x2
        # find threshold crossings
        if threshold is None:
            threshold = 0.2
        threshold = np.min(x) * float(threshold)
        up_crossings = (
                (x[:-1] < threshold) & (x[1:] >= threshold) ).nonzero()[0]
        down_crossings = (
                (x[:-1] >= threshold) & (x[1:] < threshold) ).nonzero()[0]
        # look back from each down crossing for inhalation onset
        onsets = np.zeros_like( down_crossings )
        for i in range(len(onsets)):
            local_t = down_crossings[i] + onset_scan
            try:
                local_x = x[ local_t ]
            except IndexError:
                onsets[i] = -1
                continue
            slope, intercept, _, _, pval = linregress( local_t, local_x )
            if slope != 0:
                onsets[i] = int(np.floor( -intercept/slope ))
            else:
                onsets[i] = -1
                continue
            if (onsets[i] >= len(x)) or (onsets[i] < 0):
                onsets[i] = -1
                continue
            # step forward if necessary
            if self.flow[ onsets[i] ] > 0:
                new_pos = (self.flow < 0).nonzero()[0]
                try:
                    new_pos = new_pos[ new_pos > onsets[i] ][0]
                    if np.abs(self.flow[new_pos-1]) < np.abs(self.flow[new_pos]):
                        new_pos -= 1
                except IndexError:
                    continue
                onsets[i] = new_pos
        onsets = onsets[ onsets < len(x) - 5 ]
        onsets = onsets[ onsets >= 5 ]
        onsets = np.unique(onsets)
        # look forward from each up crossing for inhalation offset
        offsets = np.zeros_like( up_crossings )
        for i in range(len(offsets)):
            local_t = up_crossings[i] + offset_scan
            local_x = x[ local_t ]
            slope, intercept = linregress( local_t, local_x )[:2]
            if slope != 0:
                offsets[i] = int(np.floor( -intercept/slope ))
            else:
                offsets[i] = -1
        offsets = offsets[ offsets < len(x) ]
        offsets = offsets[ offsets > 0 ]
        # if there is a stimulus, events have to occur while the stimulus is on
        """
        if hasattr( self, 'stim' ):
            if self.stim.mean() > 0.01:
                onsets = onsets[ self.stim[onsets].astype(bool) ]
                offsets = offsets[ self.stim[offsets].astype(bool) ]
        """
        # first offset for each onset
        offsets = A([ 
            offsets[offsets > o].min() for o in onsets if (offsets > o).any() ])
        # only full inhalations
        offsets = offsets[:len(onsets)]
        if len(onsets) > len(offsets) + 1:
            onsets = onsets[ :len(offsets)+1 ]
        # if two onsets share an offset
        if len(np.unique(offsets)) != len(offsets):
            to_keep = []
            for i in range(len(offsets)):
                if not offsets[i] in offsets[i+1:]:
                    to_keep.append(i)
            onsets = onsets[ to_keep ]
            offsets = offsets[ to_keep ]
        # merge and delete, if requested
        onsets, offsets = list(onsets), list(offsets)
        if force_merge is not None:
            for m in force_merge:
                for mi in np.sort(m[1:])[::-1]:
                    onsets.pop(mi)
                for mi in np.sort(m[:-1])[::-1]:
                    offsets.pop(mi)
        if force_delete is not None:
            for d in np.sort(force_delete)[::-1]:
                onsets.pop(d)
                offsets.pop(d)
        self._inhalation_onsets = onsets
        self._inhalation_offsets = offsets
        self._build_inhalations()

    @property
    def inhalation_durations( self ):
        return np.diff( self.inhalations )[:, 0]

    def _add_inhalation( self, t0, t1 ):
        self._inhalation_onsets = conc([ [t0], self._inhalation_onsets ])
        self._inhalation_offsets = conc([ [t1], self._inhalation_offsets ])
        self._inhalation_onsets = np.sort( self._inhalation_onsets )
        self._inhalation_offsets = np.sort( self._inhalation_offsets )
        self._build_inhalations()

    def _del_inhalation( self, idx ):
        del self._inhalation_onsets[idx]
        del self._inhalation_offsets[idx]
        self._build_inhalations()

    def _merge_inhalations( self, idx_start, idx_end ):
        assert idx_end > idx_start
        assert idx_start >= 0
        assert idx_end < len(self.inhalations)
        onsets, offsets = A(self._inhalation_onsets), A(self._inhalation_offsets)
        onsets = conc([ onsets[:idx_start+1], onsets[idx_end+1:] ])
        offsets = conc([ offsets[:idx_start], offsets[idx_end:] ])
        self._inhalation_onsets = onsets
        self._inhalation_offsets = offsets
        self._build_inhalations()

    def _build_inhalations( self ):
        """ Helper function. """
        if not isinstance(self._inhalation_onsets, list):
            self._inhalation_onsets = [i for i in self._inhalation_onsets]
        if not isinstance(self._inhalation_offsets, list):
            self._inhalation_offsets = [i for i in self._inhalation_offsets]
        onsets, offsets = self._inhalation_onsets, self._inhalation_offsets
        # save full sniff cycles
        self.full_sniff_cycles = zip( onsets[:-1], onsets[1:] )
        # save inhalations
        self.inhalations = zip( onsets, offsets )
        i = self.inhalations
        if len(i) > 0:
            if ~np.all( A(i)[:, 1] - A(i)[:, 0] > 0 ):
                tracer()
        self.sniff_phase = -np.ones_like( self.flow )
        for z in zip( onsets, offsets ):
            self.sniff_phase[ z[0]:z[1] ] = 1
        if len( self.full_sniff_cycles ) > len( self.inhalations ):
            tracer()

    def _automerge_inhalations( self, threshold_ms=30 ):
        onsets, offsets = A(self._inhalation_onsets), A(self._inhalation_offsets)
        dt = onsets[1:len(offsets)] - offsets[:-1]
        if len(dt) == 0:
            return
        elif dt.min() <= threshold_ms:
            to_merge = (dt <= threshold_ms).nonzero()[0]
            onsets[ to_merge+1 ] = -1
            offsets[ to_merge ] = -1
            onsets = onsets[onsets >= 0]
            offsets = offsets[offsets >= 0]
            self._inhalation_onsets = onsets
            self._inhalation_offsets = offsets
            self._build_inhalations()

    def _autodelete_inhalations( self, threshold_ms=30 ):
        idxs = (self.inhalation_durations <= threshold_ms).nonzero()[0]
        for idx in idxs[::-1]:
            del self._inhalation_onsets[idx]
            del self._inhalation_offsets[idx]
        self._build_inhalations()


    def __repr__( self ):
        return '<Raw Trial: %s>' % self.trial_idx

    @property
    def t0_sec( self ):
        return self.t0 / 1000.

    @property
    def time_ms( self ):
        return self.t0 + np.arange(len(self.flow))

    @property
    def time_sec( self ):
        return self.t0_sec + np.arange(len(self.flow)) / 1000.



class RawTrialList( AutoCR ):

    """ A collection of trials """

    def __init__( self, raw_trials ):
        if np.iterable(raw_trials):
            self.trials = [ RawTrial(t) for t in raw_trials ]
        else:
            self.trials = [ RawTrial(raw_trials) ]

    def __getitem__( self, i ):
        if isinstance(i, str) or isinstance(i, unicode):
            idx = [ tr.trial_idx for tr in self.trials ].index(i)
            return self.trials[idx]
        else:
            return self.trials.__getitem__( i )

    def __repr__( self ):
        return '<%d RawTrialList>' % len(self)

    def __len__( self ):
        return len( self.trials )

    def __add__( self, r2 ):
        if r2 == []:
            return self
        r_new = RawTrialList([])
        r_new.trials = self.trials + r2.trials
        return r_new

    def __radd__( r2, self ):
        if r2 == []:
            return self
        else:
            return r2.__add__( self )


    """ Properties """

    @property
    def t0_sec( self ):
        vals = np.empty( len(self) )
        for i, t in enumerate( self.trials ):
            vals[i] = t.t0_sec
        return vals

    @property
    def trial_idxs( self ):
        return [ t.trial_idx for t in self.trials ]

    """ Filtering """

    def filter( self, bool_array ):
        """ Return RawTrialList, trials subselected by the bool_array. """
        t = RawTrialList([])
        t.trials = list( A(self.trials)[bool_array] )
        return t

    def filter_func( self, func ):
        b = [ func(t) for t in self.trials ]
        return self.filter(b)


class RawSpikes( MatFile ):

    """ Spikes from a cell """

    def __init_post__( self, **kw ):
        # remove unwanted attributes
        for k in ['x', 'y', 't1', 't2', 't']:
            try:
                delattr( self, k )
            except AttributeError:
                pass
        # clean up metadata
        if hasattr( self, 'cell' ):
            self.metadata = m = CellMetadata(self.cell)
            del self.cell
            self.cell_idx = m.Id
            del m.Id
            # check mouse / sess / rec / sessCell
            mouse, sess, rec, sessCell = self.cell_idx.split('_')
            assert mouse == m.mouse
            assert int(sess) == m.sess
            assert rec == m.rec
            assert int(sessCell) == m.sessCell
            del m.mouse, m.sess, m.rec, m.sessCell
            assert m.uId.split('_')[0] == mouse
            assert int(m.uId.split('_')[1]) == int(sess)
            assert int(m.uId.split('_')[2]) == int(sessCell)
            del m.uId
        # check cell_idx in the right format
        c = self.cell_idx.split('_')
        if ( len(c[1]) != 3 ) or ( len(c[3]) != 3 ):
            self.cell_idx = '%s_%03d_%s_%03d' % ( c[0], int(c[1]), c[2], int(c[3]) )
        # more checks for mouse / sess / rec / sessCell
        mouse, sess, rec, sessCell = self.cell_idx.split('_')
        if self.__dict__.has_key('mouse'):
            assert mouse == self.__dict__['mouse']
            del self.__dict__['mouse']
        if self.__dict__.has_key('sess'):
            assert int(sess) == self.__dict__['sess']
            del self.__dict__['sess']
        if self.__dict__.has_key('rec'):
            assert rec == self.__dict__['rec']
            del self.__dict__['rec']
        if self.__dict__.has_key('sessCell'):
            assert int(sessCell) == self.__dict__['sessCell']
            del self.__dict__['sessCell']
        # check for uid
        if self.__dict__.has_key('uid'):
            mouse2, sess2, sessCell2 = self.__dict__['uid'].split('_')
            assert mouse2 == mouse
            assert int(sess) == int(sess2)
            assert int(sessCell2) == int(sessCell)
            del self.__dict__['uid']
        self.cell_idx = str( self.cell_idx )
        # single trial only
        if not isinstance( self.trial_idx, np.ndarray ):
            self.trial_idx = A([ self.trial_idx ])
            self.spikes = A([ self.spikes ])
            if hasattr( self, 't0' ):
                self.t0 = A([ self.t0 ])
        # fix odour names
        self.odour_name = A([ o.replace('-','_') for o in self.odour_name ])

    @property
    def N_trials( self ):
        return len(self.trial_idx)

    @property
    def mouse( self ):
        return self.cell_idx.split('_')[0]

    @property
    def session( self ):
        return int( self.cell_idx.split('_')[1] )

    @property
    def rec( self ):
        return self.cell_idx.split('_')[2]

    @property
    def session_cell( self ):
        return int( self.cell_idx.split('_')[3] )

    def __repr__( self ):
        return '<RawSpikes: %d trials>' % self.N_trials


class CellMetadata( MatFile ):

    """ Metadata storage. """

    def __repr__( self ):
        return '<Metadata>'


"""
====================
Processed data files
====================

These combine information from the RawTrials and RawSpikes into handy formats.

"""

class Data( AutoCR ):

    """ Data structure for trials and spikes combined. """

    @property
    def N_cells( self ):
        return self.spikes__nat.shape[0]

    @property
    def N_trials( self ):
        return self.spikes__nat.shape[1]

    @property
    def T( self ):
        if hasattr( self, 'spikes__nat' ):
            return min( self.flow__at.shape[-1], self.spikes__nat.shape[-1] )
        else:
            return self.flow__at.shape[-1]

    @property
    def N( self ):
        return self.N_cells

    @property
    def A( self ):
        return self.N_trials

    @property
    def N_inhalations( self ):
        return sum([len(i) for i in self.inhalations__ai])

    @property
    def N_sniffs( self ):
        return sum([len(i) for i in self.full_sniff_cycles__ai])

    def copy( self ):
        d = self.__class__(None, None, empty_copy=True)
        d.__dict__ = self.__dict__.copy()
        return d

    @property
    def inhalation_durations__ai( self ):
        return [list(np.diff(inh)[:, 0]) for inh in self.inhalations__ai]

    """ Filtering """

    def filter_trials( self, bool_array ):
        assert len(bool_array) == self.N_trials
        d = new_data = self.__class__( [], [], empty_copy=True )
        d.__dict__ = self.__dict__.copy()
        # filter keys
        for k in d.__dict__.keys():
            if ( k.count('__')==1 ) and ( k.split('__')[1].count('a') == 1 ):
                kshape = k.split('__')[1]
                # if list, temporarily convert to numpy array
                v = getattr( d, k )
                is_list = isinstance( v, list )
                if is_list:
                    v = A(v)
                # filter
                if kshape[0] == 'a':
                    v = v[bool_array, ...]
                elif kshape[1] == 'a':
                    v = v[:, bool_array, ...]
                elif kshape[2] == 'a':
                    v = v[:, :, bool_array, ...]
                else:
                    raise AttributeError(
                            'cannot parse attribute `%s` for filtering' % k )
                # recast as list if necessary, save
                if is_list:
                    v = [vi for vi in v]
                setattr( d, k, v )
        # return
        return d

    def filter_inhalations( self, func ):
        """ Return a copy, where inhalations satisfying func are kept."""
        d = self.copy()
        d.inhalations__ai = [[i for i in inh if func(i)] for inh in self.inhalations__ai]
        return d

    """ Preparing X/Y data for GLMs """

    def get_trigger_XY_data( self, binsize_ms=5, D_trigger=100, N_triggers=4, **kw ):
        """ Gets X and Y data ready for fitting trigger-based GLMs. """
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
        # location of triggers
        triggers__rat = np.zeros( (N_triggers, N_trials, T), dtype=bool )
        for a in range( N_trials ):
            inhs = self.inhalations__ai[a]
            for i in range(len(inhs)):
                idxs = np.linspace( inhs[i][0], inhs[i][1], N_triggers ).astype(int)
                triggers__rat[ :, a, idxs ] = np.eye(N_triggers, dtype=bool)
        # downsample
        triggers__rat = triggers__rat[:, :, :T_total].reshape(
                N_triggers, N_trials, T_ds, binsize_ms).any(axis=-1)

        # create trigger regressor
        X_trigger__td = np.zeros( (N_trials*T_ds, D_trigger*N_triggers), dtype=bool )
        for tr in range( N_triggers ):
            X_trigger__td[ :, (D_trigger*tr):(D_trigger*(tr+1)) ] = glm.construct_X__ith( 
                    triggers__rat[tr], D_trigger ).reshape( (N_trials*T_ds, D_trigger) )

        # create spike history regressor
        X_history = self._get_X_history( Y__nat, binsize_ms, **kw )
        # return
        results = Bunch({
            'X_stim__td' : X_trigger__td,
            'Y__nt' : Y__nt,
            'binsize_ms' : binsize_ms,
            'N_triggers' : N_triggers, 
            'D_trigger' : D_trigger })
        results.update( **X_history )
        return results

    def get_flow_XY_data( self, binsize_ms=5, D_flow=120, **kw ):
        """ Gets X and Y data ready for fitting flow-based GLMs. """
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
        # flow during inhalations
        flow__at = self.flow__at[:, :T_total]
        X_flow__t = -flow__at.flatten()
        X_flow__t *= ( self.sniff_phase__at[:, :T_total].flatten() == 1 )
        X_flow__t[ X_flow__t < 0 ] = 0
        # downsample
        X_flow__t = X_flow__t.reshape( (T_ds * N_trials, binsize_ms) ).mean(axis=1)
        X_flow__t /= X_flow__t.max()
        # add history
        X_flow__td = glm.construct_X__ith( X_flow__t, D_flow )
        # remove across-trial overflow of history
        X_trial__at = np.zeros( (N_trials, T_ds) )
        for a in range(N_trials):
            X_trial__at[a, :] = a + 1
        X_trial__t = X_trial__at.flatten()
        X_trial__td = glm.construct_X__ith( X_trial__t, D_flow )
        to_zero = (X_trial__td != X_trial__td[:, -1][:, None])
        X_flow__td[ to_zero ] = 0
        # create spike history regressor
        X_history = self._get_X_history( Y__nat, binsize_ms, **kw )
        # return
        results = Bunch({
            'X_stim__td' : X_flow__td,
            'Y__nt' : Y__nt, 
            'binsize_ms' : binsize_ms, 
            'D_flow' : D_flow })
        results.update( **X_history )
        return results

    def get_warp_XY_data( self, binsize_ms=5, D_warp_total=160, D_warp_inh=40, warp_mode='inh', **kw ):
        """ Gets X and Y data ready for fitting warp-based GLMs. """
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
        def dilation_matrix( dilation_factor, max_T=1000 ):
            return A([ 
                    affine_transform( delta(i, max_T), A([dilation_factor]), order=1 ) 
                    for i in range(D_warp_total) ]).T
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
                if warp_mode == 'inh':
                    dilation_factor = D_warp_inh / float(inh_dur_smp)
                elif warp_mode == None:
                    dilation_factor = 1.
                elif warp_mode == 'sniff':
                    dilation_factor = D_warp_total / float(sniff_dur_smp)
                else:
                    raise ValueError('unknown warp_mode')
                this_D = dilation_matrix( dilation_factor, max_T=max_T )
                warp__atd[ a, t0_smp : t0_smp + sniff_dur_smp, : ] = this_D[ :sniff_dur_smp, : ]
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

    def get_halfwarp_XY_data( self, binsize_ms=5, D_warp_total=160, D_warp_inh=40, **kw ):
        """ Gets X and Y data ready for fitting warp-based GLMs. """
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
                dilation_factor = D_warp_inh / float(inh_dur_smp)
                D1__td = dilation_matrix( dilation_factor, D_warp_inh, max_T=max_T )
                D2__td = dilation_matrix( 1., D_warp_total - D_warp_inh, max_T=max_T )
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

    def get_doublewarp_XY_data( self, binsize_ms=5, D_warp_total=160, D_warp_inh=40, **kw ):
        """ Gets X and Y data ready for fitting warp-based GLMs. """
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
                dilation_factor_2 = (D_warp_total - D_warp_inh) / max([float(sniff_dur_smp - inh_dur_smp), 1])
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

    def _get_X_history( self, Y__nat, binsize_ms, D_spike_history=10, 
            use_smooth_history_basis=True, smooth_history_max_tau_ms=100 ):
        """ Helper function. """
        results = Bunch({
            'D_spike_history' : D_spike_history,
            'use_smooth_history_basis' : use_smooth_history_basis
            })
        T_ds = self.T / binsize_ms
        if D_spike_history == 0:
            results['X_history__ntd'] = A([[[]]])
            return results
        # what to fill in the history with when there's no data
        ymean_n = Y__nat.mean(axis=2).mean(axis=1)
        ymean_nat = ymean_n[:, None, None]
        # two cases
        if use_smooth_history_basis:
            # how big to make the full thing
            max_tau_smp = smooth_history_max_tau_ms / binsize_ms
            max_D = max_tau_smp * 4
            # construct the basis
            def basis( tau ):
                b = np.exp( np.arange( -max_D, 0 ) / float(tau) )
                return b / np.sum(b)
            taus = np.logspace(0, np.log10(max_tau_smp), D_spike_history)
            B = A([ basis(tau) for tau in taus ]).T
            # make the history and project onto the basis
            X_history__natd = glm.construct_X__ith( Y__nat, max_D, wrap_value=ymean_nat )
            X_history__ntd = X_history__natd.reshape( 
                    (self.N_cells, self.N_trials*T_ds, max_D) ).dot(B)
            # save
            results['smooth_history_max_tau_ms'] = smooth_history_max_tau_ms 
            results['history_taus_smp'] = taus
            results['history_basis'] = B
        else:
            X_history__natd = glm.construct_X__ith( Y__nat, D_spike_history, wrap_value=ymean_nat )
            X_history__ntd = X_history__natd.reshape( 
                    (self.N_cells, self.N_trials*T_ds, D_spike_history) )
        # return
        results['X_history__ntd'] = X_history__ntd
        return results


class TrialData( Data ):

    """ Combined data format, specifically for odour data. """

    def __init__( self, trials, spikes_by_cell, empty_copy=False ):

        if empty_copy:
            return
            
        if not A([ss.N_trials == len(trials) for ss in spikes_by_cell]).all():
            #print 'wrong number of trials, fixing'
            # check that the spikes cover a subset of the trials
            assert np.all([ np.in1d(ss.trial_idx, trials.trial_idxs).all() for ss in spikes_by_cell ])
            # check that the spikes share the same trial structure
            for ss in spikes_by_cell:
                assert ss.N_trials == spikes_by_cell[0].N_trials
                assert np.all( ss.trial_idx == spikes_by_cell[0].trial_idx )
            # filter down
            trials = trials.filter(np.in1d(trials.trial_idxs, spikes_by_cell[0].trial_idx))


        try:
            # check that the number of trials is the same
            assert A([ss.N_trials == len(trials) for ss in spikes_by_cell]).all()
            # check that the trial data are the same
            for k in ['t0', 'odour_name', 'odour_conc', 'trial_idx']:
                assert (
                    A([ getattr(ss,k) for ss in spikes_by_cell ]) == 
                    A([ getattr(tt,k) for tt in trials ]) ).all()
            # check that the flow and spikes are the same duration
            flow_duration = np.unique([tt.flow.shape[0] for tt in trials])
            spikes_duration = A([ ss.spikes.shape[1] for ss in spikes_by_cell ])
            assert len(flow_duration) == 1
            assert ( spikes_duration == flow_duration[0] ).all()
        except AssertionError:
            print 
            print 'mis-match!'
            print 
            tracer()

        # only keep trials which have a sniff phase
        to_keep__a = A([ hasattr(t, 'sniff_phase') for t in trials ])
        if not to_keep__a.all():
            trials = A(trials.trials)[ to_keep__a ]

        # trial info
        self.flow__at = A([ t.flow for t in trials ])
        self.sniff_phase__at = A([ t.sniff_phase for t in trials ])
        self.full_sniff_cycles__ai = [ t.full_sniff_cycles for t in trials ]
        self.inhalations__ai = [ t.inhalations for t in trials ]
        self.t0__a = A([ t.t0 for t in trials ])
        self.trial_idx__a = A([ t.trial_idx for t in trials ])
        self.odour_conc__a = A([ t.odour_conc for t in trials ])
        self.odour_name__a = A([ t.odour_name for t in trials ])
        self.stim__at = A([ t.stim for t in trials ])

        # spikes
        """
        # this has been deprecated: the spikes are now aligned to trial onset
        self.inhalation_aligned_spikes__nat = A([ s.spikes for s in spikes_by_cell ])
        N, N_trials, T = self.inhalation_aligned_spikes__nat.shape
        self.spikes__nat = spikes__nat = np.zeros( (N, N_trials, self.T) )
        spikes__nat.fill(np.nan)
        for a in range(N_trials):
            if len( self.inhalations__ai[a] ) > 0:
                t0 = self.inhalations__ai[a][0][0]
                t_idx = np.arange(T) + (t0 - 200)
                tok = (t_idx >= 0) & (t_idx < self.T)
                t_idx = t_idx[tok]
                spikes__nat[:, a, t_idx] = (
                        self.inhalation_aligned_spikes__nat[:, a, tok] )
        """
        self.spikes__nat = A([ s.spikes for s in spikes_by_cell ])

        # only keep trials which have a sniff phase
        if not to_keep__a.all():
            #self.inhalation_aligned_spikes__nat = self.inhalation_aligned_spikes__nat[:, to_keep__a, :]
            self.spikes__nat = self.spikes__nat[ :, to_keep__a, : ]


    def __repr__( self ):
        return '<Trial Data: %d trials, %d cells>' % (
                self.N_trials, self.N_cells)

    @property
    def stim_t0__a( self ):
        """ Time of stimulus onset, relative to trial start. """
        N = self.N_trials
        stim_t0__a = np.zeros(N, dtype=int)
        for a in range(N):
            stim_t0__a[a] = self.stim__at[a].nonzero()[0][0]
        return stim_t0__a


class Sniffs( AutoCR ):

    """ Structure for sniff-aligned access to data. """

    def __init__( self, data, pre_buffer_ms=200, post_buffer_ms=200, 
            max_sniff_dur_ms=1000, empty_copy=False ):
        if empty_copy:
            return
        # save parameters
        self.pre_buffer_ms = pre_buffer_ms
        self.post_buffer_ms = post_buffer_ms
        self.max_sniff_dur_ms = max_sniff_dur_ms
        # sample resolution
        self.sampling_resolution_ms = 1
        # useful
        T = max_sniff_dur_ms + pre_buffer_ms + post_buffer_ms
        I = data.N_sniffs
        N = data.N_cells
        # containers
        self.flow__it = flow__it = np.empty( (I, T) )
        self.sniff_phase__it = sniff_phase__it = np.empty( (I, T) )
        self.spikes__nit = spikes__nit = np.empty( (N, I, T) )
        self.trial_num__i = trial_num__i = np.zeros( (I), dtype=np.uint16 )
        self.trial_idx__i = trial_idx__i = np.empty( (I), dtype=data.trial_idx__a.dtype )
        self.t0__i = t0__i = np.zeros( (I), dtype=np.uint16 )
        self.sniff_num_in_trial__i = sniff_num_in_trial__i = np.zeros( (I), dtype=np.uint16 )
        self.is_stim_on__i = is_stim_on__i = np.zeros( (I), dtype=bool )
        self.is_first_sniff__i = is_first_sniff__i = np.zeros( (I), dtype=bool )
        self.inhalations__i2 = inhalations__i2 = np.zeros( (I, 2), dtype=np.uint16 )
        self.full_sniff_cycles__i2 = full_sniff_cycles__i2 = np.zeros( (I, 2), dtype=np.uint16 )
        if hasattr( data, 'odour_conc__a' ):
            self.odour_conc__i = np.zeros( (I) )
            self.odour_name__i = np.empty( (I), dtype=data.odour_name__a.dtype )
        # empty containers
        flow__it.fill(np.nan)
        sniff_phase__it.fill(np.nan)
        spikes__nit.fill(np.nan)
        self.inhalations__i2[ :, 0 ] = pre_buffer_ms
        self.full_sniff_cycles__i2[ :, 0 ] = pre_buffer_ms
        # fill in containers
        j = 0
        for a in range(data.N_trials):
            stim_on_yet = False
            for i in range( len(data.full_sniff_cycles__ai[a]) ):
                # time range of the sniff
                t0, t1 = data.full_sniff_cycles__ai[a][i]
                dt = t1 - t0
                dt_inh = np.diff( data.inhalations__ai[a][i] )[0]
                if t1 - t0 > max_sniff_dur_ms:
                    t1 = t0 + max_sniff_dur_ms
                # buffer before and after
                t0 -= pre_buffer_ms
                t1 += post_buffer_ms
                # target indices
                u0, u1 = 0, t1-t0
                # correct for any overshoot beyond trial boundaries
                max_T = min( data.T, data.flow__at.shape[1], data.spikes__nat.shape[-1] )
                if t0 < 0:
                    u0 -= t0
                    t0 = 0
                if t1 >= data.T:
                    u1 -= (t1 - data.T)
                    t1 -= (t1 - data.T)
                # save into container
                try:
                    spikes__nit[ :, j, u0:u1 ] = data.spikes__nat[ :, a, t0:t1 ]
                except ValueError:
                    tracer()
                flow__it[ j, u0:u1 ] = data.flow__at[ a, t0:t1 ]
                sniff_phase__it[ j, u0:u1 ] = data.sniff_phase__at[ a, t0:t1 ]
                trial_num__i[ j ] = a
                trial_idx__i[ j ] = data.trial_idx__a[ a ]
                t0__i[ j ] = data.full_sniff_cycles__ai[a][i][0] - pre_buffer_ms
                sniff_num_in_trial__i[ j ] = i
                is_stim_on__i[ j ] = data.stim__at[a, t0 + pre_buffer_ms]
                if (not stim_on_yet) and is_stim_on__i[j]:
                    is_first_sniff__i[j] = True
                    stim_on_yet = True
                self.inhalations__i2[ j, 1 ] = pre_buffer_ms + dt_inh
                self.full_sniff_cycles__i2[ j, 1 ] = pre_buffer_ms + dt
                if hasattr( self, 'odour_conc__i' ):
                    self.odour_conc__i[ j ] = data.odour_conc__a[ a ]
                    self.odour_name__i[ j ] = data.odour_name__a[ a ]
                # next
                j += 1

    """ Properties """

    @property
    def t_ms( self ):
        t = np.arange( self.T ) * self.sampling_resolution_ms
        t -= self.pre_buffer_ms
        return t

    @property
    def sampling_frequency_Hz( self ):
        return 1000. / self.sampling_resolution_ms

    @property
    def T( self ):
        return self.flow__it.shape[1]

    @property
    def N_sniffs( self ):
        return self.flow__it.shape[0]

    @property
    def N_cells( self ):
        return self.spikes__nit.shape[0]

    @property
    def inhalation_durations__i( self ):
        return np.diff( self.inhalations__i2 )[:, 0]

    @property
    def sniff_durations__i( self ):
        return np.diff( self.full_sniff_cycles__i2 )[:, 0]

    @property
    def unpadded_sniff_flows__it( self ):
        flow = self.flow__it.copy()
        for i in range(self.N_sniffs):
            flow[ i, self.full_sniff_cycles__i2[i, 1]: ] = np.nan
        flow = flow[ :, self.pre_buffer_ms: ]
        return flow

    def __repr__( self ):
        return '<%s sniffs>' % self.N_sniffs

    """ Filtering """

    def filter_sniffs( self, bool_array ):
        assert len(bool_array) == self.N_sniffs
        s = new_sniffs = self.__class__( [], empty_copy=True )
        s.__dict__ = self.__dict__.copy()
        # filter keys
        for k in s.__dict__.keys():
            if ( k.count('__')==1 ) and ( k.split('__')[1].count('i') == 1 ):
                kshape = k.split('__')[1]
                if kshape[0] == 'i':
                    setattr( s, k, getattr(s,k)[bool_array, ...] )
                elif kshape[1] == 'i':
                    setattr( s, k, getattr(s,k)[:, bool_array, ...] )
                elif kshape[2] == 'i':
                    setattr( s, k, getattr(s,k)[:, :, bool_array, ...] )
                else:
                    raise AttributeError(
                            'cannot parse attribute `%s` for filtering' % k )
        # return
        return s

    @property
    def first_sniffs( self ):
        return self.filter_sniffs( self.is_first_sniff__i )

    def filter_by_odour_name( self, odour_name ):
        to_keep = self.odour_name__i == odour_name
        return self.filter_sniffs( to_keep )

    @property
    def filter_pre( self ):
        return self.filter_sniffs( self.is_stim_on__i == False )

    """ Sorting """

    @property
    def sorted_by_inhalation_duration( self ):
        order = np.argsort( self.inhalation_durations__i )
        return self.filter_sniffs( order )

    @property
    def sorted_by_sniff_duration( self ):
        order = np.argsort( self.sniff_durations__i )
        return self.filter_sniffs( order )

    """ Downsampling """

    def downsample( self, factor, attrs_to_sum=None ):
        # appropriate sizes
        factor = int( factor )
        old_T = self.T
        new_T = (self.T / factor) * factor
        # check input
        if attrs_to_sum is None:
            attrs_to_sum = []
        elif type( attrs_to_sum ) != list:
            raise TypeError('`attrs_to_sum` must be a list')
        attrs_to_sum.append('spikes')
        # make a copy
        s = new_sniffs = self.__class__( [], empty_copy=True )
        s.__dict__ = self.__dict__.copy()
        s.sampling_resolution_ms = s.sampling_resolution_ms * factor
        # filter keys
        for k in s.__dict__.keys():
            if ( k.count('__')==1 ) and ( k.split('__')[1].count('t') == 1 ):
                # must be the last index
                kprefix, kshape = k.split('__')
                if kshape[-1] != 't':
                    raise AttributeError(
                            'cannot parse attribute `%s` for downsampling' % k )
                # truncate
                v = getattr(s, k)
                if old_T != new_T:
                    v = v[ ..., :new_T ]
                # reshape
                old_shape = list(v.shape)
                new_shape = list(v.shape)[:-1]
                new_shape = tuple( new_shape + [ new_T/factor, factor ] )
                v = v.reshape( new_shape )
                # mean or sum
                if (k in attrs_to_sum) or (kprefix in attrs_to_sum):
                    v = np.nansum(v, axis=-1)
                else:
                    v = np.nanmean(v, axis=-1)
                # save
                setattr( s, k, v )
        # return
        return s

    """ PSTHs """

    @property
    def psth__nt( self ):
        y = np.nanmean(self.spikes__nit, axis=1) 
        return y * self.sampling_frequency_Hz 

    @property
    def psth_stderr__nt( self ):
        e__nt = np.nanstd( self.spikes__nit, axis=1 )
        N__nt = np.sum( np.isfinite( self.spikes__nit ), axis=1 )
        return e__nt / np.sqrt(N__nt) * self.sampling_frequency_Hz


"""
========================================
Data structures for Recordings and Cells
========================================
"""

class Recording( AutoCR ):

    """ Data structure for a recording session. """

    _short_sniff_cutoff_ms = 100
    _threshold_N_sniffs = 100

    def __init__( self, d ):
        self.__dict__.update( **d )

    def __repr__( self ):
        return '<Recording: %s>' % self.rec

    """
    # these deprecated as there's no baseline anymore
    # might need to reimplement somewhere

    @property
    def baseline_N_short_sniffs( self ):
        cutoff = self._short_sniff_cutoff_ms
        return (self.baseline_sniffs.inhalation_durations__i < cutoff).sum()

    @property
    def baseline_N_long_sniffs( self ):
        cutoff = self._short_sniff_cutoff_ms
        return (self.baseline_sniffs.inhalation_durations__i >= cutoff).sum()

    @property
    def baseline_has_short_sniffs( self ):
        return self.baseline_N_short_sniffs >= self._threshold_N_sniffs

    @property
    def baseline_has_long_sniffs( self ):
        return self.baseline_N_long_sniffs >= self._threshold_N_sniffs
    """


class Cell( AutoCR ):

    """ Data structure for a cell. """

    def __init__( self, recording, cell_idx, idx_in_recording ):
        self.recording = recording
        self.cell_idx = cell_idx
        self.idx_in_recording = idx_in_recording

    def __repr__( self ):
        return '<Cell: %s>' % self.cell_idx

    @property
    def spikes( self ):
        return self.recording.spikes_by_cell[ self.idx_in_recording ]

    @property
    def metadata( self ):
        return self.recording.spikes_by_cell[ self.idx_in_recording ].metadata

    @property
    def light( self ):
        return self.metadata.light

    @property
    def ipsi( self ):
        return self.metadata.ipsi

    @property
    def n( self ):
        return self.idx_in_recording

    """ GLMs """

    def get_glm_data( self, XY_data ):
        """ Create GLM Data object for this unit. 

        This requires a XY_data dictionary, which can be obtained by calling a
        Data object's `get_***_XY_data` method.

        """
        # data
        if XY_data.D_spike_history > 0:
            X__td = glm.combine_X( 
                    XY_data.X_stim__td, 
                    XY_data.X_history__ntd[ self.n ] )
        else:
            X__td = XY_data.X_stim__td.copy()
        Y__t = XY_data.Y__nt[ self.n ]
        # fix nans
        to_keep = np.isfinite( Y__t )
        X__td = X__td[to_keep, :]
        Y__t = Y__t[to_keep]
        X__td[ ~np.isfinite(X__td) ] = 0
        data = glm.Data( X__td, Y__t )
        # special attributes
        for k in XY_data.keys():
            if k.startswith('D_') or k.startswith('N_'):
                setattr( data, k, XY_data[k] )
        return data

    def get_trigger_asd_model( self, XY_data, testing_proportion=0, **kw ):
        """ Create TriggerASD model for this unit. 

        This requires a XY_data dictionary, which can be obtained by calling a 
        Data object's `get_trigger_XY_data` method.

        """
        data = self.get_glm_data( XY_data )
        model = glm.TriggerASD( data, testing_proportion=testing_proportion, **kw ) 
        return model

    def get_flow_asd_model( self, XY_data, testing_proportion=0, **kw ):
        """ Create FlowASD model for this unit. 

        This requires a XY_data dictionary, which can be obtained by calling a 
        Data object's `get_flow_XY_data` method.

        """
        data = self.get_glm_data( XY_data )
        model = glm.FlowASD( data, testing_proportion=testing_proportion, **kw ) 
        return model

    def get_warp_asd_model( self, XY_data, testing_proportion=0, **kw ):
        data = self.get_glm_data( XY_data )
        model = glm.WarpASD( data, testing_proportion=testing_proportion, **kw ) 
        return model

    def get_ridge_model( self, XY_data, testing_proportion=0, **kw ):
        data = self.get_glm_data( XY_data )
        model = glm.Ridge( data, testing_proportion=testing_proportion, **kw ) 
        return model


    """ Saving and loading attributes """

    def _attr_dir( self, attr ):
        return os.path.join( computations_dir, attr )

    def _attr_filename( self, attr ):
        save_prefix = self.cell_idx
        return os.path.join( self._attr_dir(attr), '%s.pickle' % save_prefix )

    def save_attribute( self, attr, overwrite=False ):
        """ Save a computed attribute to the `computations` subdirectory.

        The `overwrite` keyword determines what to do if the file already
        exists. If True, the file is overwritten. If False, an error is raised.
        If None, nothing happens.

        """
        # filename
        target_dir = self._attr_dir( attr )
        filename = self._attr_filename( attr )
        # check if it exists
        if os.path.exists( filename ):
            if overwrite is True:
                pass
            elif overwrite is False:
                raise IOError('file `%s` exists' % filename )
            elif overwrite is None:
                return
            else:
                raise ValueError('`overwrite` should be True/False/None')
        # prepare output
        output = {}
        output[attr] = getattr( self, attr )
        # does the directory exist
        if not os.path.exists( target_dir ):
            try:
                os.mkdir( target_dir )
            except OSError:
                if not os.path.exists( target_dir ):
                    os.mkdir( target_dir )
                pass
        # save
        cPickle.dump( output, open(filename, 'w'), 2 )

    def can_load_attribute( self, attr ):
        """ Whether an attribute has been computed and saved to disk. """
        filename = self._attr_filename( attr )
        return os.path.exists(filename)

    @property
    def available_attributes( self ):
        return sorted( [ a for a in os.listdir(computations_dir) 
            if self.can_load_attribute(a) ] )

    def load_attribute( self, attr, overwrite=True, ignore_if_missing=False ):
        """ Load a computed attribute. """
        if isinstance( attr, list ):
            for a in attr:
                self.load_attribute( a, overwrite=overwrite )
            return
        if hasattr( self, attr ) and not overwrite:
            return
        # filename
        filename = self._attr_filename( attr )
        # load
        try:
            d = cPickle.load( open( filename ) )
        except IOError as e:
            if ignore_if_missing:
                return
            else:
                raise e
        a = d[attr]
        # save
        setattr( self, attr, a )

"""
============
List classes
============

These are containers for Recordings and Cells.

"""


class ObjectList( AutoCR ):

    """ Superclass for Lists. """

    def __init__( self, data ):
        setattr( self, self._data_attr_name, data )

    @property
    def _data( self ):
        return getattr( self, self._data_attr_name )

    def __getitem__( self, *a, **kw ):
        return self._data.__getitem__( *a, **kw )

    def __repr__( self ):
        return '<%d %s>' % ( 
                len(self), 
                self.__class__.__name__.split('List')[0] + 's' )

    def __len__( self ):
        return len(self._data)

    def __iter__( self, *a, **kw ):
        return self._data.__iter__( *a, **kw )

    def append( self, *a, **kw ):
        return self._data.append( *a, **kw )


class RecordingList( ObjectList ):

    """ Collection of Recordings. """

    _data_attr_name = 'recordings'

    def __getitem__( self, i ):
        if isinstance(i, str) or isinstance(i, unicode):
            idx = [d.rec for d in self._data].index(i)
            return self._data[idx]
        else:
            return self._data.__getitem__( i )


class CellList( ObjectList ):

    """ Collection of Cells. """

    _data_attr_name = 'cells'

    """ Saving and loading attributes """

    def save_attribute( self, attr, **kw ):
        """ Save the computed attribute for all."""
        for c in self:
            c.save_attribute( attr, **kw )

    def load_attribute( self, attr, overwrite=True, ignore_if_missing=False ):
        """ Load the computed attribute for all."""
        for c in self:
            c.load_attribute( attr, overwrite=overwrite, 
                    ignore_if_missing=ignore_if_missing )

    def filter_by_attribute( self, attr, must_be_loaded=False ):
        """ Return a CellList with Cells for which attr is available. """
        if must_be_loaded:
            return self.__class__([ c for c in self._data if hasattr( c, attr ) ])
        else:
            return self.__class__([ c for c in self._data 
                if hasattr( c, attr ) or attr in c.available_attributes ])

    @property
    def filter_litral( self ):
        return self.__class__([ c for c in self._data if c.light == 1 ])

    @property
    def filter_other( self ):
        return self.__class__([ c for c in self._data if c.light == 0 ])

    @property
    def filter_weird( self ):
        return self.__class__([ c for c in self._data if c.light < 0 ])

    @property
    def filter_ipsi( self ):
        return self.__class__([ c for c in self._data if c.ipsi == 1 ])

    @property
    def filter_contra( self ):
        return self.__class__([ c for c in self._data if c.ipsi == 0 ])

    @property
    def filter_affinity_series( self ):
        return self.__class__([ c for c in self._data if not bool(c.recording.is_concentration_series) ])

    @property
    def filter_concentration_series( self ):
        return self.__class__([ c for c in self._data if c.recording.is_concentration_series ])

    def filter_by_bool_array( self, arr ):
        if len(arr) != len(self):
            raise ValueError('provide boolean array of correct length')
        if len(self) == 0:
            cells = A([])
        else:
            cells = A(self._data)[A(arr)]
        return self.__class__(cells)

    def filter_by_func( self, func ):
        arr = [func(c) for c in self]
        return self.filter_by_bool_array(arr)

    def __getitem__( self, *a, **kw ):
        if isinstance(a[0], str) or isinstance(a[0], unicode):
            # find idx
            matching_cells = [c for c in self if c.cell_idx == a[0]]
            if len(matching_cells) == 0:
                raise IndexError('no cell with cell_idx `%s`' % a[0])
            elif len(matching_cells) > 1:
                raise IndexError('more than one cell with cell_idx `%s`' % a[0])
            else:
                return matching_cells[0]
        else:
            return self._data.__getitem__( *a, **kw )



"""
==========================
Interface to model results
==========================
"""

class TriggerModel( AutoCR ):

    """ Interface to trigger model results. """

    _mu_length_ms = 800
    _cols = ['b', 'r', 'g', 'c', 'm', 'y']

    def __init__( self, fitted_model, durations=[60, 150], **model_kw ): 
        # save model bits
        p = fitted_model.posterior
        self.theta = p.theta
        self.v = p.v
        self.k = p.k__d
        self.Lambda = p.Lambda__dd
        self.evidence = p.evidence
        self.LL = p.LL_training_per_observation
        # save the keywords of the model
        self.__dict__.update( **model_kw )
        # durations of normalised inhalations to present
        self.durations = durations

    @property
    def _mu_length_smp( self ):
        return self._mu_length_ms / 5

    @cached
    def X__itd( durations, k, _mu_length_smp, D_trigger, N_triggers ):
        """ Design matrix """
        D = len( k )
        X = np.zeros((len(durations), _mu_length_smp, D))
        I = np.eye(D_trigger)[::-1, :]
        for i, d in enumerate(durations):
            idxs = np.round(np.linspace( 0, d / 5., N_triggers )).astype(int)
            for t in range( N_triggers ):
                X[ i, idxs[t]:idxs[t]+D_trigger, D_trigger*t:D_trigger*(t+1) ] = I
        X[:, :, -1] = 1
        return X

    @cached
    def logmu__it( k, X__itd ):
        return X__itd.dot( k )

    @cached
    def logmu_std__it( X__itd, Lambda ):
        return A([ np.sqrt(np.diag(X__td.dot(Lambda).dot(X__td.T))) for X__td in X__itd ])

    @property
    def mu__it( self ):
        return np.exp( self.logmu__it ) 

    @property
    def t_ms( self ):
        return np.arange( self._mu_length_smp ) * 5

    def plot_with_stderr( 
        self, ax=None, legend=False, xlim=[0, 500], 
        show_labels=True, show_ticks=True, return_ax=False, 
        show_legend=False, show_stderr=True, lw=4, alpha=None, **kw ):
        # figure
        if ax is None:
            fig, ax = plt.subplots( 1, 1, **kw )
        # check alpha
        if alpha is None:
            alpha = [1] * len( self.durations )
        elif len(alpha) != len(self.durations):
            raise ValueError('incorrect number of alpha values')
        # plot
        for i, d in enumerate( self.durations ):
            m = self.logmu__it[i]
            s = self.logmu_std__it[i]
            if show_stderr:
                ax.fill_between( 
                        self.t_ms, 200*np.exp(m-s), 200*np.exp(m+s), 
                        lw=0, facecolor=self._cols[i], alpha=0.3*alpha[i] )
            ax.plot( 
                    self.t_ms, 200*np.exp(m), color=self._cols[i], 
                    lw=lw, label='%d ms' % d, alpha=alpha[i] )
        # aesthetics
        ax.set_xlim( xlim )
        if show_labels:
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('spikes / s')
        if show_ticks:
            nice_spines(ax)
            yl = ax.get_ylim()
            ax.set_yticks(yl)
        else:
            nice_spines(ax, show_bottom=False, show_left=False)
            ax.set_xticks([])
            ax.set_yticks([])
        if show_legend:
            ax.legend(loc='upper right', fontsize=16, frameon=False)
        # return
        if return_ax:
            return ax


class FlowModel( TriggerModel ):

    _N_egs = 20

    def __init__( self, fit_results, cell ):
        self.__dict__.update(**fit_results)
        self.durations = [60, 150]
        # sniffs to extract example flows
        sniffs = cell.recording.baseline_sniffs
        # useful
        t0 = sniffs.pre_buffer_ms
        ds = fit_results.binsize_ms
        T = 800
        T_ds = T / ds
        # pick out the short inhalations
        idxs = np.argsort((sniffs.inhalation_durations__i - self.durations[0])**2)[:self._N_egs]
        flow_short__it = -sniffs.flow__it[idxs, t0:].copy()
        flow_short__it *= (sniffs.sniff_phase__it[idxs, t0:] == 1)
        flow_short__it[ ~np.isfinite(flow_short__it) ] = 0
        # remove the subsequent sniffs
        to_remove = ( np.diff(sniffs.sniff_phase__it[idxs, t0:]) > 0 ).nonzero()
        for j in range(len(to_remove[0])):
            flow_short__it[ to_remove[0][j], to_remove[1][j]: ] = 0
        # average
        flow_short__t = flow_short__it.mean(axis=0)
        # pick out the long inhalations
        idxs = np.argsort((sniffs.inhalation_durations__i - self.durations[1])**2)[:self._N_egs]
        flow_long__it = -sniffs.flow__it[idxs, t0:].copy()
        flow_long__it *= (sniffs.sniff_phase__it[idxs, t0:] == 1)
        flow_long__it[ ~np.isfinite(flow_long__it) ] = 0
        # remove the subsequent sniffs
        to_remove = ( np.diff(sniffs.sniff_phase__it[idxs, t0:]) > 0 ).nonzero()
        for j in range(len(to_remove[0])):
            flow_long__it[ to_remove[0][j], to_remove[1][j]: ] = 0
        # average
        flow_long__t = flow_long__it.mean(axis=0)
        # downsample
        flow_short__t = flow_short__t[:T].reshape((T_ds, ds)).mean(axis=1)
        flow_long__t = flow_long__t[:T].reshape((T_ds, ds)).mean(axis=1)
        # rescale
        scale_factor = (-sniffs.flow__it[ sniffs.sniff_phase__it == 1]).max()
        flow_short__t /= scale_factor
        flow_long__t /= scale_factor
        # add history
        X__it = A([ flow_short__t, flow_long__t ])
        X__itd = glm.construct_X__ith( X__it, fit_results.D_flow )
        # make shape right
        X__itd = np.concatenate([ X__itd, np.zeros((2, T_ds, 11)) ], axis=-1)
        X__itd[:, :, -1] = 1
        self._X__itd = X__itd

    @property
    def X__itd( self ):
        return self._X__itd

    @property
    def t_ms( self ):
        return np.arange(160) * 5


class BiFlowModel( FlowModel ):

    @property
    def X__itd( self ):
        D = self._X__itd.shape[-1] - 11
        X = np.concatenate([ self._X__itd[:, :, :-11] ]*2 + [self._X__itd[:, :, -11:]], axis=-1 )
        for i, dur in enumerate( self.durations ):
            if dur >= self.bisplit_ms:
                X[ i, :, :D ] = 0
            else:
                X[ i, :, D:2*D ] = 0
        return X


class WarpModel( TriggerModel ):

    @cached
    def X__itd( durations, D_warp_inh, D_warp_total, k ):
        def delta(i, max_T=D_warp_total):
            z = np.zeros(max_T, dtype=float)
            z[i] = 1
            return z
        def dilation_matrix( this_inh_duration_smp, max_T=D_warp_total ):
            dilation_factor = D_warp_inh / float(this_inh_duration_smp) 
            return A([ 
                    affine_transform( delta(i, max_T), A([dilation_factor]), order=1 ) 
                    for i in range(D_warp_total) ]).T 
        # warp matrix
        warp__itd = A([ dilation_matrix( d/5., D_warp_total ) for d in durations ])
        # make shape right
        D = len(k)
        Dz = ( len(durations), warp__itd.shape[1], D - warp__itd.shape[-1] )
        X__itd = np.concatenate([ warp__itd, np.zeros(Dz) ], axis=-1)
        X__itd[:, :, -1] = 1
        return X__itd

    @property
    def t_ms( self ):
        return np.arange(self.D_warp_total) * 5


class BiWarpModel( WarpModel ):

    @cached
    def X__itd( durations, D_warp_inh, D_warp_total, k, bisplit_ms ):
        def delta(i, max_T=D_warp_total):
            z = np.zeros(max_T, dtype=float)
            z[i] = 1
            return z
        def dilation_matrix( this_inh_duration_smp, max_T=D_warp_total ):
            dilation_factor = D_warp_inh / float(this_inh_duration_smp) 
            return A([ 
                    affine_transform( delta(i, max_T), A([dilation_factor]), order=1 ) 
                    for i in range(D_warp_total) ]).T 
        # warp matrix
        warp__itd = A([ dilation_matrix( d/5., D_warp_total ) for d in durations ])
        # double it and allocate
        warp__itd = np.concatenate([ warp__itd, warp__itd ], axis=-1)
        for i, dur in enumerate( durations ):
            if dur >= bisplit_ms:
                warp__itd[ i, :, :D_warp_total ] = 0
            else:
                warp__itd[ i, :, D_warp_total:2*D_warp_total ] = 0
        # make shape right
        D = len(k)
        Dz = ( len(durations), warp__itd.shape[1], D - warp__itd.shape[-1] )
        X__itd = np.concatenate([ warp__itd, np.zeros(Dz) ], axis=-1)
        X__itd[:, :, -1] = 1
        return X__itd
