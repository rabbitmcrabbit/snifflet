import warnings
warnings.filterwarnings('ignore')

from helpers import *
from copy import deepcopy
import glm
import core


# file listing
root_dir = 'data'
source_dir = os.path.join( root_dir, 'raw' )
save_dir = root_dir
file_list = [f for f in os.listdir(source_dir) if f.count('awake') > 0]
session_file_list = list(set([ f.rsplit('_', 2)[0] for f in file_list ]))
prefix_list = list(set([ f.rsplit('_', 1)[0] for f in file_list ]))
rec_file_list = sorted([r for r in prefix_list if len(r.rsplit('_')[-1]) == 1])
cell_file_list = sorted([c for c in prefix_list if len(c.rsplit('_')[-1]) == 3])


"""
========================
load trials for each rec
========================
"""

trials_filename = os.path.join( save_dir, 'trials.pickle' )

if os.path.exists(trials_filename):
    for _ in progress.dots([1], 'loading trials from disk...'):
        trials = cPickle.load(open(trials_filename))
else:
    trials = Bunch()
    for r in progress.dots( rec_file_list, 'loading trials' ):        
        filename = join( source_dir, r + '_trial.mat' )
        trials[r] = core.RawTrialList( loadmat(filename)['trial'] )
        trials[r].trials = [ t for t in trials[r].trials if t.odour_name != u'none' ]
        trials[r].trials = [ t for t in trials[r].trials if t.odour_name != u'not_an_event' ]
        
    # save
    for _ in progress.dots([1], 'saving trials to disk'):
        cPickle.dump( trials, open(trials_filename, 'w'), 2 )        
        #pass



"""
==================
Fix problem trials
==================
"""

def find_problem_trials():

    all_trials = conc([v.trials for v in trials.values()])
    problems = []
    for t in all_trials:
        if len(t.flow) == 0:
            continue
        if hasattr(t, 'odour_conc'):
            if t.odour_conc == -1:
                continue
        # at least one inhalation
        if len(t.inhalations) == 0:
            problems.append( t )
            continue
        # shortest inhalation
        if np.diff(A(t.inhalations))[:, 0].min() <= 30:
            problems.append(t)
            continue
        # mean above zero
        if A([ t.flow[slice(*i)].mean() for i in t.inhalations ]).max() > 0:
            problems.append(t)
            continue
        # time between inhalations
        dt = A(t.inhalations)[1:, 0] - A(t.inhalations)[:-1, 1]
        if len(dt)>0 and dt.min() <= 30:
            problems.append(t)
            continue
        pass
    
    return problems

# first pass
problems = find_problem_trials()
for t in problems:
    t._refit_sniff_phase( threshold=0.05 )

# second pass
def get_trial(trial_idx):
    rec = trials[ trial_idx.rpartition('_')[0] ]
    return rec[trial_idx]

# list of fixes
t = get_trial('ZKawakeM72_027_a_883007')
t._refit_sniff_phase( threshold=0.1 )
t._automerge_inhalations()
t = get_trial('KPawakeM72_817_f_2961543')
t._refit_sniff_phase( threshold=0.1 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_029_c_753015')
t._refit_sniff_phase( threshold=0.1 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_029_c_2148985')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('ZKawakeM72_029_c_2224019')
t._refit_sniff_phase( threshold=0.2 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_029_c_2406124')
t._refit_sniff_phase( threshold=0.2 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_029_c_2616865')
t._refit_sniff_phase( threshold=0.2 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_021_a_2115661')
t._refit_sniff_phase( threshold=0.2 )
t._del_inhalation(2)
t = get_trial('ZKawakeM72_022_e_3804837')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_45168')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_217691')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_262298')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_349136')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_489855')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_537872')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_917723')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_1504607')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_016_a_3330231')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_021_b_979587')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('KPawakeM72_021_b_1333897')
t._refit_sniff_phase( )
t._automerge_inhalations()
t = get_trial('KPawakeM72_021_b_1480003')
t._refit_sniff_phase( )
t._del_inhalation(1)
t._add_inhalation(420, 482)
t = get_trial('KPawakeM72_021_b_2496054')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_023_a_4119560')
t._refit_sniff_phase( threshold=0.1 )
for tt in ['KPawakeM72_023_a_2658934','KPawakeM72_023_a_3065495','KPawakeM72_023_a_3132069','KPawakeM72_023_a_3305976',
           'KPawakeM72_023_a_4262713','KPawakeM72_023_a_4376037','KPawakeM72_023_a_4419898','KPawakeM72_023_a_4615212']:
    t._refit_sniff_phase()
    t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_5745834')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_014_b_3216156')
t._refit_sniff_phase()
t._del_inhalation(1)
t = get_trial('KPawakeM72_014_b_8780022')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('KPawakeM72_014_b_9857130')
t._refit_sniff_phase( threshold=0.1 )
t._del_inhalation(0)
t = get_trial('KPawakeM72_014_b_9973524')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('KPawakeM72_014_b_10172251')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('KPawakeM72_024_a_1856229')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_013_e_2299371')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_2366833')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_006_a_5114855')
t._refit_sniff_phase()
t._del_inhalation(14)
t = get_trial('ZKawakeM72_030_e_2369660')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_d_77627')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_d_885758')
t._refit_sniff_phase(threshold=0.3)
t._del_inhalation(4)
t = get_trial('ZKawakeM72_011_d_904802')
t._refit_sniff_phase(threshold=0.1)
t._del_inhalation(4)
t._add_inhalation(2330, 2600)
t._del_inhalation(6)
t._add_inhalation(3840, 4100)
t = get_trial('ZKawakeM72_011_d_927537')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_011_d_950734')
t._autodelete_inhalations()
t._del_inhalation(4)
t._del_inhalation(3)
t._build_inhalations()
t = get_trial('ZKawakeM72_030_a_571114')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_a_1790302')
t._refit_sniff_phase(threshold=0.2)
t._inhalation_onsets[6] = 3590
t._inhalation_offsets[6] = 3770
t._build_inhalations()
t = get_trial('ZKawakeM72_030_a_1835391')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_a_2340706')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_a_2388529')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_030_a_2522789')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_030_a_2762352')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_011_c_1156334')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_c_1426272')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_c_1575242')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_c_2410483')
t._refit_sniff_phase( threshold=0.2 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_c_2690967')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_030_c_137245')
t._refit_sniff_phase()
t._inhalation_offsets[11] = 5080
t._build_inhalations()
t = get_trial('ZKawakeM72_030_c_1052470')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_c_1289959')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_030_c_1489326')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_010_f_298142')
t._refit_sniff_phase(threshold=0.4)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_010_f_1325157')
t._refit_sniff_phase(threshold=0.1)
t._inhalation_onsets[5] = 4970
t._inhalation_offsets[5] = 5230
t._build_inhalations()
t = get_trial('ZKawakeM72_010_c_1071499')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_010_f_2088471')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_010_c_3621572')
t._refit_sniff_phase(threshold=0.2)
t._inhalation_onsets[15] = 5135
t._build_inhalations()
t = get_trial('ZKawakeM72_010_b_2841365')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_010_c_4108842')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_4414801')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_2756398')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_3306829')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_2738072')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_2386021')
t._refit_sniff_phase(prefilter=False)
t = get_trial('ZKawakeM72_004_a_1900538')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_a_1988274')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_004_a_2064539')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_c_665374')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_d_1345926')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(2)
t._build_inhalations()
t = get_trial('ZKawakeM72_004_d_1409151')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_004_e_1579907')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_004_d_3340328')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_004_f_2586770')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_f_3566023')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_g_775525')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_004_g_856628')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_004_g_1099946')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_g_1124107')
t._refit_sniff_phase(threshold=0.1)
t._inhalation_onsets[3] = 1955
t._inhalation_offsets[3] = 2110
t._build_inhalations()
t = get_trial('ZKawakeM72_004_g_1175538')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t._merge_inhalations(5,6)
t = get_trial('ZKawakeM72_005_a_1282042')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_005_a_2219702')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_029_f_1963249')
t._autodelete_inhalations()
t = get_trial('KPawakeM72_021_a_291805')
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_029_f_2460454')
t._automerge_inhalations()
t = get_trial('KPawakeM72_021_a_2595280')
t._automerge_inhalations()
t = get_trial('KPawakeM72_021_a_3223645')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_029_d_1648306')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('ZKawakeM72_011_b_92981')
t._refit_sniff_phase(threshold=0.2, prefilter=False)
t._inhalation_offsets[4] = 2031
t._inhalation_offsets[5] = 2160
t._inhalation_offsets[6] = 3594
t._inhalation_offsets[7] = 3853
t._build_inhalations()
t = get_trial('ZKawakeM72_020_g_502784')
t._refit_sniff_phase( filter_w_ms=1 )
t = get_trial('ZKawakeM72_020_g_1754217')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_020_g_667414')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_020_g_2260960')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_028_a_1629059')
t._refit_sniff_phase(threshold=0.2, prefilter=True, filter_w_ms=1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_028_a_860042')
t._autodelete_inhalations()
for tt in ['ZKawakeM72_030_b_532179', 'ZKawakeM72_030_b_601864', 'ZKawakeM72_030_b_649929', 'ZKawakeM72_030_b_914337', 'ZKawakeM72_030_b_988638',
          'ZKawakeM72_030_b_1129925', 'ZKawakeM72_030_b_1429302', 'ZKawakeM72_030_b_1997635' ]:
    t = get_trial(tt)
    t._refit_sniff_phase(threshold=0.3)
    t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_2378558')
t._refit_sniff_phase(threshold=0.3)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_2427475')
t._refit_sniff_phase(threshold=0.3)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_3030537')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_2782338')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_2556035')
t._refit_sniff_phase(threshold=0.4)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_b_3096273')
t._refit_sniff_phase( threshold=0.3 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_196520')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_575132')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_857903')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_1065096')
t._refit_sniff_phase()
t._del_inhalation(-1)
t = get_trial('ZKawakeM72_013_e_1231795')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_1435235')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_1536850')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_2206910')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_2366833')
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_f_1046958')
t._refit_sniff_phase( threshold=0.1 )
t._automerge_inhalations()
t = get_trial('ZKawakeM72_013_e_2752547')
t._automerge_inhalations()
t = get_trial('KPawakeM72_016_a_45168')
t._refit_sniff_phase(prefilter=False)
t = get_trial('KPawakeM72_016_a_349136')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_262298')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_217691')
t._refit_sniff_phase(prefilter=False)
t = get_trial('KPawakeM72_016_a_489855')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_537872')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_917723')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_1504607')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_016_a_3330231')
t._refit_sniff_phase()
t = get_trial('KPawakeM72_023_a_3305976')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_3132069')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_3065495')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_2658934')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_4119560')
t._refit_sniff_phase( threshold=0.1 )
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_4262713')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_4376037')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_4419898')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_023_a_4615212')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_006_a_5114855')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_011_d_950734')
t._refit_sniff_phase(threshold=0.1)
t._del_inhalation(9)
t._add_inhalation(3260,3400)
t = get_trial('ZKawakeM72_030_a_1660618')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_030_c_1489326')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_010_b_2841365')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_010_b_2988923')
t._refit_sniff_phase()
t = get_trial('ZKawakeM72_010_b_4950856')
t._refit_sniff_phase()
t = get_trial('ZKawakeM72_028_b_3306829')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_b_4414801')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_004_c_665374')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_004_f_3566023')
t._refit_sniff_phase()
t._merge_inhalations(1,2)
t = get_trial('ZKawakeM72_005_a_1282042')
t._del_inhalation(8)
t = get_trial('KPawakeM72_021_a_291805')
t._refit_sniff_phase(prefilter=False)
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_020_g_502784')
t._refit_sniff_phase(prefilter=False)
t._inhalation_offsets[6] = 1555
t._build_inhalations()
t = get_trial('KPawakeM72_019_b_1093208')
t._refit_sniff_phase()
t._merge_inhalations(4,5)
t = get_trial('ZKawakeM72_028_a_860042')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_028_a_3152657')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_030_b_2782338')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_005_a_853150')
t._refit_sniff_phase( threshold=0.5, filter_w_ms=40, filter_zero_mean=100 )
t._del_inhalation(2)
t._del_inhalation(7)
t._add_inhalation(2800, 2990)
t = get_trial('ZKawakeM72_005_a_186900')
t._refit_sniff_phase( threshold=0.2, filter_w_ms=40, filter_zero_mean=100 )
t._del_inhalation(0)
t._del_inhalation(0)
t._del_inhalation(6)
t = get_trial('ZKawakeM72_005_a_1820800')
t._refit_sniff_phase( threshold=2 )
t = get_trial('ZKawakeM72_005_a_1386248')
t._refit_sniff_phase( threshold=2 )
t = get_trial('ZKawakeM72_005_a_1023419')
t._refit_sniff_phase( threshold=2 )
t = get_trial('KPawakeM72_027_b_2561689')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('KPawakeM72_027_b_1464466')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('KPawakeM72_027_a_2274525')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(6)
t._add_inhalation(3447, 3620)
t = get_trial('ZKawakeM72_031_a_1191878')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_914_a_1813322')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_914_g_2684375')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_g_1515851')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t._merge_inhalations(3,4)
t._inhalation_offsets[4] = 2672
t._build_inhalations()
t = get_trial('ZKawakeM72_914_g_1362557')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_g_1228338')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_g_489988')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(17)
t._add_inhalation(2778,2915)
t._inhalation_offsets[18] = 3035
t._build_inhalations()
t = get_trial('ZKawakeM72_914_h_2042152')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(5)
t = get_trial('ZKawakeM72_914_h_1966807')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(4)
t = get_trial('ZKawakeM72_914_h_1806604')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_h_1004803')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_914_h_983619')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(0)
t._add_inhalation(70, 283)
t = get_trial('ZKawakeM72_914_h_892857')
t._refit_sniff_phase(threshold=0.2)
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_914_h_804829')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_914_h_782736')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_h_760868')
t._refit_sniff_phase(threshold=0.2)
t._del_inhalation(6)
t._add_inhalation(3600, 3820)
t = get_trial('ZKawakeM72_914_h_665417')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_914_h_279683')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_914_h_62873')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_031_e_1684171')
t._refit_sniff_phase(threshold=0.25)
t = get_trial('ZKawakeM72_031_e_1662938')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_031_e_1595364')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1362242')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1341643')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1282730')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1237236')
t._refit_sniff_phase()
t._automerge_inhalations()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_031_e_1189363')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1116414')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1095717')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('ZKawakeM72_031_e_1076818')
t._refit_sniff_phase(threshold=0.25)
t = get_trial('ZKawakeM72_031_e_1008073')
t._refit_sniff_phase(threshold=0.25)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_929357')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t._autodelete_inhalations()
t = get_trial('ZKawakeM72_031_e_903746')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_807497')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_738000')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_575776')
t._refit_sniff_phase(threshold=0.25)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_552070')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_409659')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_409659')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1753057')
t._refit_sniff_phase(threshold=0.25)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_1861222')
t._refit_sniff_phase(threshold=0.25)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_2194986')
t._refit_sniff_phase( threshold=0.1)
t = get_trial('ZKawakeM72_031_e_2679003')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_3714001')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_e_3951220')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_028_c_119361')
t._refit_sniff_phase()
t._autodelete_inhalations()
t = get_trial('KPawakeM72_028_c_709506')
t._refit_sniff_phase()
t._automerge_inhalations()
t = get_trial('KPawakeM72_028_c_533565')
t._refit_sniff_phase(threshold=0.1)
t = get_trial('KPawakeM72_028_c_167171')
t._refit_sniff_phase(threshold=0.1)
t._automerge_inhalations()
t = get_trial('KPawakeM72_028_c_2634146')
t._refit_sniff_phase(threshold=0.2)
t._automerge_inhalations()
t = get_trial('ZKawakeM72_031_c_104897')
t._refit_sniff_phase( threshold=0.1 )
t = get_trial('ZKawakeM72_031_c_1285562')
t._automerge_inhalations()

"""
=========================
load spikes for each cell
=========================
"""

spikes_filename = os.path.join( save_dir, 'spikes.pickle' )

# load if we can
if os.path.exists(spikes_filename):
    for _ in progress.dots([1], 'loading spikes from disk...'):
        spikes = cPickle.load(open(spikes_filename))
        

else:

    spikes = Bunch()
    for c in progress.dots( cell_file_list, 'loading cells' ):

        # load the rasters
        filename = join( source_dir, c + '_cell.mat' )
        try:
            this_data = loadmat(filename)
            raster = this_data['raster']
        except IOError:
            print 'File not available!'
            tracer()

        # if there are multiple recs for this cell, merge
        if hasattr( raster, '__len__' ):
            # spikes object for each rec
            rec_spikes = [ core.RawSpikes(r) for r in raster ]
            # construct the new cell idx
            r = rec_spikes[0]
            merged_spikes = r
            cell_idx = r.cell_idx.rpartition('_')[0].rpartition('_')[0] + '_'
            for r in rec_spikes:
                cell_idx += r.rec
            cell_idx += '_' + r.cell_idx.split('_')[-1]
            # merge
            merged_spikes.cell_idx = cell_idx
            merged_spikes.odour_conc = conc([ r.odour_conc for r in rec_spikes ])
            merged_spikes.metadata = rec_spikes[0].metadata
            merged_spikes.odour_name = conc([ r.odour_name for r in rec_spikes ])
            merged_spikes.spikes = conc([ r.spikes for r in rec_spikes ])
            merged_spikes.t0 = conc([ r.t0 for r in rec_spikes ])
            merged_spikes.trial_idx = conc([ r.trial_idx for r in rec_spikes ])
            # save
            spikes[c] = merged_spikes

        # otherwise load
        else:
            spikes[c] = core.RawSpikes(raster)

        # make note of metadata
        spikes[c].metadata.ipsi = this_data['is_ipsi']
            
    # save
    for _ in progress.dots([1], 'saving spikes to disk'):
        cPickle.dump( spikes, open(spikes_filename, 'w'), 2 )
        



"""
====================
aggregate Recordings
====================
"""

# available recs
recs = list(np.unique([ s.cell_idx.rpartition('_')[0] for s in spikes.values() ]))

# compile
recordings = Bunch()
for r in recs:
    
    rk = recordings[r] = Bunch()
    
    # get prefixes
    prefix = r.rpartition('_')[0]
    r_idxs = r.split('_')[-1]
    rk.rec = r
    
    # compile this recording's trials
    try:
        rk['trials'] = sum([ trials[ prefix + '_' + r_idx ] for r_idx in r_idxs ], [])
    except KeyError:
        print 'missing: ', prefix + '_' + r_idx
        del recordings[r]
        continue
        
    # compile this recording's spikes
    cells = [ k for k in spikes.keys() if k.startswith(prefix) ]
    cells = sorted([ c for c in cells if str(spikes[c].rec) == r_idxs ])
    if len(cells) == 0:
        del recordings[r]
        continue
    rk.N_cells = len(cells)
    rk['spikes_by_cell'] = [ spikes[c] for c in cells ]
    
    # compile Data object
    rk.data = core.TrialData( rk.trials, rk.spikes_by_cell )

    """
    # old exception: may need to revive with whole data set
    except AttributeError:
        trial_idxs = [t.trial_idx for t in rk.trials]
        to_keep = [trial_idxs.index(i) for i in rk.spikes[0].trial_idx]
        rk.trials.trials = [rk.trials.trials[i] for i in to_keep]
        rk.trial_data = core.TrialData( rk.trials, rk.spikes )
    """
    
    # compile Sniffs
    rk.sniffs = core.Sniffs( rk.data )
    

# finalise as a RecordingList object
recordings = core.RecordingList([ core.Recording(recordings[k]) for k in sorted( recordings.keys() ) ])



"""
===============
aggregate cells
===============
"""

cells = core.CellList([])
for r in recordings:
    for n, s in enumerate( r.spikes_by_cell ):
        c = core.Cell( r, s.cell_idx, n )
        cells.append(c)



"""
====================
concentration series
====================
"""

# is it a concentration series
for r in recordings:
    r.is_concentration_series = False
    names = r.data.odour_name__a
    concs = r.data.odour_conc__a
    tok = concs > 0
    N_names = len(set(names))
    if N_names <= 2:
        N_unique = len(set(zip( names[tok], concs[tok] )))
        if N_unique > N_names:
            r.is_concentration_series = True

# aggregate
odour_names = [u'2_hydroxyacetophenone', u'menthone']

for r in recordings:
    if not r.is_concentration_series:
        continue

    on = r.data.odour_name__a
    oc = np.log10(r.data.odour_conc__a)
    conc_idx = np.zeros_like( oc )
    
    conc_idx[ (on == odour_names[0]) & np.isfinite(oc) ] = 2
    conc_idx[ (on == odour_names[0]) & (oc < -4) ] = 1
    conc_idx[ (on == odour_names[0]) & (oc > -2.5) ] = 3
    
    conc_idx[ (on == odour_names[1]) & np.isfinite(oc) ] = 5
    conc_idx[ (on == odour_names[1]) & (oc < -3.5) ] = 4
    conc_idx[ (on == odour_names[1]) & (oc > -2.2) ] = 6
    
    r.data.odour_conc_idx__a = conc_idx.astype(int)

