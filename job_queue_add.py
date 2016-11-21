""" Setup """

from rq import Queue
from job_worker_connect import conn
import progress

from numpy.random import permutation

q = Queue( connection=conn )


""" Queue jobs """

#import A1_asd_fit as A
#import A2_asd_fit_nohistory as A
#import A3_warp_slow as A
import A4_warp_fast as A

for script in [A]:
    for repeat in range(3):
        for j in progress.dots( permutation( script.jobs ), 
                '%s: adding jobs (%d)' % (script.__name__, repeat) ):
            func = j[0]
            args = tuple([a for a in j[1:]])
            result = q.enqueue_call( func, args=args, timeout=60*60*4 )
