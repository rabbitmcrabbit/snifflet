""" Setup """

from rq import Queue
from job_worker_connect import conn
import progress

from numpy.random import permutation

q = Queue( connection=conn )


for i, j in enumerate( q.jobs ):
    print '[%d]: %s' % (i, j.description)
