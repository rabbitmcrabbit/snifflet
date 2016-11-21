from rq import Queue
from job_worker_connect import conn
import progress

from numpy.random import permutation

q = Queue( "failed", connection=conn )
q.empty()
