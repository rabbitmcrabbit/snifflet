#/bin/bash

cd /users-archive/neil/olfaction

export PATH=/users-local/neil/v3/bin:/bin:/users/neil/.scripts:/users/neil/.nx/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

export PYTHONPATH=/usr/local/lib:/users-archive/neil/nib:/users-archive/neil/pykemonger:/users-archive/neil/drc:/users-archive/neil/drc/kernel.models:/users-archive/neil/natural.contrast:/users-archive/neil/ald:/users/neil/downloads/line_profiler-1.0b3/build/lib.linux-x86_64-2.7:/users-archive/neil/pyPyrTools

export logfile=/users-archive/neil/olfaction/log/$(uname -n).$(date +%H%M%S)

nice ipython --pylab --profile v3 job_worker_connect.py > $logfile
