from helpers import *
import glm
import core
import get_data
from get_data import recordings, cells

for r in progress.dots( recordings, 'preparing XY' ):
    r.XY = r.baseline_data.get_flow_XY_data()

for c in progress.numbers( cells ):
    # is it done
    if c.can_load_attribute('cv_slices_baseline'):
        continue
    # calculate slices
    c.cv_slices_baseline = []
    for _ in range(10):
        m = c.get_flow_asd_model( c.recording.XY, testing_proportion=0.2 )
        c.cv_slices_baseline.append( 
                Bunch({'training_slices':m.training_slices, 'testing_slices':m.testing_slices}) )
    # save
    c.save_attribute('cv_slices_baseline')
