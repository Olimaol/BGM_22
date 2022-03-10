from ANNarchy import *
from BGM_22 import compare_str
from CompNeuroPy import Monitors
import CompNeuroPy.analysis_functions as af
import pylab as plt

setup(dt=0.1, seed=0)
model, params = compare_str(do_compile=True)


mon = Monitors({'pop;str_d1':['spike','v','g_ampa'],
                'pop;str_d2':['spike','v','g_ampa'],
                'pop;str_fsi':['spike','v','g_ampa'],
                'pop;str_d1_old':['spike','v','g_ampa'],
                'pop;str_d2_old':['spike','v','g_ampa'],
                'pop;str_fsi_old':['spike','v','g_ampa']})
mon.start()

simulate(100)

recordings=mon.get_recordings()
recording_times=mon.get_times()

plot_list= ['1;str_d1;spike;hybrid',
            '4;str_d1;g_ampa;line',
            '7;str_d1_old;spike;hybrid',
            '10;str_d1_old;g_ampa;line',
            '2;str_d2;spike;hybrid',
            '5;str_d2;g_ampa;line',
            '8;str_d2_old;spike;hybrid',
            '11;str_d2_old;g_ampa;line',
            '3;str_fsi;spike;hybrid',
            '6;str_fsi;g_ampa;line',
            '9;str_fsi_old;spike;hybrid',
            '12;str_fsi_old;g_ampa;line']
af.plot_recordings('compare_str.png', recordings, recording_times['start'][0], recording_times['stop'][0], (4,3), plot_list)


af.plot_recordings('compare_d1.svg', recordings, recording_times['start'][0], recording_times['stop'][0], (1,1), ['1;str_d1;spike;raster','1;str_d1_old;spike;raster;r'])
af.plot_recordings('compare_d2.svg', recordings, recording_times['start'][0], recording_times['stop'][0], (1,1), ['1;str_d2;spike;raster','1;str_d2_old;spike;raster;r'])
af.plot_recordings('compare_fsi.svg', recordings, recording_times['start'][0], recording_times['stop'][0], (1,1), ['1;str_fsi;spike;raster','1;str_fsi_old;spike;raster;r'])


