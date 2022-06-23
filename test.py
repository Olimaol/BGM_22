from ANNarchy import *
from BGM_22 import BGM
from CompNeuroPy import Monitors
import CompNeuroPy.analysis_functions as af
import pylab as plt
import time

setup(dt=0.1, seed=1)
model, params = BGM(do_compile=True)



mon = Monitors({'pop;gpe_arky':['spike','g_ampa','g_gaba'],
                'pop;str_d1':['spike','g_ampa','g_gaba'],
                'pop;str_d2':['spike','g_ampa','g_gaba'],
                'pop;stn':['spike','g_ampa','g_gaba'],
                'pop;cor_go':['spike'],
                'pop;gpe_cp':['spike','g_ampa','g_gaba'],
                'pop;gpe_proto':['spike','g_ampa','g_gaba'],
                'pop;snr':['spike','g_ampa','g_gaba'],
                'pop;thal':['spike','g_ampa','g_gaba'],
                'pop;cor_stop':['spike'],
                'pop;str_fsi':['spike','g_ampa','g_gaba'],
                'pop;integrator_go':['g_ampa', 'decision']})




paramsS = {}
paramsS['trials'] = 2


### GO TRIALS ###
print('\n\nSTART GO TRIALS')
mode='GO'
zaehler_go = 0
for i in range(0,paramsS['trials']):
    start=time.time()
    ### RESET MODEL ###
    
    if i!=0: print('\nreset before trial...'); mon.reset(populations=True, projections=True, synapses=False, net_id=0)
    mon.start()

    ### TRIAL START
    print('TRIAL START, trial: '+str(i))

    ### trial INITIALIZATION simulation to get stable state
    get_population('cor_go').rates = 0
    get_population('cor_stop').rates = 0
    get_population('cor_pause').rates = 0

    print('simulate t_init and events...')
    simulate(params['t_init'])# t_init rest

    ### Integrator Reset
    get_population('integrator_go').decision = 0
    get_population('integrator_go').g_ampa = 0  
    get_population('integrator_stop').decision = 0 

    ### calculate all eventTIMES
    GOCUE     = np.round(get_time(),1)
    STOPCUE   = GOCUE + params['t_SSD']
    STN1ON    = GOCUE
    STN1OFF   = STN1ON + params['t_cortexPauseDuration']
    GOON      = GOCUE + params['t_delayGo'] + int(np.clip(params['t_delayGoSD'] * np.random.randn(),-params['t_delayGo'],None))
    STN2ON    = STOPCUE
    STN2OFF   = STN2ON + params['t_cortexPauseDuration']
    STOPON    = STOPCUE + params['t_delayStopAfterCue']
    STOPOFF   = STOPON + params['t_cortexStopDurationAfterCue']
    ENDTIME   = STOPOFF + params['t_decay']
    GOOFF     = ENDTIME
    motorSTOP = ENDTIME
    motorSTOPOFF = ENDTIME
    eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
    randn_trial_Pause = np.random.randn()
    randn_trial = np.random.randn()
    randn_trial_S = np.random.randn()

    ### simulate all EVENTS
    motorResponse=0
    GOOFFResponse=0
    responseThal=0
    responseProto2=0
    t=np.round(get_time(),1)
    end=False
    tempMode=mode
    while not(end):
        if t == STN1ON:
            get_population('cor_pause').rates = params['cor_pause__rates'] + params['cor_pause__rates_sd'] * randn_trial_Pause
            #print('STN1ON',STN1ON)
        if t == STN1OFF:
            get_population('cor_pause').rates = 0
            #print('STN1OFF',STN1OFF)
        if t == GOON:
            get_population('cor_go').rates = params['cor_go__rates'] + params['cor_go__rates_sd'] * randn_trial
            #print('GOON',GOON)
        if t == GOOFF:
            get_population('cor_go').rates = 0
            #print('GOOFF',GOOFF)                    
        if t == STN2ON and tempMode=='STOP':
            get_population('cor_pause').rates = params['cor_pause__rates_second_resp_mod']*params['cor_pause__rates'] + params['cor_pause__rates_sd'] * randn_trial_Pause
            #print('STN2ON',STN2ON)
        if t == STN2OFF and tempMode=='STOP':
            get_population('cor_pause').rates = 0
            #print('STN2OFF',STN2OFF)
        if t == STOPON and tempMode=='STOP':
            get_population('cor_stop').rates = params['cor_stop__rates_after_cue'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_cue'] > 0)
            #print('STOPON',STOPON)
        if t == motorSTOP:
            get_population('cor_stop').rates = params['cor_stop__rates_after_action'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_action'] > 0)
            motorSTOPOFF = motorSTOP + params['t_cortexStopDurationAfterAction']
            #print('motorSTOP',motorSTOP)
        if t == STOPOFF and tempMode=='STOP' and ((motorSTOPOFF==ENDTIME) or (motorSTOPOFF!=ENDTIME and t>motorSTOPOFF)):
            get_population('cor_stop').rates = 0
            #print('STOPOFF',STOPOFF)
        if t == motorSTOPOFF:
            get_population('cor_stop').rates = 0
            #print('motorSTOPOFF',motorSTOPOFF)
        if t == ENDTIME:
            end=True
            #print('ENDTIME',ENDTIME)
        else:
            nowTIME = np.round(get_time(),1)
            #print('nowTIME',nowTIME)
            eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
            nextEvent = np.max([np.min(eventTimes[eventTimes>=nowTIME]-nowTIME),1])
            #print('nextEvent',nextEvent)
            if responseThal==0 and responseProto2==0:
                simulate_until(max_duration=nextEvent, population=[get_population('integrator_go'),get_population('integrator_stop')], operator='or')
            elif responseProto2==0:
                simulate_until(max_duration=nextEvent, population=get_population('integrator_stop'))
            elif responseThal==0:
                simulate_until(max_duration=nextEvent, population=get_population('integrator_go'))
            else:
                simulate(nextEvent)
            responseThal = int(get_population('integrator_go').decision[0])
            responseProto2 = int(get_population('integrator_stop').decision[0])
            t = np.round(get_time(),1)
            #print('time:',t,'restsimulation:',np.ceil(t)-t)
            simulate(np.round(np.ceil(t)-t,1))
            t = np.round(get_time(),1)
            if responseThal == -1 and motorResponse == 0:
                motorResponse=1
                motorSTOP = t + params['t_delayStopAfterAction']
                if t<STOPCUE and tempMode=='STOP':
                    tempMode='GO'
            if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                GOOFFResponse=1
                GOOFF = t
    ### TRIAL END
    print('Integrator decision:',get_population('integrator_go').decision[0])
    print('TRIAL END')
    if i==1: simulate(50)

    if get_population('integrator_go').decision[0] == -1 :
        t= get_current_step()
        zaehler_go = zaehler_go + 1
        
    mon.pause()
    print('time:',time.time()-start)
### END GO TRIALS ###
print('\nGO TRIALS FINISHED\nzaehler_go:',zaehler_go)
      
      
recordings=mon.get_recordings()
recording_times=mon.get_recording_times()


plot_list = ['1;gpe_arky;spike;hybrid',
             '2;str_d1;spike;hybrid',
             '3;str_d2;spike;hybrid',
             '4;stn;spike;hybrid',
             '5;cor_go;spike;hybrid',
             '6;gpe_cp;spike;hybrid',
             '7;gpe_proto;spike;hybrid',
             '8;snr;spike;hybrid',
             '9;thal;spike;hybrid',
             '10;cor_stop;spike;hybrid',
             '11;str_fsi;spike;hybrid',
             '12;integrator_go;g_ampa;line']
chunk=0 # chunks are separated by resets --> different recording/recording_times, here each trial is an own chunk
compartment='gpe_arky' # here all compartments have the same timings... doesn't matter
period=0 # separated by pauses, one period = [start,stop], here one trial is one period... only one available
time_lims = recording_times.time_lims(chunk=chunk, compartment=compartment, period=period)
idx_lims  = recording_times.idx_lims(chunk=chunk, compartment=compartment, period=period)
af.plot_recordings('overview1_old.png', recordings[chunk], time_lims, idx_lims, (2,6), plot_list)


chunk=1
time_lims = recording_times.time_lims(chunk=chunk)
idx_lims  = recording_times.idx_lims(chunk=chunk)
af.plot_recordings('overview2_old.png', recordings[chunk], time_lims, idx_lims, (2,6), plot_list)


quit()


### STOP TRIALS ###
print('\nSTART STOP TRIALS')
mode='STOP'
for i in range (0,paramsS['trials']):
    ### RESET MODEL ###
    print('\nreset before trial...')
    reset(populations=True, projections=True, synapses=False, net_id=0)

    ### TRIAL START
    print('TRIAL START, trial: '+str(i))

    ### trial INITIALIZATION simulation to get stable state
    get_population('cor_go').rates = 0
    get_population('cor_stop').rates = 0
    get_population('cor_pause').rates = 0

    print('simulate t_init and events...')
    simulate(params['t_init'])# t_init rest

    ### Integrator Reset
    get_population('integrator_go').decision = 0
    get_population('integrator_go').g_ampa = 0  
    get_population('integrator_stop').decision = 0 

    ### calculate all eventTIMES
    GOCUE     = np.round(get_time(),1)
    STOPCUE   = GOCUE + params['t_SSD']
    STN1ON    = GOCUE
    STN1OFF   = STN1ON + params['t_cortexPauseDuration']
    GOON      = GOCUE + params['t_delayGo'] + int(np.clip(params['t_delayGoSD'] * np.random.randn(),-params['t_delayGo'],None))
    STN2ON    = STOPCUE
    STN2OFF   = STN2ON + params['t_cortexPauseDuration']
    STOPON    = STOPCUE + params['t_delayStopAfterCue']
    STOPOFF   = STOPON + params['t_cortexStopDurationAfterCue']
    ENDTIME   = STOPOFF + params['t_decay']
    GOOFF     = ENDTIME
    motorSTOP = ENDTIME
    motorSTOPOFF = ENDTIME
    eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
    randn_trial_Pause = np.random.randn()
    randn_trial = np.random.randn()
    randn_trial_S = np.random.randn()

    ### simulate all EVENTS
    motorResponse=0
    GOOFFResponse=0
    responseThal=0
    responseProto2=0
    t=np.round(get_time(),1)
    end=False
    tempMode=mode
    while not(end):
        if t == STN1ON:
            get_population('cor_pause').rates = params['cor_pause__rates'] + params['cor_pause__rates_sd'] * randn_trial_Pause
            #print('STN1ON',STN1ON)
        if t == STN1OFF:
            get_population('cor_pause').rates = 0
            #print('STN1OFF',STN1OFF)
        if t == GOON:
            get_population('cor_go').rates = params['cor_go__rates'] + params['cor_go__rates_sd'] * randn_trial
            #print('GOON',GOON)
        if t == GOOFF:
            get_population('cor_go').rates = 0
            #print('GOOFF',GOOFF)                    
        if t == STN2ON and mode=='STOP':
            get_population('cor_pause').rates = params['cor_pause__rates_second_resp_mod']*params['cor_pause__rates'] + params['cor_pause__rates_sd'] * randn_trial_Pause
            #print('STN2ON',STN2ON)
        if t == STN2OFF and mode=='STOP':
            get_population('cor_pause').rates = 0
            #print('STN2OFF',STN2OFF)
        if t == STOPON and mode=='STOP':
            get_population('cor_stop').rates = params['cor_stop__rates_after_cue'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_cue'] > 0)
            #print('STOPON',STOPON)
        if t == motorSTOP:
            get_population('cor_stop').rates = params['cor_stop__rates_after_action'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_action'] > 0)
            motorSTOPOFF = motorSTOP + params['t_cortexStopDurationAfterAction']
            #print('motorSTOP',motorSTOP)
        if t == STOPOFF and mode=='STOP' and ((motorSTOPOFF==ENDTIME) or (motorSTOPOFF!=ENDTIME and t>motorSTOPOFF)):
            get_population('cor_stop').rates = 0
            #print('STOPOFF',STOPOFF)
        if t == motorSTOPOFF:
            get_population('cor_stop').rates = 0
            #print('motorSTOPOFF',motorSTOPOFF)
        if t == ENDTIME:
            end=True
            #print('ENDTIME',ENDTIME)
        else:
            nowTIME = np.round(get_time(),1)
            #print('nowTIME',nowTIME)
            eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
            nextEvent = np.max([np.min(eventTimes[eventTimes>=nowTIME]-nowTIME),1])
            #print('nextEvent',nextEvent)
            if responseThal==0 and responseProto2==0:
                simulate_until(max_duration=nextEvent, population=[get_population('integrator_go'),get_population('integrator_stop')], operator='or')
            elif responseProto2==0:
                simulate_until(max_duration=nextEvent, population=get_population('integrator_stop'))
            elif responseThal==0:
                simulate_until(max_duration=nextEvent, population=get_population('integrator_go'))
            else:
                simulate(nextEvent)
            responseThal = int(get_population('integrator_go').decision)
            responseProto2 = int(get_population('integrator_stop').decision)
            t = np.round(get_time(),1)
            #print('time:',t,'restsimulation:',np.ceil(t)-t)
            simulate(np.round(np.ceil(t)-t,1))
            t = np.round(get_time(),1)
            if responseThal == -1 and motorResponse == 0:
                motorResponse=1
                motorSTOP = t + params['t_delayStopAfterAction']
                if t<STOPCUE and tempMode=='STOP':
                    tempMode='GO'
            if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                GOOFFResponse=1
                GOOFF = t
    ### TRIAL END
    print('Integrator decision:',get_population('integrator_go').decision[0])
    print('TRIAL END')
    
    
    if get_population('integrator_go').decision == -1 :
        t= get_current_step()
        zaehler_go = zaehler_go + 1

### END OF STOP TRIALS ###
print('zaehler_go:',zaehler_go)

print('\nTIME:',time.time()-start)
