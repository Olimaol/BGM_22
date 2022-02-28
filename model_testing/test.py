from ANNarchy import *
from BGM_22 import BGM
from CompNeuroPy import Monitors
import pylab as plt

setup(dt=0.1)
model, params = BGM(do_compile=True)

mon = Monitors({'pop;cor_go':['spike'], 'pop;integrator_go':['g_ampa']})
mon.start()


paramsS = {}
paramsS['trials'] = 1
paramsS['makeParameterVariations'] = 0
paramsS['saveFolder'] = 'longertestSim'


### GO TRIALS ###
print('\n\nSTART GO TRIALS')
mode='GO'
zaehler_go = 0
for i in range(0,paramsS['trials']):
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
            print('STN1ON',STN1ON)
        if t == STN1OFF:
            get_population('cor_pause').rates = 0
            print('STN1OFF',STN1OFF)
        if t == GOON:
            get_population('cor_go').rates = params['cor_go__rates'] + params['cor_go__rates_sd'] * randn_trial
            print('GOON',GOON)
        if t == GOOFF:
            get_population('cor_go').rates = 0
            print('GOOFF',GOOFF)                    
        if t == STN2ON and tempMode=='STOP':
            get_population('cor_pause').rates = params['cor_pause__rates_second_resp_mod']*params['cor_pause__rates'] + params['cor_pause__rates_sd'] * randn_trial_Pause
            print('STN2ON',STN2ON)
        if t == STN2OFF and tempMode=='STOP':
            get_population('cor_pause').rates = 0
            print('STN2OFF',STN2OFF)
        if t == STOPON and tempMode=='STOP':
            get_population('cor_stop').rates = params['cor_stop__rates_after_cue'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_cue'] > 0)
            print('STOPON',STOPON)
        if t == motorSTOP:
            get_population('cor_stop').rates = params['cor_stop__rates_after_action'] + params['cor_stop__rates_sd'] * randn_trial_S * (params['cor_stop__rates_after_action'] > 0)
            motorSTOPOFF = motorSTOP + params['t_cortexStopDurationAfterAction']
            print('motorSTOP',motorSTOP)
        if t == STOPOFF and tempMode=='STOP' and ((motorSTOPOFF==ENDTIME) or (motorSTOPOFF!=ENDTIME and t>motorSTOPOFF)):
            get_population('cor_stop').rates = 0
            print('STOPOFF',STOPOFF)
        if t == motorSTOPOFF:
            get_population('cor_stop').rates = 0
            print('motorSTOPOFF',motorSTOPOFF)
        if t == ENDTIME:
            end=True
            print('ENDTIME',ENDTIME)
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

### END GO TRIALS ###
print('\nGO TRIALS FINISHED\nzaehler_go:',zaehler_go)
      
      
      
recordings=mon.get_recordings()### TODO: add to the Monitors CompNeuroPy class: return also start and endtime of Monitors



def get_pop_rate(spikes,duration,dt=1,t_start=0,t_smooth_ms=-1):#TODO: maybe makes errors with fes/no spikes... check this TODO: add this to CompNeuroPy
    """
        spikes: spikes dictionary from ANNarchy
        duration: duration of period after (optional) initial period (t_start) from which rate is calculated in ms
        dt: timestep of simulation
        t_start: starting simulation time, from which rates should be calculated
        t_smooth_ms: time window size for rate calculation in ms, optional, standard = -1 which means automatic window size
        
        returns smoothed population rate from period after rampUp period until duration
    """
    temp_duration=duration+t_start
    t,n = raster_plot(spikes)
    if len(t)>1:#check if there are spikes in population at all
        if t_smooth_ms==-1:
            ISIs = []
            minTime=np.inf
            duration=0
            for idx,key in enumerate(spikes.keys()):
                times = np.array(spikes[key]).astype(int)
                if len(times)>1:#check if there are spikes in neuron
                    ISIs += (np.diff(times)*dt).tolist()#ms
                    minTime=np.min([minTime,times.min()])
                    duration=np.max([duration,times.max()])
                else:# if there is only 1 spike set ISI to 10ms
                    ISIs+=[10]
            t_smooth_ms = np.min([(duration-minTime)/2.*dt,np.mean(np.array(ISIs))*10+10])

        rate=np.zeros((len(list(spikes.keys())),int(temp_duration/dt)))
        rate[:]=np.NaN
        binSize=int(t_smooth_ms/dt)
        bins=np.arange(0,int(temp_duration/dt)+binSize,binSize)
        binsCenters=bins[:-1]+binSize//2
        for idx,key in enumerate(spikes.keys()):
            times = np.array(spikes[key]).astype(int)
            for timeshift in np.arange(-binSize//2,binSize//2+binSize//10,binSize//10).astype(int):
                hist,edges=np.histogram(times,bins+timeshift)
                rate[idx,np.clip(binsCenters+timeshift,0,rate.shape[1]-1)]=hist/(t_smooth_ms/1000.)

        poprate=np.nanmean(rate,0)
        timesteps=np.arange(0,int(temp_duration/dt),1).astype(int)
        time=timesteps[np.logical_not(np.isnan(poprate))]
        poprate=poprate[np.logical_not(np.isnan(poprate))]
        poprate=np.interp(timesteps, time, poprate)

        ret = poprate[int(t_start/dt):]
    else:
        ret = np.zeros(int(duration/dt))

    return ret
  

def plot_recordings(recordings, shape, plan):#TODO add this function to CompNeuroPy
    """
        recordings: dict of recordings
        shape: tuple, shape of subplots
        plan: list of strings, strings defin where to plot which data and how
    """
    
    plt.figure(figsize=([6.4*shape[1], 4.8*shape[0]]))
    for subplot in plan:
        try:
            nr, part, variable, mode = subplot.split(';')
            nr=int(nr)
        except:
            print('\nERROR plot_recordings: for each subplot give plan-string as: "nr;part;variable;mode"!\n')
            quit()
        try:
            data=recordings[';'.join([part,variable])]
        except:
            print('\nERROR plot_recordings: data',';'.join([part,variable]),'not in recordings\n')
            
        start_time=0
        end_time=1300
        
        times=np.arange(start_time,end_time,recordings['dt'])
            
        plt.subplot(shape[0],shape[1],nr)
        if variable=='spike' and mode=='raster':
            t,n=raster_plot(data)
            plt.plot(t,n,'k.')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.ylabel('# neurons')
            plt.title('Spikes '+part)
        elif variable=='spike' and mode=='mean':
            firing_rate = get_pop_rate(data,end_time-start_time,dt=recordings['dt'],t_start=start_time)
            plt.plot(times,firing_rate, color='k')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.ylabel('Mean firing rate [Hz]')
            plt.title('Mean firing rate '+part)
        elif variable=='spike' and mode=='hybrid':
            t,n=raster_plot(data)
            plt.plot(t,n,'k.')
            plt.ylabel('# neurons')
            ax=plt.gca().twinx()
            firing_rate = get_pop_rate(data,end_time-start_time,dt=recordings['dt'],t_start=start_time)
            ax.plot(times,firing_rate, color='r')
            plt.ylabel('Mean firing rate [Hz]', color='r')
            ax.tick_params(axis='y', colors='r')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.title('Activity '+part)
        elif variable!='spike' and mode=='line':
            for idx in range(data.shape[1]):#TODO I'm here
                plt.plot(times,data[:,idx], color='k')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.title('Variable '+part+' '+variable)
        elif variable!='spike' and mode=='mean':
            a=0
        else:
            print('\nERROR plot_recordings: mode',mode,'not available for variable',variable,'\n')
    plt.tight_layout()
    plt.savefig('test.svg')
        
print(recordings['integrator_go;g_ampa'].shape)
print(recordings['integrator_go;g_ampa'].shape[0])
print(recordings['integrator_go;g_ampa'].shape[1]) #TODO I'm here
quit()
plot_recordings(recordings, (2,1), ['1;cor_go;spike;hybrid', '2;integrator_go;g_ampa;line'])   

quit()


### STOP TRIALS ###
print('\nSTART STOP TRIALS')
mode='STOP'
for i in range (0,paramsS['trials']):
    ### RESET MODEL ###
    print('\nreset before trial...')
    reset(populations=True, projections=True, synapses=False, net_id=0)

    ### TRIAL START
    print('TRIAL START, cycle: '+str(i_cycle+1)+'/'+str(n_loop_cycles)+', trial: '+str(i))

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
        selection[i] = get_population('integrator_go').decision
        timeovertrial[i]=t
        zaehler_go = zaehler_go + 1                                         
    else:
        selection[i] = get_population('integrator_go').decision
        timeovertrial[i]=t

### END OF STOP TRIALS ###
print('zaehler_go:',zaehler_go)
