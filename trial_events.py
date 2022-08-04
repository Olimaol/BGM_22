from ANNarchy import get_current_step, dt, get_population
import numpy as np

def add_events(trial_procedure, params):

    ### define effects for all events (with effects)
    def cor_on(self, pop_name, append=''): get_population(pop_name).rates = self.trial_procedure.params[pop_name+'.rates'+append] + self.trial_procedure.params[pop_name+'.rates_sd'] * self.trial_procedure.rand[pop_name] * int(params[pop_name+'.rates'+append] > 0)
    def cor_off(self, pop_name): get_population(pop_name).rates = 0
    
    ### add all events
    trial_procedure.add_event(name='go_cue',
                              onset=get_current_step(),
                              trigger={'stop_cue': int(params['sim.t_SSD']/dt()),
                                       'cor_pause_on_go': 0,
                                       'cor_go_on': int(np.clip(np.random.normal(params['sim.t_delayGo'],params['sim.t_delayGoSD']), 0, None)/dt()),
                                       'end': int((params['sim.t_SSD']+params['sim.t_delayStopAfterCue']+params['sim.t_cortexStopDurationAfterCue']+params['sim.t_decay'])/dt())})
                                       
    trial_procedure.add_event(name='stop_cue',
                              requirement_string="mode==stop",
                              trigger={'cor_pause_on_stop': 0,
                                       'cor_stop_on_cue':  int(params['sim.t_delayStopAfterCue']/dt())})
                                   
    trial_procedure.add_event(name='cor_go_on',
                              effect=lambda self : cor_on(self, 'cor_go'))
                              
    trial_procedure.add_event(name='cor_go_off',
                              effect=lambda self : cor_off(self, 'cor_go'))
                                       
    trial_procedure.add_event(name='cor_pause_on_go',
                              effect=lambda self : cor_on(self, 'cor_pause', '_go'),
                              trigger={'cor_pause_off': int(params['sim.t_cortexPauseDuration']/dt())})
                              
    trial_procedure.add_event(name='cor_pause_on_stop',
                              effect=lambda self : cor_on(self, 'cor_pause', '_stop'),
                              trigger={'cor_pause_off': int(params['sim.t_cortexPauseDuration']/dt())})
                              
    trial_procedure.add_event(name='cor_pause_off',
                              effect=lambda self : cor_off(self, 'cor_pause'))
                              
    trial_procedure.add_event(name='cor_stop_on_cue',
                              effect=lambda self : cor_on(self, 'cor_stop', '_cue'),
                              trigger={'cor_stop_off': int(params['sim.t_cortexStopDurationAfterCue']/dt())})
                              
    trial_procedure.add_event(name='cor_stop_on_motor',
                              effect=lambda self : cor_on(self, 'cor_stop', '_motor'),
                              trigger={'cor_stop_off': int(params['sim.t_cortexStopDurationAfterAction']/dt())})
                              
    trial_procedure.add_event(name='cor_stop_off',
                              effect=lambda self : cor_off(self, 'cor_stop'),
                              trigger={'end': int(params['sim.t_decay']/dt())})
                              
    trial_procedure.add_event(name='end',
                              effect=lambda self : self.trial_procedure.end_sim())
                                       
    trial_procedure.add_event(name='motor_response',
                              model_trigger='integrator_go',
                              trigger={'cor_stop_on_motor': int(params['sim.t_delayStopAfterAction']/dt())})
                              
    trial_procedure.add_event(name='gpe_cp_resp',
                              model_trigger='integrator_stop',
                              requirement_string='happened_event_list==[cor_stop_on_cue] or happened_event_list==[motor_resp]',
                              trigger={'cor_go_off': 0})
