import numpy as np
from ANNarchy import simulate, simulate_until, get_population, get_current_step, dt, get_time
from CompNeuroPy.analysis_functions import get_number_of_zero_decimals

class trial_procedure_cl:
    
        def __init__(self, params, paramsS, mode, print_outs=False):
            self.params = params
            self.paramsS = paramsS
            self.print_outs = print_outs
            self.event_list = []
            self.event_name_list = []
            ### each trial can have random variation in cor activities
            self.rand={}
            self.rand['cor_go'] = np.random.randn()
            self.rand['cor_pause'] = np.random.randn()
            self.rand['cor_stop'] = np.random.randn()
            ### as long as end == False simulation runs
            self.end = False
            ### if events occur depends on mode and happened events
            self.mode = mode
            self.happened_event_list = []
            ### initialize model triggers empty, before first simulation, there should not be model_trigger_events
            ### model_trigger_list = name of populations of which the decision should be checked
            self.model_trigger_list = []
            self.past_model_trigger_list = []
                        
        def add_event(self, name, onset=None, model_trigger=None, requirement_string=None, effect=None, trigger=None):
            """
                effect: function; what happens during event
                trigger: dict; what other events are triggerd (keys) and when relative to onset of current event (vals)
            """
            self.event_list.append(self.event_cl(self, name=name, onset=onset, model_trigger=model_trigger, requirement_string=requirement_string, effect=effect, trigger=trigger))
            self.event_name_list.append(name)
            
        def run(self):
            while not(self.end):
                ### check if model triggers were activated --> if yes run the corresponding events, model_trigger events can trigger other events (with onset) --> run current_step events after model trigger events
                ### if that's the case --> model trigger event would run twice (because during first run it gets an onset) --> define here run_event_list which prevents events run twice
                self.run_event_list = []
                self.run_model_trigger_events()
                ### run the events of the current time, based on mode and happened events
                self.run_current_events()
                ### if event triggered end --> end simulation / skip rest
                if self.end: continue
                ### check then next events occur
                next_events_time = self.get_next_events_time()
                ### check if there are model triggers
                self.model_trigger_list = self.get_model_trigger_list()
                ### simulate until next event(s) or model triggers
                step_before=get_current_step()
                if self.print_outs: print('check_triggers:', self.model_trigger_list)
                if len(self.model_trigger_list)>1:
                    ### multiple model triggers
                    simulate_until(max_duration=next_events_time, population=[get_population(pop_name) for pop_name in self.model_trigger_list], operator='or')
                elif len(self.model_trigger_list)>0:
                    ### a single model trigger
                    simulate_until(max_duration=next_events_time, population=get_population(self.model_trigger_list[0]))
                else:
                    ### no model_triggers
                    simulate(next_events_time)
                step_after=get_current_step()

        def run_current_events(self):
            """
                run all events with start == current step
            """
            ### run all events of the current step
            ### repeat this until no event was run, because events can set the onset of other events to the current step
            ### due to repeat --> prevent that same event is run twice
            event_run = True
            while event_run:
                event_run = False
                for event in self.event_list:
                    if event.onset == get_current_step() and not(event.name in self.run_event_list) and event.check_requirements():
                        event.run()
                        event_run=True
                        self.run_event_list.append(event.name)
            
        def run_model_trigger_events(self):
            """
                check the current model triggers stored in self.model_trigger_list
                if they are activated --> run corresponding events
                prevent that these model triggers are stored again in self.model_trigger_list
            """
            ### loop to check if model trigger got active
            for model_trigger in self.model_trigger_list:
                if int(get_population(model_trigger).decision[0]) == -1:
                    ### -1 means git active
                    ### find the events triggerd by the model_trigger and run them
                    for event in self.event_list:
                        if event.model_trigger == model_trigger:
                            event.run()
                            self.run_event_list.append(event.name)
                    ### prevent that these model_triggers are used again
                    self.past_model_trigger_list.append(model_trigger)
                    

        def get_next_events_time(self):
            """ 
                go through all events and get onsets
                get onset which are > current_step
                return smallest diff in ms (ms value = full timesteps!)
            """
            next_event_time = np.inf
            for event in self.event_list:
                ### skip events without onset
                if event.onset==None: continue
                ### check if onset in the future and nearest
                if event.onset>get_current_step() and (event.onset-get_current_step())<next_event_time:
                    next_event_time = event.onset-get_current_step()
            ### return difference (simulation duration until nearest next event) in ms, round to full timesteps
            return round(next_event_time*dt(),get_number_of_zero_decimals(dt()))

        def get_model_trigger_list(self):
            """
                check if there are events with model_triggers
                check if these model triggers already happened
                check if the requirements of the events are met
                not happend + requirements met --> add model_trigger to model_trigger_list
                returns the (new) model_trigger_list
            """
            ret = []
            for event in self.event_list:
                if event.model_trigger!=None:
                    if not(event.model_trigger in self.past_model_trigger_list) and event.check_requirements():
                        ret.append(event.model_trigger)
            return ret
            
        def end_sim(self):
            self.end=True
            
            
        class event_cl:
    
            def __init__(self, trial_procedure, name, onset=None, model_trigger=None, requirement_string=None, effect=None, trigger=None):
                """
                    trial_procedure: outer class
                    onset: time as timesteps
                    model_trigger: name of population which can trigger the event (by decision)
                    effect: function; what happens during event
                    trigger: dict; what other events are triggerd (keys) and when relative to onset of current event (vals)
                """
                self.trial_procedure=trial_procedure; self.name=name; self.onset=onset; self.model_trigger=model_trigger; self.requirement_string=requirement_string; self.effect=effect; self.trigger=trigger

            def run(self):
                ### check requirements
                if self.check_requirements():
                    ### run the event
                    if self.trial_procedure.print_outs:print('run event:',self.name, get_time())
                    ### for events which are triggered by model --> set onset
                    if self.onset==None: self.onset=get_current_step()
                    ### run the effect
                    if self.effect != None:
                        self.effect(self)
                    ### trigger other events
                    if self.trigger != None:
                        ### loop over all triggered events
                        for name, delay in self.trigger.items():
                            ### get the other event
                            event_idx = self.trial_procedure.event_name_list.index(name)
                            ### set onset of other event
                            self.trial_procedure.event_list[event_idx].onset = self.onset + delay
                    ### store event in happened events
                    self.trial_procedure.happened_event_list.append(self.name)
                    
            def check_requirements(self):
                if self.requirement_string!=None:
                    ### check requirement with requirement string
                    return self.eval_requirement_string()
                else:
                    ### no requirement
                    return True
                
            def eval_requirement_string(self):
                """
                    evaluates a condition string in format like 'XXX==XXX and (XXX==XXX or XXX==XXX)'
                    
                    returns True/False
                """
                ### split condition string
                string = self.requirement_string
                string = string.split(' and ')
                string = [sub_string.split(' or ') for sub_string in string]

                ### loop over string splitted string parts
                final_string=[]
                for sub_idx, sub_string in enumerate(string):
                    ### combine outer list eelemts with and
                    ### and combine inner list elements with or
                    if len(sub_string)==1:
                        if sub_idx < len(string)-1:
                            final_string.append(self.get_condition_part(sub_string[0])+' and ')
                        else:
                            final_string.append(self.get_condition_part(sub_string[0]))
                    else:
                        for sub_sub_idx, sub_sub_string in enumerate(sub_string):
                            if sub_sub_idx < len(sub_string)-1:
                                final_string.append(self.get_condition_part(sub_sub_string)+' or ')
                            elif sub_idx < len(string)-1:
                                final_string.append(self.get_condition_part(sub_sub_string)+' and ')
                            else:
                                final_string.append(self.get_condition_part(sub_sub_string))
                return eval(''.join(final_string))
                            
            def get_condition_part(self, string):
                """
                    converts a string in format like '((XXX==XXX)' into '((True)'
                """
                ### remove spaces from string
                string = string.strip()
                string = string.split()
                string = ''.join(string)
                
                ### recursively remove brackets
                ### at the end evaluate term (without brackets) and then return the evaluated value with the former brackets
                if string[0]=='(':
                    return '('+self.get_condition_part(string[1:])
                elif string[-1]==')':
                    return self.get_condition_part(string[:-1])+')'
                else:
                    return str(self.eval_condition_part(string))
                    
            def eval_condition_part(self, string):
                """
                    gets string in format 'XXX==XXX'
                    
                    evaluates the term for mode and happened events
                    
                    returns True/False
                """

                var = string.split('==')[0]
                val = string.split('==')[1]
                if var == 'mode':
                    test = self.trial_procedure.mode == val
                elif var == 'happened_event_list':
                    ### remove brackets
                    val = val.strip("[]")
                    ### split entries
                    val = val.split(',')
                    ### remove spaces from entries
                    happened_event_list_from_string = [val_val.strip() for val_val in val]
                    ### check if all events are in happened_event_list, if not --> return False
                    test = True
                    for event in happened_event_list_from_string:
                        if not(event in self.trial_procedure.happened_event_list): test=False
                return test
