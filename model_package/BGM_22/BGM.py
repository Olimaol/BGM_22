import numpy as np
from CompNeuroPy.model_functions import compile_in_folder
from ANNarchy import *
from BGM_22.neuronmodels import poisson_neuron_up_down, poisson_neuron, izhikevich2007_standard, izhikevich2007_fsi, izhikevich2003, integrator_neuron, izhikevich2003_modified
from BGM_22.synapsemodels import factor_synapse
from BGM_22.sim_params import get_params


def set_params(params):
    """
        sets params of all populations
        
        params: dict with params
    """
    
    population_name_list = [ pop.name for pop in populations() ]

    ### loop over all params
    for key, val in params.items():
    
        ### split param key in pop and param name
        key_split=key.split('__')
        if len(key_split)>=2:
            pop_name = key_split[0]
            param_name = key_split[1]
            
            if param_name.split('_')[-1]=='init':
                param_name='_'.join(param_name.split('_')[:-1])
            
            ### if pop is in network --> set param
            if pop_name in population_name_list:
                setattr(get_population(pop_name), param_name, val)
                
                
def set_noise_values(params):
    """
        sets the noise values for all noise populations
        
        params: dict with params
    """
    
    population_name_list = [ pop.name for pop in populations() ]
    
    ### loop over all populations
    for pop_name in population_name_list:
        ### check if its noise population
        if '_'.join(pop_name.split('_')[-2:])=='n_exc' or '_'.join(pop_name.split('_')[-2:])=='n_inh':
            ### search both noise params and get population size
            pop_size=get_population(pop_name).size
            try:
                mean=params[pop_name+'__mean']
                sd=params[pop_name+'__sd']
            except:
                print('\nERROR: missing noise parameters for',pop_name,'\n',pop_name+'__mean', pop_name+'__sd', 'needed!\n','parameters id:', params['general__id'],'\n')
                quit()
            ### set values
            get_population(pop_name).rates = np.random.normal(mean, sd, pop_size)
            
            
def set_connections(params):
    """
        sets the connectivity and parameters of all projections
        
        params: dict with params
    """
    
    projection_name_list = [ proj.name for proj in projections() ]
    already_set_params = {}# dict for each projection, which params were already set during connectivity definition
    
    ### set connectivity
    ### loop over all projections
    for proj_name in projection_name_list:
        ### get the type of connectivity for projection
        try:
            connectivity = params[proj_name+'__connectivity']
        except:
            print('\nERROR: missing connectivity parameter for',proj_name,'\n',proj_name+'__connectivity', 'needed!\n','parameters id:', params['general__id'],'\n')
            quit()
            
        if connectivity=='connect_fixed_number_pre':
            get_projection(proj_name).connect_fixed_number_pre(number=params[proj_name+'__nr_con'], weights=eval(str(params[proj_name+'__weights'])), delays=eval(str(params[proj_name+'__delays'])))
            already_set_params[proj_name] = ['connectivity', 'nr_con', 'weights', 'delays']
        elif connectivity=='connect_all_to_all':
            get_projection(proj_name).connect_all_to_all(weights=eval(str(params[proj_name+'__weights'])), delays=eval(str(params[proj_name+'__delays'])))
            already_set_params[proj_name] = ['connectivity', 'weights', 'delays']
        else:
            print('\nERROR: wrong connectivity parameter for',proj_name+'__connectivity!\n','parameters id:', params['general__id'],'\n')
            quit()
            

    ### set parameters
    ### loop over all params
    for key, val in params.items():
    
        ### split param key in proj and param name
        key_split=key.split('__')
        if len(key_split)>=3:
            proj_name = '__'.join(key_split[:2])
            param_name = key_split[2]
                        
            ### if proj is in network --> set param
            if proj_name in projection_name_list and not(param_name in already_set_params[proj_name]):
                setattr(get_projection(proj_name), param_name, val)


def BGM(do_compile=False, compile_folder_name='annarchy_BGM'):
    """
        generates a basal ganglia model (model_id = BGM) and optionally compiles the network
        
        returns a list of the names of the populations an projections (for later access)
    """
    
    params = get_params('BGM')
        
    ################################################################################################################################
    ####################################################   POPULATIONS   ###########################################################
    ################################################################################################################################

    ### cortex / input populations
    cor_go    = Population(params['general__population_size'], poisson_neuron_up_down, name="cor_go")
    cor_pause = Population(params['general__population_size'], poisson_neuron_up_down, name="cor_pause")
    cor_stop  = Population(params['general__population_size'], poisson_neuron_up_down, name="cor_stop")

    ### Str Populations
    str_d1  = Population(params['general__population_size'], izhikevich2007_standard, name="str_d1")
    str_d2  = Population(params['general__population_size'], izhikevich2007_standard, name="str_d2")
    str_fsi = Population(params['general__population_size'], izhikevich2007_fsi, name="str_fsi")
    
    ### BG Populations
    stn       = Population(params['general__population_size'], izhikevich2003, name="stn")
    snr       = Population(params['general__population_size'], izhikevich2003, name="snr")
    gpe_proto = Population(params['general__population_size'], izhikevich2003_modified, name="gpe_proto")
    gpe_arky  = Population(params['general__population_size'], izhikevich2003_modified, name="gpe_arky")
    gpe_cp    = Population(params['general__population_size'], izhikevich2003_modified, name="gpe_cp")
    thal      = Population(params['general__population_size'], izhikevich2003, name="thal")
    
    ### integrator Neurons
    integrator_go   = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_go")
    integrator_stop = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_stop")
    
    ### Noise
    str_d1_n_exc    = Population(params['general__population_size'], poisson_neuron, name="str_d1_n_exc")
    str_d2_n_exc    = Population(params['general__population_size'], poisson_neuron, name="str_d2_n_exc")
    str_fsi_n_exc   = Population(params['general__population_size'], poisson_neuron, name="str_fsi_n_exc")
    stn_n_exc       = Population(params['general__population_size'], poisson_neuron, name="stn_n_exc")
    snr_n_exc       = Population(params['general__population_size'], poisson_neuron, name="snr_n_exc")
    gpe_proto_n_exc = Population(params['general__population_size'], poisson_neuron, name="gpe_proto_n_exc")
    gpe_arky_n_exc  = Population(params['general__population_size'], poisson_neuron, name="gpe_arky_n_exc")
    gpe_cp_n_exc    = Population(params['general__population_size'], poisson_neuron, name="gpe_cp_n_exc")
    thal_n_exc      = Population(params['general__population_size'], poisson_neuron, name="thal_n_exc")
    
    
    ################################################################################################################################
    ####################################################   PROJECTIONS   ###########################################################
    ################################################################################################################################
    
    ### cortex go output
    cor_go__str_d1  = Projection(pre=cor_go, post=str_d1,  target='ampa', synapse=factor_synapse, name='cor_go__str_d1')
    cor_go__str_d2  = Projection(pre=cor_go, post=str_d2,  target='ampa', synapse=factor_synapse, name='cor_go__str_d2')
    cor_go__str_fsi = Projection(pre=cor_go, post=str_fsi, target='ampa', synapse=factor_synapse, name='cor_go__str_fsi')
    cor_go__thal    = Projection(pre=cor_go, post=thal,    target='ampa', synapse=factor_synapse, name='cor_go__thal')
    
    ### cortex stop output
    cor_stop__gpe_arky = Projection(pre=cor_stop, post=gpe_arky, target='ampa', synapse=factor_synapse, name='cor_stop__gpe_arky')
    cor_stop__gpe_cp   = Projection(pre=cor_stop, post=gpe_cp,   target='ampa', synapse=factor_synapse, name='cor_stop__gpe_cp')
    
    ### cortex pause output
    cor_pause__stn = Projection(pre=cor_pause, post=stn, target='ampa', synapse=factor_synapse, name='cor_pause__stn')
    
    ### str d1 output
    str_d1__snr    = Projection(pre=str_d1, post=snr,    target='gaba', synapse=factor_synapse, name='str_d1__snr')
    str_d1__gpe_cp = Projection(pre=str_d1, post=gpe_cp, target='gaba', synapse=factor_synapse, name='str_d1__gpe_cp')
    str_d1__str_d1 = Projection(pre=str_d1, post=str_d1, target='gaba', synapse=factor_synapse, name='str_d1__str_d1')
    str_d1__str_d2 = Projection(pre=str_d1, post=str_d2, target='gaba', synapse=factor_synapse, name='str_d1__str_d2')
    
    ### str d2 output
    str_d2__gpe_proto = Projection(pre=str_d2, post=gpe_proto, target='gaba', synapse=factor_synapse, name='str_d2__gpe_proto')
    str_d2__gpe_arky  = Projection(pre=str_d2, post=gpe_arky,  target='gaba', synapse=factor_synapse, name='str_d2__gpe_arky')
    str_d2__gpe_cp    = Projection(pre=str_d2, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='str_d2__gpe_cp')
    str_d2__str_d1    = Projection(pre=str_d2, post=str_d1,    target='gaba', synapse=factor_synapse, name='str_d2__str_d1')
    str_d2__str_d2    = Projection(pre=str_d2, post=str_d2,    target='gaba', synapse=factor_synapse, name='str_d2__str_d2')
    
    ### str fsi output
    str_fsi__str_d1  = Projection(pre=str_fsi, post=str_d1,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d1')
    str_fsi__str_d2  = Projection(pre=str_fsi, post=str_d2,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d2')
    str_fsi__str_fsi = Projection(pre=str_fsi, post=str_fsi, target='gaba', synapse=factor_synapse, name='str_fsi__str_fsi')
    
    ### stn output
    stn__snr       = Projection(pre=stn, post=snr,       target='ampa', synapse=factor_synapse, name='stn__snr')
    stn__gpe_proto = Projection(pre=stn, post=gpe_proto, target='ampa', synapse=factor_synapse, name='stn__gpe_proto')
    stn__gpe_arky  = Projection(pre=stn, post=gpe_arky,  target='ampa', synapse=factor_synapse, name='stn__gpe_arky')
    stn__gpe_cp    = Projection(pre=stn, post=gpe_cp,    target='ampa', synapse=factor_synapse, name='stn__gpe_cp')
    
    ### gpe proto output
    gpe_proto__stn      = Projection(pre=gpe_proto, post=stn,      target='gaba', synapse=factor_synapse, name='gpe_proto__stn')
    gpe_proto__snr      = Projection(pre=gpe_proto, post=snr,      target='gaba', synapse=factor_synapse, name='gpe_proto__snr')
    gpe_proto__gpe_arky = Projection(pre=gpe_proto, post=gpe_arky, target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_arky')
    gpe_proto__str_fsi  = Projection(pre=gpe_proto, post=str_fsi,  target='gaba', synapse=factor_synapse, name='gpe_proto__str_fsi')
    gpe_proto__gpe_cp   = Projection(pre=gpe_proto, post=gpe_cp,   target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_cp')
    
    ### gpe arky output
    gpe_arky__str_d1    = Projection(pre=gpe_arky, post=str_d1,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d1')
    gpe_arky__str_d2    = Projection(pre=gpe_arky, post=str_d2,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d2')
    gpe_arky__gpe_proto = Projection(pre=gpe_arky, post=gpe_proto, target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_proto')
    gpe_arky__str_fsi   = Projection(pre=gpe_arky, post=str_fsi,   target='gaba', synapse=factor_synapse, name='gpe_arky__str_fsi')
    gpe_arky__gpe_cp    = Projection(pre=gpe_arky, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_cp')
    
    ### gpe cp output
    gpe_cp__gpe_proto       = Projection(pre=gpe_cp, post=gpe_proto, target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_proto')
    gpe_cp__str_d1          = Projection(pre=gpe_cp, post=str_d1,    target='gaba', synapse=factor_synapse, name='gpe_cp__str_d1')
    gpe_cp__str_d2          = Projection(pre=gpe_cp, post=str_d2,    target='gaba', synapse=factor_synapse, name='gpe_cp__str_d2')
    gpe_cp__str_fsi         = Projection(pre=gpe_cp, post=str_fsi,   target='gaba', synapse=factor_synapse, name='gpe_cp__str_fsi')
    gpe_cp__gpe_arky        = Projection(pre=gpe_cp, post=gpe_arky,  target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_arky')
    gpe_cp__integrator_stop = Projection(pre=gpe_cp, post=gpe_arky,  target='ampa', synapse=factor_synapse, name='gpe_cp__integrator_stop')
    
    ### snr output
    snr__thal = Projection(pre=snr, post=thal, target='gaba', synapse=factor_synapse, name='snr__thal')
    
    ### thal output
    thal__integrator_go = Projection(pre=thal, post=integrator_go, target='ampa', synapse=factor_synapse, name='thal__integrator_go')
    thal__str_d1        = Projection(pre=thal, post=str_d1,        target='ampa', synapse=factor_synapse, name='thal__str_d1')
    thal__str_d2        = Projection(pre=thal, post=str_d2,        target='ampa', synapse=factor_synapse, name='thal__str_d2')
    thal__str_fsi       = Projection(pre=thal, post=str_fsi,       target='ampa', synapse=factor_synapse, name='thal__str_fsi')
    
    
    
    set_params(params)
    set_noise_values(params)
    set_connections(params)
    compile()
    print(vars(thal__integrator_go))
    quit()
    ####TODO works until here

    ### Noise/Baseline inputs
    GPeEGPe_Proto = Projection(
        pre  = GPeE,
        post = GPe_Proto,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    GPeEGPe_Proto.mod_factor = 0

    GPEGPe_Arky = Projection(
        pre  = GPeE,
        post = GPe_Arky,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    GPEGPe_Arky.mod_factor = 0

    SNrESNr = Projection(
        pre  = SNrE,
        post = SNr,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    SNrESNr.mod_factor = params['weights_SNrE_SNr']

    STNESTN = Projection(
        pre  = STNE,
        post = STN,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    STNESTN.mod_factor = params['weights_STNE_STN']

    STRESTR_D1 = Projection(
        pre  = ED1,
        post = STR_D1,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    STRESTR_D1.mod_factor = params['weights_StrD1E_StrD1']

    STRESTR_D2 = Projection(
        pre  = ED2,
        post = STR_D2,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    STRESTR_D2.mod_factor = params['weights_StrD2E_StrD2']

    TestThalnoiseThal = Projection(     
        pre = TestThalnoise,
        post = Thal,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1)
    TestThalnoiseThal.mod_factor = params['weights_ThalE_Thal']

    STRESTR_FSI = Projection(
        pre  = EFSI,
        post = STR_FSI,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    STRESTR_FSI.mod_factor = params['weights_StrFSIE_StrFSI']

    EProto1GPe_Proto = Projection(
        pre  = EProto1,
        post = GPe_Proto,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    EProto1GPe_Proto.mod_factor = params['weights_GPeProtoE_GPeProto']

    EProto2GPe_Proto2 = Projection(
        pre  = EProto2,
        post = GPe_Proto2,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    EProto2GPe_Proto2.mod_factor = params['weights_GPeCpE_GPeCp']

    EArkyGPe_Arky = Projection(
        pre  = EArky,
        post = GPe_Arky,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    EArkyGPe_Arky.mod_factor = params['weights_GPeArkyE_GPeArky']

    EArkyGPe_Arky2 = Projection(
        pre  = EArky,
        post = GPe_Arky2,
        target = 'ampa',
        synapse = FactorSynapse
    ).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general__synDelays']))
    EArkyGPe_Arky2.mod_factor = params['weights_GPeArkyE_GPeArky']*params['GPeArkyCopy_On']


    
    
    
    
    
    
    
    
    
    

    
    
    if do_compile:
        compile_in_folder(compile_folder_name)
    
    return {'populations':[ pop.name for pop in populations() ], 'projections':[ proj.name for proj in projections() ]}
