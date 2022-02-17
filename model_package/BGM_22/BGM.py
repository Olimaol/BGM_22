import numpy as np
from CompNeuroPy.model_functions import compile_in_folder
from ANNarchy import Population, Projection, Uniform, populations, projections, get_population, compile
from BGM_22.neuronmodels import poisson_neuron_up_down, poisson_neuron, izhikevich2007_standard, izhikevich2007_fsi, izhikevich2003, integrator_neuron, izhikevich2003_modified
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
                print('\nERROR: missing noise parameters for',pop_name)
                print(pop_name+'__mean', pop_name+'__sd', 'needed!')
                print('parameters id:', params['general_id'],'\n')
                quit()
            print(pop_name, (15-len(pop_name))*' ', pop_size,'\t', mean,'\t', sd)
            get_population(pop_name).rates = np.random.normal(mean, sd, pop_size)
                
                

def BGM(do_compile=False, compile_folder_name='annarchy_BGM'):
    """
        generates a basal ganglia model (model_id = BGM) and optionally compiles the network
        
        returns a list of the names of the populations an projections (for later access)
    """
    
    params = get_params('BGM')
    
    
    ### POPULATIONS
    ### cortex / input populations
    cor_go    = Population(params['general_population_size'], poisson_neuron_up_down, name="cor_go")
    cor_pause = Population(params['general_population_size'], poisson_neuron_up_down, name="cor_pause")
    cor_stop  = Population(params['general_population_size'], poisson_neuron_up_down, name="cor_stop")

    ### Str Populations
    str_d1  = Population(params['general_population_size'], izhikevich2007_standard, name="str_d1")
    str_d2  = Population(params['general_population_size'], izhikevich2007_standard, name="str_d2")
    str_fsi = Population(params['general_population_size'], izhikevich2007_fsi, name="str_fsi")
    
    ### BG Populations
    stn       = Population(params['general_population_size'], izhikevich2003, name="stn")
    snr       = Population(params['general_population_size'], izhikevich2003, name="snr")
    gpe_proto = Population(params['general_population_size'], izhikevich2003_modified, name="gpe_proto")
    gpe_arky  = Population(params['general_population_size'], izhikevich2003_modified, name="gpe_arky")
    gpe_cp    = Population(params['general_population_size'], izhikevich2003_modified, name="gpe_cp")
    thal      = Population(params['general_population_size'], izhikevich2003, name="thal")
    
    ### integrator Neurons
    integrator_go   = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_go")
    integrator_stop = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_stop")
    
    ### Noise
    str_d1_n_exc    = Population(params['general_population_size'], poisson_neuron, name="str_d1_n_exc")
    str_d2_n_exc    = Population(params['general_population_size'], poisson_neuron, name="str_d2_n_exc")
    str_fsi_n_exc   = Population(params['general_population_size'], poisson_neuron, name="str_fsi_n_exc")
    stn_n_exc       = Population(params['general_population_size'], poisson_neuron, name="stn_n_exc")
    snr_n_exc       = Population(params['general_population_size'], poisson_neuron, name="snr_n_exc")
    gpe_proto_n_exc = Population(params['general_population_size'], poisson_neuron, name="gpe_proto_n_exc")
    gpe_arky_n_exc  = Population(params['general_population_size'], poisson_neuron, name="gpe_arky_n_exc")
    gpe_cp_n_exc    = Population(params['general_population_size'], poisson_neuron, name="gpe_cp_n_exc")
    thal_n_exc      = Population(params['general_population_size'], poisson_neuron, name="thal_n_exc")
    
    
    set_params(params)
    set_noise_values(params)
    
    ####TODO works until here

        
        
    ### PROJECTIONS
    ### cor_go outputs

    
    
    if do_compile:
        compile_in_folder(compile_folder_name)
    
    return {'populations':[ pop.name for pop in populations() ], 'projections':[ proj.name for proj in projections() ]}
