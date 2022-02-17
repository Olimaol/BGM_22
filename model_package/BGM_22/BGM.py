from CompNeuroPy.model_functions import compile_in_folder
from ANNarchy import Population, Projection, Uniform, populations, projections, get_population, compile
from BGM_22.neuronmodels import poisson_neuron_up_down, izhikevich2007_standard, izhikevich2007_fsi, izhikevich2003, integrator_neuron
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
    gpe_proto = Population(params['general_population_size'], izhikevich2003, name="gpe_proto")
    gpe_arky  = Population(params['general_population_size'], izhikevich2003, name="gpe_arky")
    gpe_cp    = Population(params['general_population_size'], izhikevich2003, name="gpe_cp")
    thal      = Population(params['general_population_size'], izhikevich2003, name="thal")
    
    ### integrator Neurons
    integrator_go   = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_go")
    integrator_stop = Population(1, integrator_neuron, stop_condition="decision == -1", name="integrator_stop")

    set_params(params)
    quit()
    ###TODO works until here

    ### Noise / Baseline Input populations
    #GPe noise
    gpe_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="gpe_n_exc")
    gpe_n_exc.rates   = 0
    gpe_n_exc.act     = 0
    #GPe-Proto noise
    gpe_proto_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="gpe_proto_n_exc")
    gpe_proto_n_exc__values = np.random.normal(params['gpe_n_exc__rates'],params['gpe_n_exc__sd'],params['general_population_size'])
    gpe_proto_n_exc.rates   = gpe_proto_n_exc__values
    gpe_proto_n_exc.act     = gpe_proto_n_exc__values
    #GPe-Cp noise
    gpe_cp_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="gpe_cp_n_exc")
    gpe_cp_n_exc__values = np.random.normal(params['gpe_n_exc__rates'],params['gpe_n_exc__sd'],params['general_population_size'])
    gpe_cp_n_exc.rates   = gpe_cp_n_exc__values
    gpe_cp_n_exc.act     = gpe_cp_n_exc__values
    #GPe-Arky noise
    gpe_arky_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="gpe_arky_n_exc")
    gpe_arky_n_exc__values = np.random.normal(params['gpe_n_exc__rates'],params['gpe_n_exc__sd'],params['general_population_size'])
    gpe_arky_n_exc.rates   = gpe_arky_n_exc__values
    gpe_arky_n_exc.act     = gpe_arky_n_exc__values
    #snr noise
    snr_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="snr_n_exc")
    snr_n_exc__values = np.random.normal(params['snr_n_exc__rates'],params['snr_n_exc__sd'],params['general_population_size'])
    snr_n_exc.rates   = snr_n_exc__values
    snr_n_exc.act     = snr_n_exc__values
    #stn noise
    stn_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="stn_n_exc")
    stn_n_exc__values = np.random.normal(params['stn_n_exc__rates'],params['stn_n_exc__sd'],params['general_population_size'])
    stn_n_exc.rates   = stn_n_exc__values
    stn_n_exc.act     = stn_n_exc__values
    #Str noise
    str_n_exc       = Population(params['general_population_size'], poisson_neuron_up_down, name="str_n_exc")
    str_n_exc.rates = 0
    str_n_exc.act   = 0
    #StrD1 noise
    str_d1_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="str_d1_n_exc")
    str_d1_n_exc__values = np.random.normal(params['str_d1_n_exc__rates'],params['str_d1_n_exc__sd'],params['general_population_size'])
    str_d1_n_exc.rates   = str_d1_n_exc__values
    str_d1_n_exc.act     = str_d1_n_exc__values
    #StrD2 noise
    str_d2_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="str_d2_n_exc")
    str_d2_n_exc__values = np.random.normal(params['str_d2_n_exc__rates'],params['str_d2_n_exc__sd'],params['general_population_size'])
    str_d2_n_exc.rates   = str_d2_n_exc__values
    str_d2_n_exc.act     = str_d2_n_exc__values
    #StrFSI noise
    str_fsi_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="str_fsi_n_exc")
    str_fsi_n_exc__values = np.random.normal(params['str_fsi_n_exc__rates'],params['str_fsi_n_exc__sd'],params['general_population_size'])
    str_fsi_n_exc.rates   = str_fsi_n_exc__values
    str_fsi_n_exc.act     = str_fsi_n_exc__values
    #thalamus noise
    thal_n_exc         = Population(params['general_population_size'], poisson_neuron_up_down, name="thalNoise")
    thal_n_exc__values = np.random.normal(params['thalE_rates'],params['thalE_sd'],params['general_population_size'])
    thal_n_exc.rates   = thal_n_exc__values
    thal_n_exc.act     = thal_n_exc__values
        
        
    ### PROJECTIONS
    ### cor_go outputs

    
    set_params(params)
    
    if do_compile:
        compile_in_folder(compile_folder_name)
    
    return {'populations':[ pop.name for pop in populations() ], 'projections':[ proj.name for proj in projections() ]}
