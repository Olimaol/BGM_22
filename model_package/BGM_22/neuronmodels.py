from ANNarchy import Neuron


izhikevich2007_standard = Neuron(
    parameters="""
        C        = 0 : population
        k        = 0 : population
        v_r      = 0 : population
        v_t      = 0 : population
        a        = 0 : population
        b        = 0 : population
        c        = 0 : population
        d        = 0 : population
        v_peak   = 0 : population
        tau_ampa = 1 : population
        tau_gaba = 1 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population
        I_add    = 0
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2007_standard",
    description = "Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents."
)


izhikevich2007_standard_new_noise = Neuron(
    parameters="""
        C              = 0 : population
        k              = 0 : population
        v_r            = 0 : population
        v_t            = 0 : population
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        v_peak         = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        tau_noise      = 0 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        E_noise        = 0 : population
        I_add          = 0
        prob_noise_event     = 0 : population
        increase_noise = 0 : population
        isi_variations = 0
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        dg_noise/dt = ite(next_noise_event > t, -g_noise/tau_noise, -g_noise/tau_noise + increase_noise/dt)
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba) - g_noise*(v - E_noise)
        du/dt      = a*(b*(v - v_r) - u)
        
        next_noise_event = ite(next_noise_event > t, next_noise_event, next_noise_event + NegBinomial(1,prob_noise_event)*dt + isi_variations) : min=t+dt
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2007_standard",
    description = "Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents."
)


izhikevich2007_fsi = Neuron(
    parameters="""
        C        = 0 : population
        k        = 0 : population
        v_r      = 0 : population
        v_t      = 0 : population
        v_b      = 0 : population
        a        = 0 : population
        b        = 0 : population
        c        = 0 : population
        d        = 0 : population
        v_peak   = 0 : population
        tau_ampa = 1 : population
        tau_gaba = 1 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population
        I_add    = 0
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = if v<v_b: -a * u else: a * (b * (v - v_b)**3 - u)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2007_fsi",
    description = "Fast spiking cortical interneuron model from Izhikevich (2007) with additional conductance based synapses."
)


izhikevich2003 = Neuron(
    parameters = """    
        a        = 0 : population 
        b        = 0 : population
        c        = 0 : population
        d        = 0 : population
        tau_ampa = 1 : population
        tau_gaba = 1 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population 
        I_add    = 0
    """,
    equations = """
        dg_ampa/dt = -g_ampa / tau_ampa
        dg_gaba/dt = -g_gaba / tau_gaba
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = I_add + I_ampa + I_gaba            
        dv/dt      = 0.04 * v * v + 5 * v + 140 - u + I : init = -70 
        du/dt      = a * (b * v - u)                    : init = -18.55
    """,
    spike = """
        v >= 30
    """,
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2003",
    description = "Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents."
)


izhikevich2003_modified = Neuron(
    parameters = """    
        a        = 0 : population 
        b        = 0 : population
        c        = 0 : population
        d        = 0 : population
        n2       = 0 : population
        n1       = 0 : population
        n0       = 0 : population
        tau_ampa = 1 : population
        tau_gaba = 1 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population 
        I_add    = 0
    """,
    equations = """
        dg_ampa/dt = -g_ampa / tau_ampa
        dg_gaba/dt = -g_gaba / tau_gaba
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = I_add + I_ampa + I_gaba            
        dv/dt      = n2 * v * v + n1 * v + n0 - u + I : init = -70 
        du/dt      = a * (b * v - u)                  : init = -18.55
    """,
    spike = """
        v >= 30
    """,
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2003_modified",
    description = "Modified neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents."
)


poisson_neuron_up_down = Neuron(        
    parameters ="""
        rates   = 0
        tau_up   = 1 : population
        tau_down = 1 : population
    """,
    equations ="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt
        dact/dt = if (rates - act) > 0:
                      (rates - act) / tau_up
                  else:
                      (rates - act) / tau_down        
    """,
    spike ="""    
        p <= act
    """,    
    reset ="""    
        p = 0.0
    """,
    name = "poisson_neuron_up_down",
    description = "Poisson neuron whose rate can be specified and is reached with time constants tau_up and tau_down."
)


poisson_neuron = Neuron(        
    parameters ="""
        rates   = 0
    """,
    equations ="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt      
    """,
    spike ="""    
        p <= rates
    """,    
    reset ="""    
        p = 0.0
    """,
    name = "poisson_neuron",
    description = "Poisson neuron whose rate can be specified and is reached instanteneous."
)

integrator_neuron = Neuron(        
    parameters = """
        tau       = 1 : population
        threshold = 0 : population
        neuron_id = 0
    """,
    equations = """
        dg_ampa/dt = - g_ampa / tau
        ddecision/dt = 0
    """,
    spike = """
        g_ampa >= threshold        
    """,
    reset = """
        decision = neuron_id
    """,
    name = "integrator_neuron",
    description = "Integrator Neuron, which integrates incoming spikes with value g_ampa and emits a spike when reaching a threshold. After spike decision changes, which can be used as stop condition"
)
