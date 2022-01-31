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
        tau_ampa = 0 : population
        tau_gaba = 0 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population
        I_add    = 0 : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = I_add + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2007_rs",
    description = "Standard neuron model from Izhikevich (2007) with additional conductance based synapses."
)


izhikevich2007_fs = Neuron(
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
        tau_ampa = 0 : population
        tau_gaba = 0 : population
        E_ampa   = 0 : population
        E_gaba   = 0 : population
        I_add    = 0 : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = I_add + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I
        U_v        = if v<v_b: 0 else: b*(v - v_b)**3
        du/dt      = a*(U_v - u)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "izhikevich2007_fs",
    description = "Fast spiking cortical interneuron model from Izhikevich (2007) with additional conductance based synapses."
)


poisson_neuron_up_down = Neuron(        
    parameters ="""
        rates   = 0
        tau_up   = 0 : population
        tau_down = 0 : population
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
