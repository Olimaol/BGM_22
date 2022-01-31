from ANNarchy import*
from BGmodelSST.sim_params import params

Izhikevich_neuron = Neuron(
    parameters = """    
        a        = 'IzhikevichNeuron_a'  : population 
        b        = 'IzhikevichNeuron_b'  : population
        c        = 'IzhikevichNeuron_c'  : population
        d        = 'IzhikevichNeuron_d'  : population
        n0       = 'IzhikevichNeuron_n0' : population 
        n1       = 'IzhikevichNeuron_n1' : population
        n2       = 'IzhikevichNeuron_n2' : population
        I        = 'IzhikevichNeuron_I'  : population 
        tau_ampa = 'general_tauAMPA'             : population
        tau_gaba = 'general_tauGABA'             : population
        E_ampa   = 'general_EAMPA'               : population
        E_gaba   = 'general_EGABA'               : population 
    """,
    equations = """
        dg_ampa/dt = -g_ampa / tau_ampa                                                               : init = 0
        dg_gaba/dt = -g_gaba / tau_gaba                                                               : init = 0
        dv/dt      = n2 * v * v + n1 * v + n0 - u + I - g_ampa * (v - E_ampa) - g_gaba * (v - E_gaba) : init = -70 
        du/dt      = a * (b * (v) - u)                                                                : init = -18.55
    """,
    spike = """
        v >= 30
    """,
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich Neuron",
    description = "Izhikevich Neuron with conductance based synapses.",
    extra_values = params)

Izhikevich_STR_neuron = Neuron(         
    parameters ="""
        a        = 'IzhikevichStrNeuron_a'  : population
        b        = 'IzhikevichStrNeuron_b'  : population
        c        = 'IzhikevichStrNeuron_c'  : population
        d        = 'IzhikevichStrNeuron_d'  : population
        n0       = 'IzhikevichStrNeuron_n0' : population
        n1       = 'IzhikevichStrNeuron_n1' : population
        n2       = 'IzhikevichStrNeuron_n2' : population
        I        = 'IzhikevichStrNeuron_I'  : population
        tau_ampa = 'general_tauAMPA'        : population
        tau_gaba = 'general_tauGABA'        : population
        E_ampa   = 'general_EAMPA'          : population
        E_gaba   = 'general_EGABA'          : population
        Vr       = 'IzhikevichStrNeuron_Vr' : population
        C        = 'IzhikevichStrNeuron_C'  : population
    """,
    equations ="""
        dg_ampa/dt = -g_ampa / tau_ampa                                                                       : init = 0
        dg_gaba/dt = -g_gaba / tau_gaba                                                                       : init = 0
        dv/dt      = n2 * v * v + n1 * v + n0 - u / C + I / C - g_ampa * (v - E_ampa) - g_gaba * (v - E_gaba) : init = -70
        du/dt      = a *(b*(v-Vr)-u)                                                                          : init = -18.55
    """,
    spike ="""    
        v >= 40
    """,
    reset ="""
        v = c
        u = u + d
    """,
    name = "Izhikevich Str Neuron",
    description = "Izhikevich Neuron with conductance based synapses for Str neurons.",
    extra_values = params)

STR_FSI_neuron = Neuron(
    parameters="""
        a        = 'IzhikevichStrFSINeuron_a'     : population
        b        = 'IzhikevichStrFSINeuron_b'     : population
        c        = 'IzhikevichStrFSINeuron_c'     : population
        d        = 'IzhikevichStrFSINeuron_d'     : population
        n0       = 'IzhikevichStrFSINeuron_n0'    : population
        n1       = 'IzhikevichStrFSINeuron_n1'    : population
        n2       = 'IzhikevichStrFSINeuron_n2'    : population
        I        = 'IzhikevichStrFSINeuron_I'     : population
        tau_ampa = 'general_tauAMPA'              : population
        tau_gaba = 'general_tauGABA'              : population
        E_ampa   = 'general_EAMPA'                : population
        E_gaba   = 'general_EGABA'                : population
        Vr       = 'IzhikevichStrFSINeuron_Vr'    : population
        Vb       = 'IzhikevichStrFSINeuron_Vb'    : population
        Vpeak    = 'IzhikevichStrFSINeuron_Vpeak' : population
        C        = 'IzhikevichStrFSINeuron_C'     : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa / tau_ampa                                                                       : init = 0
        dg_gaba/dt = -g_gaba / tau_gaba                                                                       : init = 0
        dv/dt      = n2 * v * v + n1 * v + n0 - u / C + I / C - g_ampa * (v - E_ampa) - g_gaba * (v - E_gaba) : init = -70.
        du/dt      = if v < Vb:
                         - a * u
                     else:
                         a*(b*(v-Vb)**3-u)                                                                    : init = -18.55
    """, 
    spike = """
        v >= Vpeak
    """,
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich StrFSI Neuron",
    description = "Izhikevich Neuron with conductance based synapses for StrFSI neurons.",
    extra_values = params)

Integrator_neuron = Neuron(        
    parameters = """
        decision  = 'IntegratorNeuron_decision'
        tau       = 'IntegratorNeuron_tau'
        threshold = 'IntegratorNeuron_threshold'
        id        = 'IntegratorNeuron_id'
    """,
    equations = """
        dg_ampa/dt = - g_ampa / tau
    """,
    spike = """
        g_ampa >= threshold        
    """,
    reset = """
        decision = id
    """,
    name = "Integrator Neuron",
    description = "Integrator Neuron, which integrates incoming spikes with value g_ampa and emits a spike when reaching a threshold.",
    extra_values = params)

Poisson_neuronUpDown = Neuron(        
    parameters ="""
        rates   = 'PoissonNeuronUpDown_rates'
        tauUP   = 'PoissonNeuronUpDown_tauUP'   : population
        tauDOWN = 'PoissonNeuronUpDown_tauDOWN' : population
    """,
    equations ="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt
        dact/dt = if (rates - act) > 0:
                      (rates - act) / tauUP
                  else:
                      (rates - act) / tauDOWN        
    """,
    spike ="""    
        p <= act
    """,    
    reset ="""    
        p = 0.0
    """,
    name = "Poisson Neuron",
    description = "Poisson neuron whose rate can be specified and is reached with time constants tauUP and tauDOWN.",
    extra_values = params)
