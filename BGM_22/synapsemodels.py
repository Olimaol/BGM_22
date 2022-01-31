from ANNarchy import*
from BGmodelSST.sim_params import params

FactorSynapse = Synapse(
    parameters = """
        max_trans  = 'FacSyn_maxTrans' 
        mod_factor = 'FacSyn_modFactor'
    """,    
    equations = "",    
    pre_spike = """
        g_target += w * mod_factor : max = max_trans
    """,
    name = "Factor Synapse",
    description = "Synapse which scales the transmitted value by a specified factor. Factor is equivalent to the connection weight.",
    extra_values = params)
