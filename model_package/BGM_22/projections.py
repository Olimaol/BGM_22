from ANNarchy import*

from BGmodelSST.synapsemodels import FactorSynapse
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, GPe_Arky2, TestThalnoise, GPe_Proto2, EProto1, EProto2, EArky, ED1, ED2, EFSI
from BGmodelSST.sim_params import params

# Apply mod_factors instead of weights. Thus, changes don't cause recompilation.
# CAUTION: Will not be considered in the automatic report!
# CAUTION: appears to slow down spike transmission!

### CortexGo outputs
Cortex_GSTR_D1 = Projection (
    pre = Cortex_G,
    post = STR_D1,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_GSTR_D1.mod_factor = params['weights_cortexGo_StrD1']

Cortex_GSTR_D2 = Projection (
    pre = Cortex_G,
    post = STR_D2,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_GSTR_D2.mod_factor = params['weights_cortexGo_StrD2']

Cortex_GThal = Projection (
    pre = Cortex_G,
   post = Thal,
   target = 'ampa',
   synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_GThal.mod_factor = params['weights_cortexGo_Thal']

Cortex_GSTR_FSI = Projection (
    pre = Cortex_G,
    post = STR_FSI,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'],  weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_GSTR_FSI.mod_factor = params['weights_cortexGo_StrFSI']

### CortexStop outputs
Cortex_SGPe_Arky = Projection (
    pre = Cortex_S,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_SGPe_Arky.mod_factor = params['weights_cortexStop_GPeArky']

Cortex_SGPe_Proto = Projection (
    pre = Cortex_S,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_SGPe_Proto.mod_factor = 0

Cortex_SGPe_Proto2 = Projection (
    pre = Cortex_S,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_SGPe_Proto2.mod_factor = params['weights_cortexStop_GPeCp']

Cortex_SGPe_Arky2 = Projection (
    pre = Cortex_S,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
Cortex_SGPe_Arky2.mod_factor = 0

### CortexPause outputs
Stoppinput1STN = Projection (
    pre = Stoppinput1,
    post = STN,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = 0)
Stoppinput1STN.mod_factor = params['weights_cortexPause_STN']

### StrD1 outputs
STR_D1SNr = Projection (
    pre = STR_D1,
    post = SNr,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D1SNr.mod_factor = params['weights_StrD1_SNr']

STR_D1GPe_Proto2 = Projection (
    pre = STR_D1,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D1GPe_Proto2.mod_factor = params['weights_StrD1_GPeCp']

STR_D1STR_D1 = Projection(
    pre  = STR_D1,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D1STR_D1.mod_factor = params['weights_StrD1_StrD1']

STR_D1STR_D2 = Projection(
    pre  = STR_D1,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D1STR_D2.mod_factor = params['weights_StrD1_StrD2']

### StrD2 outputs
STR_D2GPe_Proto = Projection (
    pre = STR_D2,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2GPe_Proto.mod_factor = params['weights_StrD2_GPeProto']

STR_D2GPe_Arky = Projection (
    pre = STR_D2,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2GPe_Arky.mod_factor = params['weights_StrD2_GPeArky']

STR_D2STR_D1 = Projection(
    pre  = STR_D2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2STR_D1.mod_factor = params['weights_StrD2_StrD1']

STR_D2STR_D2 = Projection(
    pre  = STR_D2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2STR_D2.mod_factor = params['weights_StrD2_StrD2']

STR_D2GPe_Proto2 = Projection (
    pre = STR_D2,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2GPe_Proto2.mod_factor = params['weights_StrD2_GPeCp']

STR_D2GPe_Arky2 = Projection (
    pre = STR_D2,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_D2GPe_Arky2.mod_factor = params['weights_StrD2_GPeArky']*params['GPeArkyCopy_On']

### StrFSI outputs
STR_FSISTR_D1 = Projection(
    pre  = STR_FSI,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_FSISTR_D1.mod_factor = params['weights_StrFSI_StrD1']

STR_FSISTR_D2 = Projection(
    pre  = STR_FSI,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_FSISTR_D2.mod_factor = params['weights_StrFSI_StrD2']

STR_FSISTR_FSI = Projection(
    pre  = STR_FSI,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STR_FSISTR_FSI.mod_factor = params['weights_StrFSI_StrFSI']

### STN outputs
STNSNr = Projection (
    pre = STN,
    post = SNr,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = 10)
STNSNr.mod_factor = params['weights_STN_SNr']

STNGPe_Proto = Projection (
    pre = STN,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STNGPe_Proto.mod_factor = params['weights_STN_GPeProto']

STNGPe_Arky = Projection (
    pre = STN,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STNGPe_Arky.mod_factor = params['weights_STN_GPeArky']

STNGPe_Proto2 = Projection (
    pre = STN,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STNGPe_Proto2.mod_factor = params['weights_STN_GPeCp']

STNGPe_Arky2 = Projection (
    pre = STN,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STNGPe_Arky2.mod_factor = params['weights_STN_GPeArky']*params['GPeArkyCopy_On']

### GPeProto outputs
GPe_ProtoSTN = Projection (
    pre = GPe_Proto,
    post = STN,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoSTN.mod_factor = params['weights_GPeProto_STN']

GPe_ProtoSNr = Projection (
    pre = GPe_Proto,
    post = SNr,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoSNr.mod_factor = params['weights_GPeProto_SNr']

GPe_ProtoGPe_Arky = Projection (
    pre = GPe_Proto,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoGPe_Arky.mod_factor = params['weights_GPeProto_GPeArky']

GPe_ProtoSTR_FSI = Projection (
    pre = GPe_Proto,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoSTR_FSI.mod_factor = params['weights_GPeProto_StrFSI']

GPe_ProtoGPe_Proto2 = Projection (
    pre = GPe_Proto,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoGPe_Proto2.mod_factor = params['weights_GPeProto_GPeCp']

GPe_ProtoGPe_Arky2 = Projection (
    pre = GPe_Proto,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ProtoGPe_Arky2.mod_factor = params['weights_GPeProto_GPeArky']*params['GPeArkyCopy_On']

### GPeArky outputs
GPe_ArkySTR_D1 = Projection (
    pre = GPe_Arky,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ArkySTR_D1.mod_factor = params['weights_GPeArky_StrD1']

GPe_ArkySTR_D2 = Projection (
    pre = GPe_Arky,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays'])) 
GPe_ArkySTR_D2.mod_factor = params['weights_GPeArky_StrD2']

GPe_ArkyGPe_Proto = Projection (
    pre = GPe_Arky,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse  
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ArkyGPe_Proto.mod_factor = params['weights_GPeArky_GPeProto']

GPe_ArkySTR_FSI = Projection (
    pre = GPe_Arky,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ArkySTR_FSI.mod_factor = params['weights_GPeArky_StrFSI']

GPe_ArkyGPe_Proto2 = Projection (
    pre = GPe_Arky,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_ArkyGPe_Proto2.mod_factor = params['weights_GPeArky_GPeCp']

### GPeArkyCopy outputs
GPe_Arky2STR_D1 = Projection (
    pre = GPe_Arky2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Arky2STR_D1.mod_factor = params['weights_GPeArky_StrD1']*params['GPeArkyCopy_On']

GPe_Arky2STR_D2 = Projection (
    pre = GPe_Arky2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays'])) 
GPe_Arky2STR_D2.mod_factor = params['weights_GPeArky_StrD2']*params['GPeArkyCopy_On']

GPe_Arky2STR_FSI = Projection (
    pre = GPe_Arky2,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Arky2STR_FSI.mod_factor = params['weights_GPeArky_StrFSI']*params['GPeArkyCopy_On']

### GPeCp outputs
GPe_Proto2GPe_Proto = Projection (
    pre = GPe_Proto2,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2GPe_Proto.mod_factor = params['weights_GPeCp_GPeProto']

GPe_Proto2STR_D1 = Projection (
    pre = GPe_Proto2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2STR_D1.mod_factor = params['weights_GPeCp_StrD1']

GPe_Proto2STR_D2 = Projection (
    pre = GPe_Proto2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2STR_D2.mod_factor = params['weights_GPeCp_StrD2']

GPe_Proto2STR_FSI = Projection (
    pre = GPe_Proto2,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2STR_FSI.mod_factor = params['weights_GPeCp_StrFSI']

GPe_Proto2GPe_Arky = Projection (
    pre = GPe_Proto2,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2GPe_Arky.mod_factor = params['weights_GPeCp_GPeArky']

GPe_Proto2IntegratorStop = Projection (
    pre = GPe_Proto2,
    post = IntegratorStop,
    target = 'ampa',
    synapse = FactorSynapse
).connect_all_to_all(weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2IntegratorStop.mod_factor = params['weights_GPeCp_IntStop']

GPe_Proto2GPe_Arky2 = Projection (
    pre = GPe_Proto2,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPe_Proto2GPe_Arky2.mod_factor = params['weights_GPeCp_GPeArky']*params['GPeArkyCopy_On']

### SNr outputs
SNrThal = Projection (
    pre = SNr,
    post = Thal,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
SNrThal.mod_factor = params['weights_SNr_Thal']

### Thal outputs
ThalIntegrator = Projection (
    pre = Thal,
    post = Integrator,
    target = 'ampa',
    synapse = FactorSynapse
).connect_all_to_all(weights = 1, delays = Uniform(0.0, params['general_synDelays']))
ThalIntegrator.mod_factor = params['weights_Thal_IntGo']

ThalSD1 = Projection (
    pre = Thal,
    post = STR_D1,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
ThalSD1.mod_factor = params['weights_Thal_StrD1']

ThalSD2 = Projection (
    pre = Thal,
    post = STR_D2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
ThalSD2.mod_factor = params['weights_Thal_StrD2']

ThalFSI = Projection (
    pre = Thal,
    post = STR_FSI,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))
ThalFSI.mod_factor = params['weights_Thal_StrFSI']

### Noise/Baseline inputs
GPeEGPe_Proto = Projection(
    pre  = GPeE,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPeEGPe_Proto.mod_factor = 0

GPEGPe_Arky = Projection(
    pre  = GPeE,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
GPEGPe_Arky.mod_factor = 0

SNrESNr = Projection(
    pre  = SNrE,
    post = SNr,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
SNrESNr.mod_factor = params['weights_SNrE_SNr']

STNESTN = Projection(
    pre  = STNE,
    post = STN,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STNESTN.mod_factor = params['weights_STNE_STN']

STRESTR_D1 = Projection(
    pre  = ED1,
    post = STR_D1,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STRESTR_D1.mod_factor = params['weights_StrD1E_StrD1']

STRESTR_D2 = Projection(
    pre  = ED2,
    post = STR_D2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
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
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
STRESTR_FSI.mod_factor = params['weights_StrFSIE_StrFSI']

EProto1GPe_Proto = Projection(
    pre  = EProto1,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
EProto1GPe_Proto.mod_factor = params['weights_GPeProtoE_GPeProto']

EProto2GPe_Proto2 = Projection(
    pre  = EProto2,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
EProto2GPe_Proto2.mod_factor = params['weights_GPeCpE_GPeCp']

EArkyGPe_Arky = Projection(
    pre  = EArky,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
EArkyGPe_Arky.mod_factor = params['weights_GPeArkyE_GPeArky']

EArkyGPe_Arky2 = Projection(
    pre  = EArky,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
EArkyGPe_Arky2.mod_factor = params['weights_GPeArkyE_GPeArky']*params['GPeArkyCopy_On']










































































