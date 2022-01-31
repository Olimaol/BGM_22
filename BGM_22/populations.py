from ANNarchy import*
import numpy as np

from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron, STR_FSI_neuron, Integrator_neuron, Poisson_neuronUpDown
from BGmodelSST.sim_params import params

### cortex / input populations
#cortex-Go
Cortex_G         = Population(params['general_populationSize'], Poisson_neuronUpDown, name="Cortex-Go")
Cortex_G.tauUP   = params['cortexGo_tauUP']
Cortex_G.tauDOWN = params['cortexGo_tauDOWN']
#cortex-Pause
Stoppinput1         = Population(params['general_populationSize'], Poisson_neuronUpDown, name="Cortex_Pause")
Stoppinput1.tauUP   = params['cortexPause_tauUP']
Stoppinput1.tauDOWN = params['cortexPause_tauDOWN']
#cortex-Stop
Cortex_S         = Population(params['general_populationSize'], Poisson_neuronUpDown, name="Cortex_Stop")
Cortex_S.tauUP   = params['cortexStop_tauUP']
Cortex_S.tauDOWN = params['cortexStop_tauDOWN']
### Str Populations
#StrD1
STR_D1 = Population(params['general_populationSize'], Izhikevich_STR_neuron, name="STR_D1")
#StrD2
STR_D2 = Population(params['general_populationSize'], Izhikevich_STR_neuron, name="STR_D2")
#StrFSI
STR_FSI = Population(params['general_populationSize'], STR_FSI_neuron, name="FSI")
### BG Populations
#STN
STN   = Population(params['general_populationSize'], Izhikevich_neuron, name="STN")
STN.a = params['STN_a']
STN.b = params['STN_b']
STN.c = params['STN_c']
STN.d = params['STN_d']
STN.I = params['STN_I']
#SNr
SNr   = Population(params['general_populationSize'], Izhikevich_neuron, name="SNr")
SNr.a = params['SNr_a']
SNr.b = params['SNr_b']
SNr.c = params['SNr_c']
SNr.d = params['SNr_d']
SNr.I = params['SNr_I']
#GPe-Proto (optimized fit of Abdi,Mallet et al.)
GPe_Proto            = Population(params['general_populationSize'], Izhikevich_neuron, name="GPe Proto")
GPe_Proto.a          = params['GPeProto_a']
GPe_Proto.b          = params['GPeProto_b']
GPe_Proto.c          = params['GPeProto_c']
GPe_Proto.d          = params['GPeProto_d']
GPe_Proto.n0         = params['GPeProto_n0']
GPe_Proto.n1         = params['GPeProto_n1']
GPe_Proto.n2         = params['GPeProto_n2']
GPe_Proto.refractory = params['GPeProto_refractory']
GPe_Proto.I          = params['GPeProto_I']
#GPe-Arky (optimized fit of Abdi,Mallet et al.)
GPe_Arky            = Population(params['general_populationSize'], Izhikevich_neuron, name="GPe Arky")
GPe_Arky.a          = params['GPeArky_a']
GPe_Arky.b          = params['GPeArky_b']
GPe_Arky.c          = params['GPeArky_c']
GPe_Arky.d          = params['GPeArky_d']
GPe_Arky.n0         = params['GPeArky_n0']
GPe_Arky.n1         = params['GPeArky_n1']
GPe_Arky.n2         = params['GPeArky_n2']
GPe_Arky.refractory = params['GPeArky_refractory']
GPe_Arky.I          = params['GPeArky_I']
#GPe-Cp
GPe_Proto2            = Population(params['general_populationSize'], Izhikevich_neuron, name="GPe Proto2")
GPe_Proto2.a          = params['GPeCp_a']
GPe_Proto2.b          = params['GPeCp_b']
GPe_Proto2.c          = params['GPeCp_c']
GPe_Proto2.d          = params['GPeCp_d']
GPe_Proto2.n0         = params['GPeCp_n0']
GPe_Proto2.n1         = params['GPeCp_n1']
GPe_Proto2.n2         = params['GPeCp_n2']
GPe_Proto2.refractory = params['GPeCp_refractory']
GPe_Proto2.I          = params['GPeCp_I']
#GPe-ArkyCopy
GPe_Arky2            = Population(params['general_populationSize'], Izhikevich_neuron, name="GPe Arky2")
GPe_Arky2.a          = params['GPeArky_a']
GPe_Arky2.b          = params['GPeArky_b']
GPe_Arky2.c          = params['GPeArky_c']
GPe_Arky2.d          = params['GPeArky_d']
GPe_Arky2.n0         = params['GPeArky_n0']
GPe_Arky2.n1         = params['GPeArky_n1']
GPe_Arky2.n2         = params['GPeArky_n2']
GPe_Arky2.refractory = params['GPeArky_refractory']
GPe_Arky2.I          = params['GPeArky_I']
#thalamus
Thal   = Population(params['general_populationSize'], Izhikevich_neuron, name="Thal")
Thal.a = params['Thal_a']
Thal.b = params['Thal_b']
Thal.c = params['Thal_c']
Thal.d = params['Thal_d']
Thal.I = params['Thal_I']
### Integrator Neurons
#IntegratorGo
Integrator     = Population(1, Integrator_neuron, stop_condition =  "decision == -1", name="Integrator")
Integrator.tau = params['IntGo_tau']
#IntegratorStop
IntegratorStop     = Population(1, Integrator_neuron, stop_condition =  "decision == -1", name="IntegratorStop")
IntegratorStop.tau = params['IntStop_tau']
### Noise / Baseline Input populations
#GPe noise
GPeE         = Population(params['general_populationSize'], Poisson_neuronUpDown, name="GPeE")
GPeE.rates   = 0
GPeE.act     = 0
#GPe-Proto noise
EProto1       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="EProto1")
Proto1Noise   = np.random.normal(params['GPeE_rates'],params['GPeE_sd'],params['general_populationSize'])
EProto1.rates = Proto1Noise
EProto1.act   = Proto1Noise
#GPe-Cp noise
EProto2       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="EProto2")
Proto2Noise   = np.random.normal(params['GPeE_rates'],params['GPeE_sd'],params['general_populationSize'])
EProto2.rates = Proto2Noise
EProto2.act   = Proto2Noise
#GPe-Arky noise
EArky       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="EArky")
ArkyNoise   = np.random.normal(params['GPeE_rates'],params['GPeE_sd'],params['general_populationSize'])
EArky.rates = ArkyNoise
EArky.act   = ArkyNoise
#SNr noise
SNrE       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="SNrE")
SNrNoise   = np.random.normal(params['SNrE_rates'],params['SNrE_sd'],params['general_populationSize'])
SNrE.rates = SNrNoise
SNrE.act   = SNrNoise
#STN noise
STNE       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="STNE")
STNNoise   = np.random.normal(params['STNE_rates'],params['STNE_sd'],params['general_populationSize'])
STNE.rates = STNNoise
STNE.act   = STNNoise
#Str noise
STRE       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="STRE")
STRE.rates = 0
STRE.act   = 0
#StrD1 noise
ED1       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="ED1")
StrNoise  = np.random.normal(params['StrE_rates'],params['StrE_sd'],params['general_populationSize'])
ED1.rates = StrNoise
ED1.act   = StrNoise
#StrD2 noise
ED2       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="ED2")
StrNoise  = np.random.normal(params['StrE_rates'],params['StrE_sd'],params['general_populationSize'])
ED2.rates = StrNoise
ED2.act   = StrNoise
#StrFSI noise
EFSI       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="EFSI")
StrNoise   = np.random.normal(params['StrE_rates'],params['StrE_sd'],params['general_populationSize'])
EFSI.rates = StrNoise
EFSI.act   = StrNoise
#thalamus noise
TestThalnoise       = Population(params['general_populationSize'], Poisson_neuronUpDown, name="ThalNoise")
randThal            = np.random.normal(params['ThalE_rates'],params['ThalE_sd'],params['general_populationSize'])
TestThalnoise.rates = randThal
TestThalnoise.act   = randThal
