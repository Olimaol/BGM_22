from ANNarchy import get_current_step, get_population
import numpy as np


def add_events(trial_procedure):
    paramsS = trial_procedure.paramsS

    ### define effects for all events (with effects)
    def cor_on(self, pop_name, append=""):
        cor_rates = paramsS[f"{pop_name}.rates{append}"]
        cor_rates_sd = paramsS[f"{pop_name}.rates_sd"]
        cor_randn = self.trial_procedure.rand[pop_name]
        cor_rates_0 = int(paramsS[f"{pop_name}.rates{append}"] > 0)
        get_population(pop_name).rates = (
            cor_rates + cor_rates_sd * cor_randn * cor_rates_0
        )

    def cor_off(self, pop_name):
        get_population(pop_name).rates = 0

    def trial_init(self):
        get_population("cor_go").rates = 0
        get_population("cor_stop").rates = 0
        get_population("cor_pause").rates = 0

    def integrator_reset(self):
        get_population("integrator_go").decision = 0
        get_population("integrator_go").g_ampa = 0
        get_population("integrator_stop").decision = 0

    ### add all events
    trial_procedure.add_event(
        name="trial_init",
        onset=get_current_step(),
        trigger={"integrator_reset": int(paramsS["t.init"] / paramsS["timestep"])},
        effect=trial_init,
    )

    trial_procedure.add_event(
        name="integrator_reset", trigger={"go_cue": 0}, effect=integrator_reset
    )

    trial_procedure.add_event(
        name="go_cue",
        trigger={
            "stop_cue": int(paramsS["t.ssd"] / paramsS["timestep"]),
            "cor_pause_on_go": 0,
            "cor_go_on": int(
                np.clip(
                    trial_procedure.rng.normal(
                        paramsS["t.cor_go__delay"], paramsS["t.cor_go__delay_sd"]
                    ),
                    0,
                    None,
                )
                / paramsS["timestep"]
            ),
            "end": int(
                (
                    paramsS["t.ssd"]
                    + paramsS["t.cor_stop__delay_cue"]
                    + paramsS["t.cor_stop__dur_cue"]
                    + paramsS["t.decay"]
                )
                / paramsS["timestep"]
            ),
        },
    )

    trial_procedure.add_event(
        name="stop_cue",
        requirement_string="mode==stop",
        trigger={
            "cor_pause_on_stop": 0,
            "cor_stop_on_cue": int(
                paramsS["t.cor_stop__delay_cue"] / paramsS["timestep"]
            ),
        },
    )

    trial_procedure.add_event(
        name="cor_go_on", effect=lambda self: cor_on(self, "cor_go")
    )

    trial_procedure.add_event(
        name="cor_go_off", effect=lambda self: cor_off(self, "cor_go")
    )

    trial_procedure.add_event(
        name="cor_pause_on_go",
        effect=lambda self: cor_on(self, "cor_pause", "_go"),
        trigger={
            "cor_pause_off": int(paramsS["t.cor_pause__dur"] / paramsS["timestep"])
        },
    )

    trial_procedure.add_event(
        name="cor_pause_on_stop",
        effect=lambda self: cor_on(self, "cor_pause", "_stop"),
        trigger={
            "cor_pause_off": int(paramsS["t.cor_pause__dur"] / paramsS["timestep"])
        },
    )

    trial_procedure.add_event(
        name="cor_pause_off", effect=lambda self: cor_off(self, "cor_pause")
    )

    trial_procedure.add_event(
        name="cor_stop_on_cue",
        effect=lambda self: cor_on(self, "cor_stop", "_cue"),
        trigger={
            "cor_stop_off": int(paramsS["t.cor_stop__dur_cue"] / paramsS["timestep"])
        },
    )

    trial_procedure.add_event(
        name="cor_stop_on_respose",
        effect=lambda self: cor_on(self, "cor_stop", "_response"),
        trigger={
            "cor_stop_off": int(
                paramsS["t.cor_stop__dur_response"] / paramsS["timestep"]
            )
        },
    )

    trial_procedure.add_event(
        name="cor_stop_off",
        effect=lambda self: cor_off(self, "cor_stop"),
        trigger={"end": int(paramsS["t.decay"] / paramsS["timestep"])},
    )

    trial_procedure.add_event(
        name="end", effect=lambda self: self.trial_procedure.end_sim()
    )

    trial_procedure.add_event(
        name="motor_response",
        model_trigger="integrator_go",
        trigger={
            "cor_stop_on_respose": int(
                paramsS["t.cor_stop__delay_response"] / paramsS["timestep"]
            )
        },
    )

    trial_procedure.add_event(
        name="gpe_cp_resp",
        model_trigger="integrator_stop",
        requirement_string="happened_event_list==[cor_stop_on_cue] or happened_event_list==[cor_stop_on_respose]",
        trigger={"cor_go_off": 0},
    )
