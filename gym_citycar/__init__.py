from gym.envs.registration import register
import os

register(
    id='gym_citycar-v0',
    entry_point='gym_citycar.envs:CityCarEnv',
    kwargs={'path_to_conf_file': os.path.join("config", "simulator", "default.json"),
            "list_vars_to_subscribe":
                ["interval",
                 "max_pos_acc", "max_neg_acc", "max_speed", "min_gap", "headway_time",
                 "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader",
                 "leader_max_pos_acc", "leader_max_neg_acc", "leader_max_speed",
                 "leader_speed", "dist_to_leader",]}
)
