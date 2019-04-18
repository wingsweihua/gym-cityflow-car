This is a gym environment for controlling the vehicles in cityflow.

Please download the repository and run
```bash
pip install -e .
```

Then, in your own code, you can run 
```python
env = gym.make('gym_citycar-v0')
```
This make function has two arguments with default values:
```python
path_to_conf_file=os.path.join("config", "simulator", "default.json")
list_vars_to_subscribe=LIST_VARS
```
where the second argument can be a subset of the following list
```python
[
        # simulation params
        "interval",
        # bulk obs
        "current_time",
        # all vehicle obs

        # all lane obs

        # current vehicle static params
        "max_pos_acc", "max_neg_acc", "max_speed", "min_gap", "headway_time",
        # current vehicle dynamic obs
        "speed", "lane_id", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader",
        # leader vehicle static params
        "leader_max_pos_acc", "leader_max_neg_acc", "leader_max_speed",
        # leader vehicle dynamic obs
        "leader_speed", "dist_to_leader",

    ]
```
