from gym.envs.registration import register

register(
    id='gym_citycar-v0',
    entry_point='gym_citycar.envs:CityCarEnv',
    kwargs={'path_to_conf_file': 'config/default.json'}
)