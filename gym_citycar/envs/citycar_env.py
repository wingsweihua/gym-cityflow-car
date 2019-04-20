import gym
import cityflow as engine
from gym import error, spaces, utils
from gym.utils import seeding
import json
import os
import sys
import pandas as pd
import numpy as np
import math

class CityCarEnv(gym.Env):

    LIST_VARS = [
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

    dic_feature_range = {
        "interval": (0.1, 1),
        "max_pos_acc": (0, 10),
        "max_neg_acc": (0, 10),
        "max_speed": (0, 20),
        "min_gap": (0, 10),
        "headway_time": (0, 10),
        "speed": (0, 20),
        "pos_in_lane": (0, 1000),
        "lane_max_speed": (0, 20),
        "if_exit_lane": (0, 1),
        "dist_to_signal": (0, 1000),
        "phase": (0, 1),
        "if_leader": (0, 1),
        "leader_max_pos_acc": (0, 10),
        "leader_max_neg_acc": (0, 10),
        "leader_max_speed": (0, 20),
        "leader_speed": (0, 20),
        "dist_to_leader": (0, 1000),
    }

    action_range = (0, 20)

    def __init__(self, **kwargs):
        self.path_to_conf_file = kwargs["path_to_conf_file"]
        self.list_vars_to_subscribe = kwargs["list_vars_to_subscribe"]
        print("path to conf files", self.path_to_conf_file)
        self.eng = None
        self.dic_static_sim_params = {}
        self.signal_plan = None

        self.action_space = spaces.Box(low=np.array([self.action_range[0]]), high=np.array([self.action_range[1]]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.dic_feature_range[_][0] for _ in self.list_vars_to_subscribe]),
            high=np.array([self.dic_feature_range[_][1] for _ in self.list_vars_to_subscribe]),
            dtype=np.float32)
        self.observation_header = self.list_vars_to_subscribe
        self.action_header = ["action"]
        self.reset()



    def step(self, n_action, n_info):

        current_time = self.eng.get_current_time()
        if current_time % 100 == 0:
            print(current_time)

        # take action
        self._set_vehicle_speed(n_action, n_info)

        # run one step
        self.eng.next_step()

        # observations for next step
        self._set_signal()
        n_obs, n_reward, n_done, n_info = self._get_description()
        # n_info: vec_id, next_speed_est, priority, current_time, lane_id

        return n_obs, n_reward, n_done, n_info # todo - "done" may need to be changed

    def _set_vehicle_speed(self, n_action, n_info):

        # todo - no protection speed yet, decide where to add  (speed can not exceed the maximum speed allowed by the situation)

        for ind in range(len(n_action)):
            self.eng.set_vehicle_speed(n_info["priority"][ind], n_action[ind][0])


    def reset(self):
        self.eng = engine.Engine(self.path_to_conf_file)
        self._extract_sim_params()
        self._load_signal_plan()

        self._set_signal()
        n_obs, n_reward, n_done, n_info = self._get_description()

        return n_obs, n_info

    def render(self, mode='human', close=False):
        pass

    def _get_current_phase(self, inter_id):
        current_time = self.eng.get_current_time()
        if current_time == len(self.signal_plan):
            return int(self.signal_plan[inter_id][int(current_time-1)])
        else:
            return int(self.signal_plan[inter_id][int(current_time)])

    def _load_signal_plan(self):

        dic_conf = json.load(open(self.path_to_conf_file, "r"))
        path_to_signal_plan_file = os.path.join(dic_conf["dir"], dic_conf["signalPlanFile"])
        print(path_to_signal_plan_file)
        print(os.getcwd())
        self.signal_plan = pd.read_csv(path_to_signal_plan_file, sep="\t", header=0, dtype=int)

    def _set_signal(self):
        for inter_id in self.signal_plan.columns:
            self.eng.set_tl_phase(inter_id,
                                  self._get_current_phase(inter_id))

    def _cal_facing_inter(self, l_id):

        lane_id_split = l_id.split('_')

        x = int(lane_id_split[1])
        y = int(lane_id_split[2])
        d = int(lane_id_split[3])

        if d == 0:
            facing_inter = "intersection_{0}_{1}".format(x + 1, y)
        elif d == 1:
            facing_inter = "intersection_{0}_{1}".format(x, y + 1)
        elif d == 2:
            facing_inter = "intersection_{0}_{1}".format(x - 1, y)
        elif d == 3:
            facing_inter = "intersection_{0}_{1}".format(x, y - 1)
        else:
            sys.exit()

        return facing_inter

    def _extract_sim_params(self):

        dic_conf = json.load(open(self.path_to_conf_file, "r"))
        self.dic_static_sim_params["interval"] = dic_conf["interval"]

        self.dic_static_sim_params["intersection_params"] = {}
        self.dic_static_sim_params["lane_params"] = {}
        path_to_roadnet_file = os.path.join(dic_conf["dir"], dic_conf["roadnetFile"])
        dic_roadnet = json.load(open(path_to_roadnet_file, "r"))
        # get inter params

        for inter in dic_roadnet["intersections"]:
            inter_id = inter["id"]
            dic_inter_params = dict()
            dic_inter_params["virtual"] = inter["virtual"]
            dic_inter_params["phase_allowed_lanes"] = {}
            road_links = inter["roadLinks"]
            for ind_phase, signal_phase in enumerate(inter["trafficLight"]["lightphases"]):
                this_phase_allowed_lanes = []
                for link in signal_phase["availableRoadLinks"]:
                    # add one link
                    rl = road_links[link]
                    this_phase_allowed_lanes += ["{0}_{1}".format(rl["startRoad"], ll["startLaneIndex"]) for ll in rl["laneLinks"]]
                this_phase_allowed_lanes = np.unique(this_phase_allowed_lanes).tolist()
                dic_inter_params["phase_allowed_lanes"][ind_phase] = this_phase_allowed_lanes.copy()
            self.dic_static_sim_params["intersection_params"][inter_id] = dic_inter_params.copy()

        # get lane params
        for road in dic_roadnet["roads"]:
            road_coor = road["points"]
            p1 = road_coor[0]
            p2 = road_coor[-1]
            road_length = math.sqrt(
                (p1['x'] - p2['x']) * (p1['x'] - p2['x']) + (p1['y'] - p2['y']) * (p1['y'] - p2['y'])
            )
            for ind_lane, lane in enumerate(road["lanes"]):
                # determine the property of one lane
                lane_id = "{0}_{1}".format(road["id"], ind_lane)
                dic_lane_params = dict()
                dic_lane_params["max_speed"] = lane["maxSpeed"]
                dic_lane_params["facing_intersection"] = self._cal_facing_inter(lane_id)
                dic_lane_params["if_exit_lane"] = 1 if self.dic_static_sim_params["intersection_params"][dic_lane_params["facing_intersection"]]["virtual"] else 0
                dic_lane_params["length"] = road_length
                self.dic_static_sim_params["lane_params"][lane_id] = dic_lane_params.copy()

        self.dic_static_sim_params["flow_params"] = {}
        path_to_flow_file = os.path.join(dic_conf["dir"], dic_conf["flowFile"])
        list_flow = json.load(open(path_to_flow_file, "r"))
        for ind_flow, flow in enumerate(list_flow):
            flow_id = "flow_{}".format(ind_flow)
            dic_flow_params = dict()
            dic_flow_params["max_pos_acc"] = flow["vehicle"]["maxPosAcc"]
            dic_flow_params["max_neg_acc"] = flow["vehicle"]["maxNegAcc"]
            dic_flow_params["min_gap"] = flow["vehicle"]["minGap"]
            dic_flow_params["max_speed"] = flow["vehicle"]["maxSpeed"]
            dic_flow_params["headway_time"] = flow["vehicle"]["headwayTime"]
            self.dic_static_sim_params["flow_params"][flow_id] = dic_flow_params.copy()

    def _get_vec_flow(self, vec_id):  # todo - need to check

        vec_id_split = vec_id.split('_')

        return "flow_{}".format(vec_id_split[1])

    def _get_one_hot_phase_for_lane(self, l_id):

        facing_inter = self.dic_static_sim_params["lane_params"][l_id]["facing_intersection"]
        allowed_lane = self.dic_static_sim_params["intersection_params"][facing_inter]["phase_allowed_lanes"][
            self._get_current_phase(facing_inter)]
        phase_in_bit = 1 if l_id in allowed_lane else 0

        return phase_in_bit

    def _get_description(self):

        list_obs = []
        list_reward = []
        list_done = []
        dic_info = {"vec_id": list(), "next_speed_est": list(), "priority": list(), "current_time": list(), "lane_id": list()}

        current_time = self.eng.get_current_time()  # return a double, time past in seconds
        lane_vehicles = self.eng.get_lane_vehicles()  # return a dict, {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        vehicle_speed = self.eng.get_vehicle_speed()  # return a dict, {vehicle_id: vehicle_speed, ...}
        vehicle_distance = self.eng.get_vehicle_distance()  # return a dict, {vehicle_id: vehicle_distance, ...}
        distance_to_leader = self.eng.get_vehicle_gap()
        vehicle_leaders = self.eng.get_vehicle_leader()

        vehicle_next_speed = self.eng.get_vehicle_next_speed()
        vehicle_priority = self.eng.get_vehicle_priority()

        for lane_id, lane_vec in lane_vehicles.items():
            for vec_id in lane_vec:

                dic_vec = {}

                leader = vehicle_leaders[vec_id]
                dist_tl = distance_to_leader[vec_id]

                if (leader != '' and dist_tl == -1) or (leader == '' and dist_tl != -1):
                    print("wrong vehicle")
                    print(leader, dist_tl, vec_id)
                    leader = ''
                    dist_tl = -1
                    # sys.exit()

                # =================== current vehicle informations =====================

                dic_vec["vec_id"] = vec_id

                # simulation params
                dic_vec["interval"] = self.dic_static_sim_params["interval"]
                # bulk obs
                dic_vec["current_time"] = current_time

                # current vehicle static params
                dic_vec["max_pos_acc"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(vec_id)]["max_pos_acc"]
                dic_vec["max_neg_acc"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(vec_id)]["max_neg_acc"]
                dic_vec["max_speed"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(vec_id)]["max_speed"]
                dic_vec["min_gap"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(vec_id)]["min_gap"]
                dic_vec["headway_time"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(vec_id)]["headway_time"]
                # current vehicle dynamic obs
                dic_vec["speed"] = vehicle_speed[vec_id]
                dic_vec["lane_id"] = lane_id
                dic_vec["pos_in_lane"] = vehicle_distance[vec_id]
                dic_vec["lane_max_speed"] = self.dic_static_sim_params["lane_params"][lane_id]["max_speed"]
                dic_vec["if_exit_lane"] = self.dic_static_sim_params["lane_params"][lane_id]["if_exit_lane"]
                dic_vec["dist_to_signal"] = self.dic_static_sim_params["lane_params"][lane_id]["length"] - dic_vec["pos_in_lane"]
                dic_vec["phase"] = 1 if dic_vec["if_exit_lane"] else self._get_one_hot_phase_for_lane(lane_id)
                dic_vec["if_leader"] = 1 if leader != '' else 0

                # =================== current vehicle informations =====================

                # =================== leader vehicle informations =====================

                # leader vehicle static params todo - pay attention to the default values
                dic_vec["leader_max_pos_acc"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(leader)]["max_pos_acc"] if leader != '' else 100
                dic_vec["leader_max_neg_acc"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(leader)]["max_neg_acc"] if leader != '' else 100
                dic_vec["leader_max_speed"] = self.dic_static_sim_params["flow_params"][self._get_vec_flow(leader)]["max_speed"] if leader != '' else 100
                # leader vehicle dynamic obs
                dic_vec["leader_speed"] = vehicle_speed[leader] if leader != '' else 100
                dic_vec["dist_to_leader"] = dist_tl if leader != '' else 100

                # =================== leader vehicle informations =====================

                dic_vec["next_speed_est"] = max(vehicle_next_speed[vec_id], 0)
                dic_vec["priority"] = vehicle_priority[vec_id]

                obs = np.array([dic_vec[_] for _ in self.list_vars_to_subscribe])
                r = 0
                done = False
                dic_info["vec_id"].append(dic_vec["vec_id"])
                dic_info["next_speed_est"].append(dic_vec["next_speed_est"])
                dic_info["priority"].append(dic_vec["priority"])
                dic_info["current_time"].append(dic_vec["current_time"])
                dic_info["lane_id"].append(dic_vec["lane_id"])

                list_obs.append(obs)
                list_reward.append(r)
                list_done.append(done)

        return list_obs, list_reward, list_done, dic_info



if __name__ == "__main__":

    env = CityCarEnv(path_to_conf_file=os.path.join("config", "simulator", "default.json"),
                     list_vars_to_subscribe=["interval",
                 "max_pos_acc", "max_neg_acc", "max_speed", "min_gap", "headway_time",
                 "speed", "pos_in_lane", "lane_max_speed", "if_exit_lane", "dist_to_signal", "phase", "if_leader",
                 "leader_max_pos_acc", "leader_max_neg_acc", "leader_max_speed",
                 "leader_speed", "dist_to_leader",])
    observation, info = env.reset()
    for i in range(500):
        action = [np.array([a]) for a in info["next_speed_est"]]
        observation, reward, done, info = env.step(action,
                                                   info)
