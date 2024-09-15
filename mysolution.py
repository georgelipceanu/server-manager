
import json
import numpy as np
import pandas as pd
from scipy.stats import truncweibull_min

def save_solution(solution, path):
    def save_json(path, data):
        with open(path, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=4)
    # Saves a solution into a json file.
    if isinstance(solution, pd.DataFrame):
        solution = solution.to_dict('records')
    return save_json(path, solution)

def get_actual_demand(demand):
    def get_known(key):
        # STORE SOME CONFIGURATION VARIABLES
        if key == 'datacenter_id':
            return ['DC1', 
                    'DC2', 
                    'DC3', 
                    'DC4']
        elif key == 'actions':
            return ['buy',
                    'dismiss']
        elif key == 'server_generation':
            return ['CPU.S1', 
                    'CPU.S2', 
                    'CPU.S3', 
                    'CPU.S4', 
                    'GPU.S1', 
                    'GPU.S2', 
                    'GPU.S3']
        elif key == 'latency_sensitivity':
            return ['high', 
                    'medium', 
                    'low']
        elif key == 'required_columns':
            return ['time_step', 
                    'datacenter_id', 
                    'server_generation', 
                    'server_id',
                    'action']
        elif key == 'time_steps':
            return 168
        elif key == 'datacenter_fields':
            return ['datacenter_id', 
                    'cost_of_energy',
                    'latency_sensitivity', 
                    'slots_capacity']
            
    def get_random_walk(n, mu, sigma):
        # HELPER FUNCTION TO GET A RANDOM WALK TO CHANGE THE DEMAND PATTERN
        r = np.random.normal(mu, sigma, n)
        ts = np.empty(n)
        ts[0] = r[0]
        for i in range(1, n):
            ts[i] = ts[i - 1] + r[i]
        ts = (2 * (ts - ts.min()) / np.ptp(ts)) - 1
        return ts
    
    # CALCULATE THE ACTUAL DEMAND AT TIME-STEP t
    actual_demand = []
    for ls in get_known('latency_sensitivity'):
        for sg in get_known('server_generation'):
            d = demand[demand['latency_sensitivity'] == ls]
            sg_demand = d[sg].values.astype(float)
            rw = get_random_walk(sg_demand.shape[0], 0, 2)
            sg_demand += (rw * sg_demand)

            ls_sg_demand = pd.DataFrame()
            ls_sg_demand['time_step'] = d['time_step']
            ls_sg_demand['server_generation'] = sg
            ls_sg_demand['latency_sensitivity'] = ls
            ls_sg_demand['demand'] = sg_demand.astype(int)
            actual_demand.append(ls_sg_demand)

    actual_demand = pd.concat(actual_demand, axis=0, ignore_index=True)
    actual_demand = actual_demand.pivot(index=['time_step', 'server_generation'], columns='latency_sensitivity')
    actual_demand.columns = actual_demand.columns.droplevel(0)
    actual_demand = actual_demand.loc[actual_demand[get_known('latency_sensitivity')].sum(axis=1) > 0]
    actual_demand = actual_demand.reset_index(['time_step', 'server_generation'], col_level=1, inplace=False)
    return actual_demand


def get_my_solution(d):
    def get_time_step_demand(demand, ts):
        # GET THE DEMAND AT A SPECIFIC TIME-STEP t
        d = demand[demand['time_step'] == ts]
        d = d.set_index('server_generation', drop=True, inplace=False)
        d = d.drop(columns='time_step', inplace=False)
        return d

    def find_action_by_id(server_action_map, server_id):
        if server_id in server_action_map and 'buy' in server_action_map[server_id]:
            return server_action_map[server_id]['buy']['time_step']
        return None
    
    def calculate_servers_needed(demand, failure_rate, server_capacity):
        effective_capacity_per_server = server_capacity * (1 - failure_rate)
        servers_needed = np.ceil(demand / effective_capacity_per_server)
        return int(servers_needed)
    
    def sample_failure_rate_truncated_weibull(shape, lower_bound=0.05, upper_bound=0.1, size=1):
        scale = upper_bound

        a = lower_bound / scale
        b = upper_bound / scale
    
        truncated_samples = truncweibull_min.rvs(shape, a=a, b=b, scale=scale, size=size)
    
        return truncated_samples if size > 1 else truncated_samples[0]
    
    solution = []
    server_action_map = {} ## FOR QUICK ACCESS
    
    servers = pd.read_csv('./data/servers.csv')
    server_gens = servers.server_generation
    server_gens_map = {name: idx for idx, name in enumerate(server_gens)}
    
    capspace = pd.read_csv('./data/datacenters.csv')
    dc1_cap = capspace['slots_capacity'].iloc[0]
    dc2_cap = capspace['slots_capacity'].iloc[1]
    dc3_cap = capspace['slots_capacity'].iloc[2]
    dc4_cap = capspace['slots_capacity'].iloc[3]

    datacenter_slots = pd.read_csv('./data/capacity.csv') ## EMPTY 7*4 DATAFRAME, SERVER GENS=ROWS DATACENTERS=COLUMNS
    id_map = pd.read_csv('./data/capacity.csv')
    remove_id_map = pd.read_csv('./data/capacity.csv') ## KEEPING TRACK OF WHAT HAS BEEN REMOVED
    servers_needed_previous_timestep_map = pd.read_csv('./data/capacity.csv')
    
    ## Rule-Based Heuristic APPROACH
    
    for time_step in range(1,169):
        if (time_step >= 96): ## LIFESPAN OF ALL SERVERS
            for server_id, actions in server_action_map.items():
                if server_action_map[server_id]['dismiss'] == False:
                    buy_time = actions['buy']['time_step']
                    if time_step - buy_time >= 95:
                        datacenter_id = actions['buy']['datacenter_id']
                        server_gen = actions['buy']['server_generation']
                        server_id_ = actions['buy']['server_id']
                        slot_size = 2 if "CPU" in server_id_ else 4 if "GPU" in server_id_ else 1

                        new_action = {
                            "time_step": time_step,
                            "datacenter_id": datacenter_id,
                            "server_generation": server_gen,
                            "server_id": server_id,
                            "action": "dismiss"
                        }
                        solution.append(new_action)
                        server_action_map[server_id]['dismiss'] = True
                        server_action_map[server_id]['dismissed'] = True  # MARK AS DISMISSED

                        # UPDATE SLOT USAGE IN DC
                        if datacenter_id == "DC1":
                            datacenter_slots.at[server_gens_map[server_gen], 'DC1'] -= slot_size
                            remove_id_map['DC1'].iloc[server_gens_map[server_gen]] += 1
                        elif datacenter_id == "DC2":
                            datacenter_slots.at[server_gens_map[server_gen], 'DC2'] -= slot_size
                            remove_id_map['DC2'].iloc[server_gens_map[server_gen]] += 1
                        elif datacenter_id == "DC3":
                            datacenter_slots.at[server_gens_map[server_gen], 'DC3'] -= slot_size
                            remove_id_map['DC3'].iloc[server_gens_map[server_gen]] += 1
                        elif datacenter_id == "DC4":
                            datacenter_slots.at[server_gens_map[server_gen], 'DC4'] -= slot_size
                            remove_id_map['DC4'].iloc[server_gens_map[server_gen]] += 1
        
        current_demand = get_time_step_demand(d, time_step)
        for row in range(0,current_demand.shape[0]):
            server_gen = current_demand.index[row]
            slot_size = servers.iloc[server_gens_map[server_gen],4]
            release_time = servers.iloc[server_gens_map[server_gen],2]
            capacity = servers.iloc[server_gens_map[server_gen],6]
            start_time, end_time = map(int, release_time.strip("[]").split(","))
            
            for column in range(0,current_demand.shape[1]):
                ## COLUMNS: 0=high(DC3,4), 1=low(DC1), 2=medium(DC1)
                demand_for_server = current_demand.iloc[row,column]
                servers_needed = round(calculate_servers_needed(demand_for_server,sample_failure_rate_truncated_weibull(1.5),capacity), -1)
                
                if servers_needed > servers_needed_previous_timestep_map.iloc[row,column+1]:            
                    ## BUYING 
                    for j in range(servers_needed_previous_timestep_map.iloc[row,column+1], servers_needed):
                        
                        if (column==1 and dc1_cap-slot_size>=datacenter_slots['DC1'].sum() and start_time <= time_step < end_time and 168 - time_step > 5):
                            
                            datacenter_slots.at[server_gens_map[server_gen],'DC1'] += slot_size
                            new_action = {
                                "time_step": time_step,
                                "datacenter_id": "DC1",
                                "server_generation": server_gen,
                                "server_id": "DC1-" + server_gen + "-" + str(id_map['DC1'].iloc[server_gens_map[server_gen]]),
                                "action": "buy"
                            }
                            server_action_map["DC1-" + server_gen + "-" + str(id_map['DC1'].iloc[server_gens_map[server_gen]])] = {
                                'buy': new_action,
                                'dismiss': False}
                            id_map['DC1'].iloc[server_gens_map[server_gen]] += 1
                            solution.append(new_action)
                        
                        elif (column==2 and dc2_cap-slot_size>=datacenter_slots['DC2'].sum() and start_time <= time_step < end_time and 168 - time_step > 5):
                            datacenter_slots.at[server_gens_map[server_gen],'DC2'] += slot_size
                            new_action = {
                                "time_step": time_step,
                                "datacenter_id": "DC2",
                                "server_generation": server_gen,
                                "server_id": "DC2-" + server_gen + "-" + str(id_map['DC2'].iloc[server_gens_map[server_gen]]),
                                "action": "buy"
                            }
                            server_action_map["DC2-" + server_gen + "-" + str(id_map['DC2'].iloc[server_gens_map[server_gen]])] = {
                                'buy': new_action,
                                'dismiss': False}
                            id_map['DC2'].iloc[server_gens_map[server_gen]] += 1
                            solution.append(new_action)

                        elif (column==0 and start_time <= time_step < end_time and 168 - time_step > 5):
                            if (dc3_cap-slot_size>=datacenter_slots['DC3'].sum()):
                                datacenter_slots.at[server_gens_map[server_gen],'DC3'] += slot_size
                                new_action = {
                                    "time_step": time_step,
                                    "datacenter_id": "DC3",
                                    "server_generation": server_gen,
                                    "server_id": "DC3-" + server_gen + "-" + str(id_map['DC3'].iloc[server_gens_map[server_gen]]),
                                    "action": "buy"
                                }
                                server_action_map["DC3-" + server_gen + "-" + str(id_map['DC3'].iloc[server_gens_map[server_gen]])] = {
                                    'buy': new_action,
                                    'dismiss': False}
                                id_map['DC3'].iloc[server_gens_map[server_gen]] += 1
                                solution.append(new_action)
                            elif (dc4_cap-slot_size>=datacenter_slots['DC4'].sum()):
                                datacenter_slots.at[server_gens_map[server_gen],'DC4'] += slot_size
                                new_action = {
                                    "time_step": time_step,
                                    "datacenter_id": "DC4",
                                    "server_generation": server_gen,
                                    "server_id": "DC4-" + server_gen + "-" + str(id_map['DC4'].iloc[server_gens_map[server_gen]]),
                                    "action": "buy"
                                }
                                server_action_map["DC4-" + server_gen + "-" + str(id_map['DC4'].iloc[server_gens_map[server_gen]])] = {
                                    'buy': new_action,
                                    'dismiss': False}
                                id_map['DC4'].iloc[server_gens_map[server_gen]] += 1
                                solution.append(new_action)
                    
                elif(servers_needed < servers_needed_previous_timestep_map.iloc[row,column+1]):
                    servers_to_remove = servers_needed_previous_timestep_map.iloc[row,column+1] - servers_needed
                    ## DISMISSING
                    for j in range(remove_id_map.iloc[row,column+1], remove_id_map.iloc[row,column+1] + servers_to_remove):
                        if (column==1 and datacenter_slots['DC1'].sum() - slot_size > 0 and datacenter_slots['DC1'].iloc[server_gens_map[server_gen]]>0):
                            
                            buy_time = find_action_by_id(server_action_map, "DC1-" + server_gen + "-" + str(remove_id_map['DC1'].iloc[server_gens_map[server_gen]]))
                            if (buy_time is not None and time_step - buy_time >= 20 and
                                not server_action_map["DC1-" + server_gen + "-" + str(remove_id_map['DC1'].iloc[server_gens_map[server_gen]])].get('dismiss', False)):
                                datacenter_slots.at[server_gens_map[server_gen],'DC1'] -= slot_size
                                new_action = {
                                    "time_step": time_step,
                                    "datacenter_id": "DC1",
                                    "server_generation": server_gen,
                                    "server_id": "DC1-" + server_gen + "-" + str(remove_id_map['DC1'].iloc[server_gens_map[server_gen]]),
                                    "action": "dismiss"
                                }
                                server_action_map["DC1-" + server_gen + "-" + str(remove_id_map['DC1'].iloc[server_gens_map[server_gen]])]['dismiss'] = True
                                remove_id_map['DC1'].iloc[server_gens_map[server_gen]] += 1
                                solution.append(new_action)
                        elif (column==2 and datacenter_slots['DC2'].sum() - slot_size > 0 and datacenter_slots['DC2'].iloc[server_gens_map[server_gen]]>0):
                            buy_time = find_action_by_id(server_action_map, "DC2-" + server_gen + "-" + str(remove_id_map['DC2'].iloc[server_gens_map[server_gen]]))
                            if (buy_time is not None and time_step - buy_time >= 20 and
                                not server_action_map["DC2-" + server_gen + "-" + str(remove_id_map['DC2'].iloc[server_gens_map[server_gen]])].get('dismiss', False)):
                                datacenter_slots.at[server_gens_map[server_gen],'DC2'] -= slot_size
                                new_action = {
                                    "time_step": time_step,
                                    "datacenter_id": "DC2",
                                    "server_generation": server_gen,
                                    "server_id": "DC2-" + server_gen + "-" + str(remove_id_map['DC2'].iloc[server_gens_map[server_gen]]),
                                    "action": "dismiss"
                                }
                                server_action_map["DC2-" + server_gen + "-" + str(remove_id_map['DC2'].iloc[server_gens_map[server_gen]])]['dismiss'] = True
                                remove_id_map['DC2'].iloc[server_gens_map[server_gen]] += 1
                                solution.append(new_action)
                        elif (column==0):
                            if (datacenter_slots['DC4'].sum() - slot_size  > 0 and datacenter_slots['DC4'].iloc[server_gens_map[server_gen]]>0):
                                buy_time = find_action_by_id(server_action_map, "DC4-" + server_gen + "-" + str(remove_id_map['DC4'].iloc[server_gens_map[server_gen]]))
                                if (buy_time is not None and time_step - buy_time >= 20 and
                                    not server_action_map["DC4-" + server_gen + "-" + str(remove_id_map['DC4'].iloc[server_gens_map[server_gen]])].get('dismiss', False)):
                                    datacenter_slots.at[server_gens_map[server_gen],'DC4'] -= slot_size
                                    new_action = {
                                        "time_step": time_step,
                                        "datacenter_id": "DC4",
                                        "server_generation": server_gen,
                                        "server_id": "DC4-" + server_gen + "-" + str(remove_id_map['DC4'].iloc[server_gens_map[server_gen]]),
                                        "action": "dismiss"
                                    }
                                    server_action_map["DC4-" + server_gen + "-" + str(remove_id_map['DC4'].iloc[server_gens_map[server_gen]])]['dismiss'] = True
                                    remove_id_map['DC4'].iloc[server_gens_map[server_gen]] += 1
                                    solution.append(new_action)
                            elif(datacenter_slots['DC3'].sum() - slot_size  > 0) and datacenter_slots['DC3'].iloc[server_gens_map[server_gen]]>0: 
                                buy_time = find_action_by_id(server_action_map, "DC3-" + server_gen + "-" + str(remove_id_map['DC3'].iloc[server_gens_map[server_gen]]))
                                if (buy_time is not None and time_step - buy_time >= 20 and
                                    not server_action_map["DC3-" + server_gen + "-" + str(remove_id_map['DC3'].iloc[server_gens_map[server_gen]])].get('dismiss', False)):
                                    datacenter_slots.at[server_gens_map[server_gen],'DC3'] -= slot_size
                                    new_action = {
                                        "time_step": time_step,
                                        "datacenter_id": "DC3",
                                        "server_generation": server_gen,
                                        "server_id": "DC3-" + server_gen + "-" + str(remove_id_map['DC3'].iloc[server_gens_map[server_gen]]),
                                        "action": "dismiss"
                                    }
                                    server_action_map["DC3-" + server_gen + "-" + str(remove_id_map['DC3'].iloc[server_gens_map[server_gen]])]['dismiss'] = True
                                    remove_id_map['DC3'].iloc[server_gens_map[server_gen]] += 1
                                    solution.append(new_action)
                
                servers_needed_previous_timestep_map.iloc[row,column+1] = servers_needed  
                    
    print(datacenter_slots)
    return solution
                

seeds = [1111,2222] ## ANY NUMBER


demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    print("--------------" + str(seed))
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)
    #print(actual_demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)
    #print(solution)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./{seed}.json')

      