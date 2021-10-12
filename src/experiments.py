### Experiments ###

import learners
import specs
import envs
import os
import pickle
import copy
from time import time
import random
import numpy as np
import utils
import itertools
from csv import reader
import sys
import torch
from torch import tensor as tt

# from environments.test_envs import mg0

# Debugging
test_hps = {'actual_dist': True,
            'augment_data': False,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'entropy_weight': 10.0,
            'epsilon': 0.00,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'l2_weight' : 0.0001,
            # 'learning_rates': { 'actors': (('constant', 0.0001),),
            #                     'critics': ('constant', 0.001),
            #                     'lagrange_multiplier': (('constant', 0.005),) },
            'learning_rates': { 'actors': (('constant', 1.0),),
                                'critics': ('constant', 1.0),
                                'lagrange_multiplier': (('constant', 1.0),) },
            'models': { 'actors': {'type':'dnn', 'shape':[64,64]},
                        'critics': {'type':'dnn', 'shape':[64,64]} },
            'normalise_advantages' : True,
            'num_updates' : { 'actors': None,
                              'critics': None },
            'optimisers': { 'actors': 'adam',
                            'critics': 'adam' },
            'patient_updates': True,
            'sequential': False,
            'spec_reward': 10,
            'until_converged' : { 'actors': True,
                                  'critics': False },
            'update_after': { 'actors': 100,
                              'critics': 10 }}


def oc_test(root=utils.get_root(), max_steps=1000000, hps=test_hps, repetitions=1):

    location = '{}/experiments/oc_test'.format(root)

    env = envs.EnvWrapper('oc_test_game', 'overcooked', envs.OvercookedGame(envs.oc0.map, envs.oc0.recipes))

    # All
    # specifications = ('(F G phi) | (F G psi)', 'F (psi & X phi)')
    # reward_functions = ({'reward':mg0.reward_1, 'discount':0.8}, {'reward':mg0.reward_2, 'discount':0.9})
    # objectives = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    # Minimal spec
    specifications = ('F (tomatopickup0 | tomatopickup1)',)
    # specifications = ('(G psi) | (G phi)',)
    # specifications = ('(G phi) | (G psi)',)
    reward_functions = ()
    objectives = (1,)
    # objectives = (np.array((1.0,0.0)),np.array((0.0,1.0)))

    spec_controller = specs.Spec_Controller(specifications, location, load_from=location)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    learner = make_learner('almanac', obs_size, act_sizes, objectives, hps)
    prefix = "{}-{}-{}-{}".format(env.name, learner.name, 0, 0)

    for _ in range(repetitions):

        # Create learner
        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, verbose=True)
    
    # run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=True)




def debug(root=utils.get_root(), max_steps=100000, hps=test_hps, repetitions=1):

    location = '{}/experiments/debug'.format(root)

    env = envs.EnvWrapper('debug_game', 'mg', envs.mg_0)

    all_states = tt([[[1,0,1,0,0,0], # (0,0)
                    [1,0,0,1,0,0],   # (0,1)
                    [1,0,0,0,1,0],   # (0,2)
                    [1,0,0,0,0,1]],  # (0,3)
                    [[0,1,1,0,0,0],  # (1,0)
                    [0,1,0,1,0,0],   # (1,1)
                    [0,1,0,0,1,0],   # (1,2)
                    [0,1,0,0,0,1]]]) # (1,3)

    # All
    # specifications = ('(F G phi) | (F G psi)', 'F (psi & X phi)')
    # reward_functions = ({'reward':mg0.reward_1, 'discount':0.8}, {'reward':mg0.reward_2, 'discount':0.9})
    # objectives = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    # Minimal spec
    specifications = ('G (phi -> X psi)','G F phi')
    # specifications = ('(G psi) | (G phi)',)
    # specifications = ('(G phi) | (G psi)',)
    reward_functions = ()
    objectives = (2,1)
    # objectives = (np.array((1.0,0.0)),np.array((0.0,1.0)))

    spec_controller = specs.Spec_Controller(specifications, location, load_from=location)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    learner = make_learner('almanac', obs_size, act_sizes, objectives, hps)
    prefix = "{}-{}-{}-{}".format(env.name, learner.name, 0, 0)

    for _ in range(repetitions):

        # Create learner
        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000)
    
    run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=True)


# Experiment 0
exp0_hps = {'actual_dist': True,
            'augment_data': False,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'epsilon': 0,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'learning_rates': { 'actors': (('constant', 0.001),),
                                'critics': ('constant', 0.01),
                                'lagrange_multiplier': (('constant', 0.01),) },
            'models': { 'actors': {'type':'dnn', 'shape':[12]},
                        'critics': {'type':'dnn', 'shape':[12]} },
            'optimisers': { 'actors': 'sgd',
                            'critics': 'sgd' },
            'patient_updates': True,
            'spec_reward': 10,
            'update_after': { 'actors': 1000,
                              'critics': 1000 }}

def exp0(root, id, max_steps, hps=exp0_hps, repetitions=10):

    location = '{}/experiments/0'.format(root)

    if id["run"] == 1:
        env = envs.mg_1
        specifications = ('G (phi & psi)', '(!chi) U phi')
        reward_functions = ()
        objectives = ()
    elif id["run"] == 2:
        env = envs.mg_2
        specifications = ('G !chi', 'G F phi', 'F psi')
        reward_functions = ()
        objectives = ()
    elif id["run"] == 3:
        env = envs.mg_3
        specifications = ('F G psi', 'F phi')
        reward_functions = ()
        objectives = ()
    else:
        pass

    spec_controller = specs.Spec_Controller(specifications, location, load_from=location)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    for r in range(repetitions):

        # Create learner
        learner = make_learner('almanac', obs_size, act_sizes, objectives, hps)
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id["run"], r)

        run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000)

# Experiment 1
exp1_hps ={'actual_dist': True,
            'augment_data': True,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'epsilon': 0,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'learning_rates': { 'actors': (('constant', 0.001),),
                                'critics': ('constant', 0.75),
                                'lagrange_multiplier': (('constant', 0.01),) },
            'models': { 'actors': {'type':'dnn', 'shape':[24, 32, 24]},
                        'critics': {'type':'dnn', 'shape':[24, 24, 24]} },
            'optimisers': { 'actors': 'sgd',
                            'critics': 'sgd' },
            'patient_updates': True,
            'sequential': False,
            'spec_reward': 10,
            'update_after': { 'actors': 1000,
                              'critics': 1000 }}

def exp1(num_specs, num_actors, num_states, num_run, root, id, max_steps, hps=exp1_hps):
    
    assert num_specs in [1,2]

    location = '{}/experiments/1'.format(root)
    name = '{}-{}-{}-{}'.format(num_states, num_actors, num_specs, num_run)

    # Specifications
    labels = ['l{}'.format(i) for i in range(num_states)]
    possible_specs = [  lambda x : 'F {}'.format(x),\
                        lambda x : 'G {}'.format(x),\
                        lambda x : 'F G {}'.format(x),\
                        lambda x : 'G F {}'.format(x),\
                        lambda x : 'X (X {})'.format(x),\
                        lambda x, y : '{} U {}'.format(x, y),\
                        lambda x, y : 'F ({} & {})'.format(x, y),\
                        lambda x, y : 'G ({} | (X {}))'.format(x, y),\
                        lambda x, y : 'F G ({} | {})'.format(x, y),\
                        lambda x, y : 'G F ({} & (X {}))'.format(x, y)  ]
    possible_weights = [2, 5, 8]
    if num_states == 1:
        possible_specs = possible_specs[:5]
    
    # Map parameters
    state_size = num_states
    action_sizes = num_actors * [2]
    num_rules = max(1, int(0.5 * state_size))
    num_antecedents = max(1, int(0.5 * state_size))

    completed = False
    while not completed:

        # Form objectives
        specifications = [s(*random.sample(labels, k = s.__code__.co_argcount)) for s in random.sample(possible_specs, k=num_specs)]
        objectives = [list(range(len(specifications)))]
        weights = [1 for _ in specifications]

        # Form env
        smg = envs.StructuredMarkovGame(state_size, action_sizes, num_rules, num_antecedents, deterministic=True, single_init=True, sink_prob=0.8)
        env = envs.EnvWrapper('smg-{}-{}'.format(id, num_run), 'smg', smg)

        # Form input parameters and LDBAs
        spec_controller = specs.Spec_Controller(specifications, location, load_from=location)
        obs_size = env.get_obs_size() + sum(spec_controller.num_states)
        act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

        # Evaluate using PRISM
        spec_controller.save_props(location, name, weights)
        env.model.create_prism_model(num_run, spec_controller.specs, location)

        # Sometimes the MMG and spec combination generated is trivial, if so we retry
        prism_prob, prism_time = utils.run_prism(location, name, weights)
        if prism_prob == 0.0:
            continue
  
        # Create learner
        learner = make_learner('almanac', obs_size, act_sizes, objectives, hps)
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id, num_run)

        # Create state representations
        possible_game_states = list(itertools.product([0, 1], repeat=env.get_obs_size()))
        possible_spec_states = [[tuple((1 if i == j else 0) for i in range(s_s)) for j in range(s_s)] for s_s in spec_controller.num_states]
        possible_states = [utils.flatten(p_s) for p_s in itertools.product(possible_game_states, *possible_spec_states)]
        possible_state_tensors = torch.stack([tt(p_s).float() for p_s in possible_states])

        # Compare against Almanac
        probs = {'reg':[], 'det':[]}
        almanac_time = 0.0
        steps_taken = 0
        finished = False
        steps_per_round = 100
        while not finished:

            # Run Almamac
            resume = False if steps_taken == 0 else True
            t0 = time()
            run(learner, env, steps_per_round, spec_controller, [], objectives, location, prefix, resume=resume, verbose=True, num_plot_points=10)
            t1 = time()
            almanac_time += (t1 - t0)
            steps_taken += steps_per_round

            # Evaluate policies
            p_dists = learner.get_policy_dists(possible_states, possible_state_tensors)
            env.model.create_prism_model(num_run, spec_controller.specs, location, policy=p_dists)
            env.model.create_prism_model(num_run, spec_controller.specs, location, policy=p_dists, det=True)
            policy_prob, _ = utils.run_prism(location, name, weights, policy=True)
            det_policy_prob, _ = utils.run_prism(location, name, weights, policy=True, det=True)
            probs['reg'].append(policy_prob)
            probs['det'].append(det_policy_prob)

            # Check to see if we can stop
            target = 1.0 if prism_prob == None else prism_prob
            min_updates = int((hps['update_after']['actors'] * 10) / steps_per_round)
            policy_converged = utils.converged(probs['reg'], target=target, tolerance=0.01, minimum_updates=min_updates)
            det_policy_converged = utils.converged(probs['det'], target=target, tolerance=0.01, minimum_updates=min_updates)
            if (steps_taken >= max_steps) or abs(policy_prob - prism_prob) < 0.01 or abs(det_policy_prob - prism_prob) < 0.01 or (policy_converged and det_policy_converged):
                finished = True

        # Record results
        results = { 'prism_prob': prism_prob, 
                    'prism_time': prism_time,
                    'almanac_time': almanac_time,
                    'policy_prob': policy_prob,
                    'det_policy_prob': det_policy_prob,
                    'policy_converged': policy_converged,
                    'det_policy_converged': det_policy_converged }
        with open('{}/results/{}.txt'.format(location, name), 'w') as f:
            f.write(str(results))

        completed = True


# Experiment 2
exp2_hps_almanac ={'actual_dist': True,
            'augment_data': False,
            'buffers': { 'actors': {'size': 1000, 'batch': 32},
                         'critics': {'size': 1000, 'batch': 32} },
            'continue_prob': 0.9,
            'entropy_weight': 10.0,
            'epsilon': 0.00,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'l2_weight' : 0.0001,
            # 'learning_rates': { 'actors': (('constant', 0.0001),),
            #                     'critics': ('constant', 0.001),
            #                     'lagrange_multiplier': (('constant', 0.005),) },
            'learning_rates': { 'actors': (('constant', 1.0),),
                                'critics': ('constant', 1.0),
                                'lagrange_multiplier': (('constant', 1.0),) },
            'models': { 'actors': {'type':'dnn', 'shape':[64,64]},
                        'critics': {'type':'dnn', 'shape':[64,64]} },
            'normalise_advantages' : True,
            'num_updates' : { 'actors': None,
                              'critics': None },
            'optimisers': { 'actors': 'adam',
                            'critics': 'adam' },
            'patient_updates': True,
            'sequential': False,
            'spec_reward': 10,
            'until_converged' : { 'actors': True,
                                  'critics': False },
            'update_after': { 'actors': 100,
                              'critics': 10 }}

exp2_hps_rmappo ={'actual_dist': True,
            'augment_data': False,
            'buffers': {'size': 1000, 'batch': 32},
            'continue_prob': 0.9,
            'entropy_weight': 10.0,
            'epsilon': 0.00,
            'gamma_Phi' : 0.99,
            'kl_target' : 0.025,
            'l2_weight' : 0.0001,
            # 'learning_rates': { 'actors': (('constant', 0.0001),),
            #                     'critics': ('constant', 0.001),
            #                     'lagrange_multiplier': (('constant', 0.005),) },
            'learning_rates': { 'actors': (('constant', 1.0),),
                                'critics': ('constant', 1.0),
                                'lagrange_multiplier': (('constant', 1.0),) },
            'models': { 'actors': {'type':'dnn', 'shape':[64,64]},
                        'critics': {'type':'dnn', 'shape':[64,64]} },
            'normalise_advantages' : True,
            'num_updates' : None,
            'optimisers': 'adam',
            'patient_updates': False,
            'sequential': False,
            'spec_reward': 10,
            'until_converged' : False,
            'update_after': 100}


exp2_hps = {'almanac': exp2_hps_almanac, 'rmappo': exp2_hps_rmappo}

def exp2(num_specs, num_agents, num_landmarks, num_run, root, id, max_steps, hps=exp2_hps):

    location = '{}/experiments/2'.format(root)
    name = '{}-{}-{}-{}'.format(num_landmarks, num_agents, num_specs, num_run)

    # Create environment
    mpe = envs.MPE(name, num_agents, num_landmarks, moving_landmarks=False, collisions=False)
    env = envs.EnvWrapper('mpe-{}-{}'.format(id, num_run), 'mpe', mpe)

    # Create specifications
    labels = mpe.labels
    possible_specs = [  lambda x : 'F {}'.format(x),\
                        lambda x : 'G {}'.format(x),\
                        lambda x : 'F G {}'.format(x),\
                        lambda x : 'G F {}'.format(x),\
                        lambda x : 'X (X {})'.format(x),\
                        lambda x, y : '{} U {}'.format(x, y),\
                        lambda x, y : 'F ({} & {})'.format(x, y),\
                        lambda x, y : 'G ({} | (X {}))'.format(x, y),\
                        lambda x, y : 'F G ({} | {})'.format(x, y),\
                        lambda x, y : 'G F ({} & (X {}))'.format(x, y)  ]
    possible_weights = [2, 5, 8]
    if num_landmarks == 1:
        possible_specs = possible_specs[:5]

    # Form objectives TODO
    specifications = [s(*random.sample(labels, k = s.__code__.co_argcount)) for s in random.sample(possible_specs, k=num_specs)]
    objectives = [list(range(len(specifications)))]
    weights = [1 for _ in specifications]

    # Form input parameters and LDBAs
    spec_controller = specs.Spec_Controller(specifications, location, load_from=location)
    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    # Run learning algorithms
    for name in ['almanac', 'rmappo']:
        learner = make_learner(name, obs_size, act_sizes, objectives, hps[name])
        prefix = "{}-{}-{}-{}".format(env.name, learner.name, id, num_run)
        run(learner, env, max_steps, spec_controller, [], objectives, location, prefix, verbose=True, num_plot_points=1000)


# Experiment 3



def make_learner(learner_name, obs_size, act_sizes, objectives, hps):

    if learner_name == 'almanac':
        learner = learners.Almanac(obs_size, act_sizes, objectives, hps)
    else:
        print("Error: Agent name must be \'almanac\' or")
        return

    return learner


def run(learner, env, max_steps, spec_controller, reward_functions, objectives, location, prefix, num_plot_points=1000, evaluate=False, resume=False, verbose=True, until_converged=False):

    # Check for stupid mistakes
    num_specs = len(spec_controller.specs)
    num_reward_functions = len(reward_functions)
    num_objectives = num_specs + num_reward_functions
    if len(objectives) != num_objectives:
        print('Error: All objectives must be included in lexicographic ranking')
        return
    if len(learner.lrs['actors']) != num_objectives or len(learner.lrs['lagrange_multiplier']) != num_objectives:
        print('Error: Must be the same number of learning rates for actors and Lagrange multipliers as objectives')
        return

    # Prepare for saving scores
    score_interval = int(max_steps / num_plot_points)
    recent_scores = [0.0 for _ in objectives]
    total_scores = [0.0 for _ in objectives]

    suffix = "-eval" if evaluate else ""
    mode = 'a' if resume else 'w'
    with open('{}/scores/{}{}.txt'.format(location, prefix, suffix), mode) as f:
        obj_list = [str(i) for i in range(num_objectives)]
        if not resume:
            f.write("recent_" + ",recent_".join(obj_list) + ",total_" + ",total_".join(obj_list) + "\n")
        else:
            f.write(",".join(2 * [None for _ in range(num_objectives)]))

    # Avoid duplicating the computation of automaton transitions
    if learner.hps['augment_data']:
        transitions = dict()

    s = 0
    while s < max_steps:

        # Initialise environment
        game_state = env.reset()
        spec_states, acceptances = spec_controller.reset() 
        done = False
        t = 0
        current_discounts = np.ones(num_objectives)
        if learner.name == 'almanac':
            learner.Z = [dict() for _ in range(num_objectives)]
        if learner.name == 'rmappo':
            learner.reset()
        # print("End of episode, resetting...")

        while not done and s < max_steps:

            # Form state vectors 
            f_game_state = env.featurise(game_state)
            f_spec_states = spec_controller.featurise(spec_states)
            prod_state = torch.cat([f_game_state] + f_spec_states, 0)
            
            # Perform joint action
            if learner.name == 'rmappo':
                hidden = learner.get_hidden_states()
            joint_action = learner.act(prod_state)
            if learner.name == 'rmappo':
                new_hidden = learner.get_hidden_states()

            # If an epsilon transition is made the game (and possibly spec) state remains the same
            e_ts = spec_controller.is_epsilon_transition(joint_action, env.get_act_sizes())
            if len(e_ts) > 0:
                is_e_t = True
                new_game_state, done = game_state, done
            else:
                is_e_t = False
                new_game_state, done = env.step(joint_action)
    
            # if verbose:
            #     print(env.model.overcooked)

            # Update LDBAs
            label_set = env.label(new_game_state)
            new_spec_states, acceptances = spec_controller.step(e_ts, label_set)

            # for k in env.model.labels.keys():
            #     for a in [0,1]:
            #         if env.model.labels[k][a]:
            #             print("Agent {} did {}".format(a, k))
            #             print("hurrah")

            # if env.model.labels['tomato_pickup'][0] or env.model.labels['tomato_pickup'][1]:
            #     print("HYPE")

            # held = False
            # if new_game_state.players[0].held_object != None and game_state.players[0].held_object == None:
            #     print("Agent 0 is holding {}".format(new_game_state.players[0].held_object))
            #     held = True
            # if new_game_state.players[1].held_object != None and game_state.players[1].held_object == None:
            #     print("Agent 1 is holding {}".format(new_game_state.players[1].held_object))
            #     held = True

            # if held:
            #     print("w")

            # if label_set != []:
            #     print(label_set)              
            #     print("ww!")

            # Form new state vectors
            f_new_game_state = env.featurise(new_game_state)
            f_new_spec_states = spec_controller.featurise(new_spec_states)

            if random.random() > learner.hps['continue_prob']:
                done = True

            if not evaluate:

                # Form learning input
                info = {"s": game_state,
                        "f(s)": f_game_state,
                        "q": spec_states,
                        "f(q)": f_spec_states,
                        "a": joint_action,
                        "s'": new_game_state,
                        "f(s')": f_new_game_state,
                        "q'": new_spec_states,
                        "f(q')": f_new_spec_states,
                        "R": reward_functions,
                        "F": acceptances,
                        "t": t,
                        "o_s": objectives,
                        "done": done,
                        "is_e_t": is_e_t}
                if learner.hps['augment_data']:
                    if (e_ts, label_set) not in transitions.keys():
                        transitions[(e_ts, label_set)] = spec_controller.get_transitions(e_ts, label_set)
                    info["f(q)_s"], info["f(q')_s"], info["F_s"] = transitions[(e_ts, label_set)]
                else:
                    info["f(q)_s"], info["f(q')_s"], info["F_s"] = [f_spec_states], [f_new_spec_states], [acceptances]
                if learner.name == 'rmappo':
                    info["h"], info["h'"] = hidden, new_hidden
                
                # Update learners
                learner.step(info, objectives)
                
            # Record Almanac-esque scores
            discounts = []
            rewards = []
            for j in range(num_specs):
                discounts.append(learner.hps['gamma_Phi'] if acceptances[j] else 1.0)
                rewards.append(learner.hps['spec_reward'] if acceptances[j] else 0.0)
            for j in range(num_specs,num_objectives):
                discounts.append(reward_functions[j]['discount'] if acceptances[j] else 1.0)
                rewards.append(reward_functions[j]['reward'] if acceptances[j] else 0.0)
            current_discounts *= np.array(discounts)
            discounted_rewards = current_discounts * rewards
            # utilities = [sum(z[0] * z[1] for z in zip(o, discounted_rewards)) for o in objectives]
            utilities = [discounted_rewards[j] for j in objectives]
            recent_scores = [recent_scores[i] + utilities[i] for i in range(num_objectives)]

            # Update variables for next step
            game_state = new_game_state
            spec_states = new_spec_states
            t += 1
            s += 1

            # Save and occasionally print (average) score
            if s % score_interval == 0:
                total_scores = [total_scores[i] + recent_scores[i] for i in range(num_objectives)]
                rs_s = ",".join(["{:.5f}".format(rs / score_interval) for rs in recent_scores])
                ts_s = ",".join(["{:.5f}".format(ts / s) for ts in total_scores])
                if verbose:
                    print("Average Score ({}/{}): {} (recent)   {} (total)".format(int(s / score_interval), num_plot_points, rs_s, ts_s))
                with open('{}/scores/{}{}.txt'.format(location, prefix, suffix), 'a') as f:
                    f.write(rs_s + ',' + ts_s + '\n')
                recent_scores = [0.0 for _ in objectives]

    # Save model
    if not evaluate:
        learner.save_model(location, prefix)

    return

# debug()
# oc_test()
# exp1(num_specs=2, num_actors=2, num_states=3, num_run=1, root=utils.get_root(), id=1, max_steps=100000, hps=test_hps)
exp2(num_specs=2, num_agents=2, num_landmarks=2, num_run=1, root=utils.get_root(), id=1, max_steps=100000, hps=exp2_hps)