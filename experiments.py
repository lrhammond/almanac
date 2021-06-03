### Experiments ###

import learners
import specs
import envs
from os import path
import pickle
import copy
import random
import sys
import torch


# Debugging
debug   = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': envs.mg_0,
            'env_name': 'debug_mg0',
            'env_type': 'mg',
            'epsilon': lambda e: 0.01,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': envs.l_0,
            'learning_rates': { 'actor': ('constant', 0.01),
                                'critic': ('constant', 1.0),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.01) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [32, 32, 32]),
                        'critic': ('dnn', [32, 32, 32]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'patient_updates': True,
            'run_id': None,
            'reward_weight': 10,
            'specs': [ ('F G psi', 0.7),
                       ('F phi', 0.3) ],
            'steps': 100000 }


# Experiment 0

# MG 1
exp_0_1 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': envs.mg_0,
            'env_name': 'exp1_mg1',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': envs.l_0,
            'learning_rates': { 'actor': ('constant', 0.1),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.01),
                                'hasty_nat_grad': ('constant', 0.01),
                                'mu': ('constant', 0.01), },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'reward_weight': 10,
            'specs': [ ('G (phi & psi)', 0.7),
                       ('(!chi) U phi', 0.3) ],
            'steps': 100000 }

# MG 2
exp_0_2 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': envs.mg_0,
            'env_name': 'exp1_mg1',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': envs.l_0,
            'learning_rates': { 'actor': ('constant', 0.1),
                                'critic': ('constant', 1.0),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.1) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'reward_weight': 10,
            'specs': [ ('G !chi', 0.4),
                       ('G F phi', 0.4),
                       ('F psi', 0.2) ],
            'steps': 100000 }

# MG 3
exp_0_3 = { 'actor_neg_entropy_regulariser_weight': 0.1,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.9,
            'discounts': { 'hasty': 0.8,
                           'patient': 0.9 },
            'env': envs.mg_0,
            'env_name': 'exp1_mg1',
            'env_type': 'mg',
            'epsilon': lambda e: 0.1,
            'nat_grad_l2_regulariser_weight': 0.001,
            'labeller': envs.l_0,
            'learning_rates': { 'actor': ('constant', 0.05),
                                'critic': ('constant', 1.0),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.01), },
            'local': True,
            'nat_grad_convergence_tolerance': 0.05,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [16, 24, 16]),
                        'critic': ('dnn', [16, 24, 16]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'run_id': None,
            'reward_weight': 10,
            'specs': [ ('F G psi', 0.7),
                       ('F phi', 0.3) ],
            'steps': 500000 }


# Experiment 1
mmg_hps = { 'actor_neg_entropy_regulariser_weight': 0.5,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.0,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.95,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': { 'hasty': 0.85,
                           'patient': 0.95 },
            'env': None,
            'env_name': None,
            'env_type': 'mmg',
            'epsilon': lambda e: 0.05,
            'l2_regulariser_weight': 0.001,
            'labeller': None,
            'learning_rates': { 'actor': ('constant', 0.001),
                                'critic': ('constant', 0.75),
                                'patient_nat_grad': ('constant', 1.00),
                                'hasty_nat_grad': ('constant', 1.00),
                                'mu': ('constant', 0.01) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.25,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [24, 32, 24]),
                        'critic': ('dnn', [24, 24, 24]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'adam',
                            'critic': 'sgd' },
            'patient_updates': True,
            'run_id': None,
            'reward_weight': 10,
            'specs': None,
            'steps': 5000 }

labels = ['phi','psi','chi','xi']
possible_specs = ['G F psi', 'F G ((!phi) | (!xi))', 'G ((!phi) | (X (chi)))', 'F xi', 'G ((!psi) | (F phi))', 'G chi', '(!xi) U psi']
possible_weights = [0.2, 0.5, 0.8]

def exp1(state_size, num_actors, num_specs, run_num):

    completed = False
    while not completed:

        specifications = random.sample(possible_specs, num_specs)
        weights = random.sample(possible_weights, num_specs)
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
        action_sizes = [random.randint(2,4) for r in range(num_actors)]
        mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels, sparsity=0.6, structured_labels=True, nonlinearities=0.4)

        mmg_hps['env'] = mmg
        mmg_hps['env_name'] = None
        mmg_hps['labeller'] = mmg.labeller
        mmg_hps['model_id'] = None
        mmg_hps['specs'] = list(zip(specifications, weights))

        # filename = 'results/experiment_1/scores/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, run_num)
        # modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
        # with open(filename, 'w') as f:
        #     f.write("State size: {}\n".format(state_size))
        #     a_s_line = ', '.join([str(a) for a in action_sizes])
        #     f.write("Action sizes: {}\n".format(a_s_line))
        #     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
        #     f.write("Specs: {}\n".format(s_line))
        #     f.write("Run: {}\n\n".format(run_num))

        specs_name = 'results/experiment_1/specs/matrix_markov_games/{}-{}-{}-{}.props'.format(state_size, num_actors, num_specs, run_num)
        with open(specs_name, 'w') as f:
            if num_specs == 1:
                f.write('Pmax=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            else:
                f.write('multi( Pmax=? [ X ( ' + specifications[0] + ' ) ] , Pmax=? [ X ( ' + specifications[1] + ' ) ] )\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[1] + ' ) ]\n\n')

        weights_name = 'results/experiment_1/specs/matrix_markov_games/{}-{}-{}-{}.weights'.format(state_size, num_actors, num_specs, run_num)
        with open(weights_name, 'w') as f:
            for w in weights:
                f.write('{}\n'.format(w))

        completed = run(mmg_hps, run_num, filename)

        possible_game_states = list(itertools.product([0, 1], repeat=env.get_obs_size()))
        possible_spec_states = []
        for s_s in self.ldba_state_sizes:
            spec_states = [[0 for i in range(s_s)] for j in range(s_s)]
            for k in range(s_s):
                spec_states[k][k] = 1
            possible_spec_states.append([tuple(s_s) for s_s in spec_states])
        possible_product_states = itertools.product(possible_game_states, *possible_spec_states)
        self.possible_states = [tuple(utils.flatten(p_s)) for p_s in possible_product_states]
        self.possible_state_tensors = torch.stack([tt(p_s).float() for p_s in self.possible_states])
        self.policy_dists = None

        if s % score_interval == 0:

            # Sometimes the mmg generated is trivial, if so we return false and regenerate the mmg
            if self.env_kind == 'mmg':
                if len(first_50) < 60:
                    first_50.append(recent_game_score/score_interval)
                else:
                    if not on2awinner:
                        if np.var(first_50[10:]) < 0.01:
                            return False
                        else:
                            on2awinner = True
            
            last_score_interval.append(recent_game_score / score_interval)

            with open(filename, 'a') as f:
                f.write('{}, {}, {}\n'.format(int(s / score_interval), average_game_score, recent_game_score / score_interval))
            recent_game_score = 0

            if s > (steps / 10) and average_game_score > best + 0.05:
                if self.env_kind == 'mmg':
                    self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                best = average_game_score

        if len(last_score_interval) == score_interval:
            if np.var(last_score_interval) < 0.001:
                average_game_score = game_score / s
                if average_game_score > best + 0.05:
                    if self.env_kind == 'mmg':
                        self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                    best = average_game_score
                s = steps + 1


# Experiment 2


# Experiment 3


# Run experiment instance
def run(hp, num, filename=None, modelname=None):

    env = envs.EnvWrapper(hp['env_name'], hp['env_type'], hp['env'], hp['labeller'])
    spec = []
    for s in hp['specs']:
        f = "specs/" + s[0] + '.pickle'
        if path.isfile(f):
            old_spec = pickle.load(open(f, "rb"))
            spec.append(old_spec)
        else:
            new_spec = specs.Spec(s[0], s[1])
            new_spec.save()
            spec.append(new_spec)
    model_constants = { 'discounts': hp['discounts'],
                        'l2_reg': hp['l2_regulariser_weight'],
                        'lrs': hp['learning_rates'] }
    almanac = learners.Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants, hp['model_id'])
    train_constants = { 'continue_prob': hp['continue_prob'],
                        'epsilon': hp['epsilon'],
                        'nat_grad_tolerance': hp['nat_grad_convergence_tolerance'],
                        'neg_ent_reg': hp['actor_neg_entropy_regulariser_weight'],
                        'non_det_reg': hp['actor_nondeterminism_regulariser_weight'],
                        'sum_val_reg': hp['critic_sum_value_regulariser_weight'],
                        'a_neg_var_reg': hp['actor_neg_variance_regulariser_weight'],
                        'c_neg_var_reg': hp['critic_neg_variance_regulariser_weight'],
                        'max_nat_grad_norm': hp['max_nat_grad_norm'],
                        'max_critic_norm': hp['max_critic_norm'],
                        'reward_weight': hp['reward_weight'] }
    trained = almanac.train(hp['steps'], env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
    if hp['env_type'] == 'mmg':
        env.model.create_prism_model(num, spec)
        env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists())
        env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists(), det=True)
    if modelname != None:
        almanac.lrs = None
        almanac.patient_buffer = None
        almanac.hasty_buffer = None
        with open(modelname, 'wb') as f:
            pickle.dump(almanac, f)
    
    return trained



def make_learner(learner_name, obs_size, act_sizes):

    pass



def experiment(env, formulae, learner_name):

    spec_controller = specs.Spec_Controller(formulae)

    if augment_data:
        spec_states_transition_matrix = spec_controller.get_transition_matrix()

    obs_size = env.get_obs_size() + sum(spec_controller.num_states)
    act_sizes = [a_s + sum(spec_controller.epsilon_act_sizes) for a_s in env.get_act_sizes()]

    learner = make_learner(learner_name, obs_size, act_sizes)

    run()

    # Save experiment inputs
    spec_controller.save_model(location)
    rewards, objectives  


def run(learner, env, max_steps, spec_controller, reward_functions, objectives, save_location, continue_prob=0.99, augment_data=True, num_plot_points=1000):

    score_interval = int(max_steps / num_plot_points)
    recent_scores = [0.0 for o in objectives]
    total_scores = [0.0 for o in objectives]

    if augment_data:
        transitions = dict()

    location = './{}/{}/{}/'.format(save_location, env.name, learner.name)

    with open(location + 'scores.txt', 'w') as f:
        f.write('Average Scores\n')

    num_objectives = len(objectives)
    s = 0

    while s < max_steps:

        # Initialise environment
        game_state = env.reset()
        spec_states, acceptances = spec_controller.reset() 
        done = False
        t = 0
        # print("End of episode, resetting...")

        while not done and s < max_steps:

            # Form state vectors 
            f_game_state = env.featurise(game_state)
            f_spec_states = spec_controller.featurise(spec_states)
            prod_state = torch.cat([f_game_state] + f_spec_states, 0)
            
            # Perform joint action
            joint_action = learner.act(prod_state)

            # If an epsilon transition is made the game (and possibly spec) state remains the same
            e_ts = spec_controller.is_epsilon_transition(joint_action)
            if len(e_ts) > 0:
                is_e_t = True
                new_game_state, done = game_state, done
            else:
                is_e_t = False
                new_game_state, done = env.step(joint_action)
            
            # Update LDBAs
            label_set = env.label(new_game_state)
            new_spec_states, acceptances = spec_controller.step(e_ts, label_set)

            # Form new state vectors
            f_new_game_state = env.featurise(new_game_state)
            f_new_spec_states = spec_controller.featurise(new_spec_states)

            if random.random() > continue_prob:
                done = True

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
                    "d": done,
                    "is_e_t": is_e_t}
            if augment_data:
                if (e_ts, label_set) not in transitions.keys():
                    transitions((e_ts, label_set)) = spec_controller.get_transitions(e_ts, label_set)
                info["f(q)_s"], info["f(q')_s"], info["F_s"] = transitions[(e_ts, label_set)]
            else:
                info["f(q)_s"], info["f(q')_s"], info["F_s"] = [f_spec_states], [f_new_spec_states], [acceptances]
            
            # Update learners
            payoffs = learner.step(info, augment_data, continue_prob)
            recent_scores = [recent_scores[i] + payoffs[i] for i in range(num_objectives)]

            # Update variables for next step
            game_state = new_game_state
            spec_states = new_spec_states
            t += 1
            s += 1

            # Save and occasionally print (average) score
            if s % score_interval == 0:
                total_scores = [total_scores[i] + recent_scores[i] for i in range(num_objectives)]
                print("Average Score ({}/{}): {} (recent)   {} (total)".format(s / score_interval, num_plot_points, recent_scores / score_interval, total_scores / s))
                with open(location + 'scores.txt', 'a') as f:
                    for r_s in recent_scores:
                        f.write('{},'.format(r_s / score_interval))
                    for t_s in recent_scores:
                        f.write('{},'.format(t_s / s))
                    f.write('\n')
                recent_scores = [0.0 for o in objectives]

    # Save model
    learner.save_model(location)
    
























    for i in range(interacts):

        state = env.reset()

        action = agent.act(state)
        step += 1

        if int_action:
            action = int(action)
        if type(action)!=int:
            action = action.squeeze().cpu().float()

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        try:
            cost = info['cost']
        except:
            try:
                cost = info['constraint_costs'][0]
            except:
                cost = 0

        if mode==1:
            r = reward
        elif mode==2:
            r = reward-cost
        elif mode==3:
            r = [-cost, reward]
        elif mode==4:
            r = [reward, cost]
        elif mode==5:
            r = [reward, -cost]

        #time.sleep(0.0001)
        agent.step(state, action, r, next_state, done)

        if done or (step >= max_ep_length):
            step = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

        with open(filename, 'a') as f:
            f.write('{},{}\n'.format(reward, cost))