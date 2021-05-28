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
import numpy as np
import copy

# Run experiment instance
def run(hp, num, filename=None, modelname=None):

    env = envs.EnvWrapper(hp['env_name'], hp['env_type'], hp['env'])

    spec = hp['specs']
    
    model_constants = { 'discounts': hp['discounts'],
                        'l2_reg': hp['l2_regulariser_weight'],
                        'lrs': hp['learning_rates'] }
    almanac = learners.Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants)
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
    trained, _, score_at_end, _ = almanac.train(hp['steps'], env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
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
    # if trained:
    #     final_score = almanac.test(300000, env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
    # else:
    #     final_score = 0
    final_score = 0
    
    return trained, final_score, score_at_end


# Debugging

# debug   = { 'actor_neg_entropy_regulariser_weight': 0.1,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.9,
#             'discounts': { 'hasty': 0.8,
#                            'patient': 0.9 },
#             'env': envs.mg_0,
#             'env_name': 'debug_mg0',
#             'env_type': 'mg',
#             'epsilon': lambda e: 0.01,
#             'nat_grad_l2_regulariser_weight': 0.001,
#             'labeller': envs.l_0,
#             'learning_rates': { 'actor': ('constant', 0.01),
#                                 'critic': ('constant', 1.0),
#                                 'patient_nat_grad': ('constant', 0.1),
#                                 'hasty_nat_grad': ('constant', 0.1),
#                                 'mu': ('constant', 0.01) },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.05,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [32, 32, 32]),
#                         'critic': ('dnn', [32, 32, 32]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'sgd',
#                             'critic': 'sgd' },
#             'patient_updates': True,
#             'run_id': None,
#             'reward_weight': 10,
#             'specs': [ ('F G psi', 0.7),
#                        ('F phi', 0.3) ],
#             'steps': 100000 }


# Experiment 0

# # MG 1
# exp_0_1 = { 'actor_neg_entropy_regulariser_weight': 0.1,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.9,
#             'discounts': { 'hasty': 0.8,
#                            'patient': 0.9 },
#             'env': envs.mg_0,
#             'env_name': 'exp1_mg1',
#             'env_type': 'mg',
#             'epsilon': lambda e: 0.1,
#             'nat_grad_l2_regulariser_weight': 0.001,
#             'labeller': envs.l_0,
#             'learning_rates': { 'actor': ('constant', 0.1),
#                                 'critic': ('constant', 0.1),
#                                 'patient_nat_grad': ('constant', 0.01),
#                                 'hasty_nat_grad': ('constant', 0.01),
#                                 'mu': ('constant', 0.01), },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.05,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [16, 24, 16]),
#                         'critic': ('dnn', [16, 24, 16]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'sgd',
#                             'critic': 'sgd' },
#             'run_id': None,
#             'reward_weight': 10,
#             'specs': [ ('G (phi & psi)', 0.7),
#                        ('(!chi) U phi', 0.3) ],
#             'steps': 100000 }

# # MG 2
# exp_0_2 = { 'actor_neg_entropy_regulariser_weight': 0.1,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.9,
#             'discounts': { 'hasty': 0.8,
#                            'patient': 0.9 },
#             'env': envs.mg_0,
#             'env_name': 'exp1_mg1',
#             'env_type': 'mg',
#             'epsilon': lambda e: 0.1,
#             'nat_grad_l2_regulariser_weight': 0.001,
#             'labeller': envs.l_0,
#             'learning_rates': { 'actor': ('constant', 0.1),
#                                 'critic': ('constant', 1.0),
#                                 'patient_nat_grad': ('constant', 0.1),
#                                 'hasty_nat_grad': ('constant', 0.1),
#                                 'mu': ('constant', 0.1) },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.05,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [16, 24, 16]),
#                         'critic': ('dnn', [16, 24, 16]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'sgd',
#                             'critic': 'sgd' },
#             'run_id': None,
#             'reward_weight': 10,
#             'specs': [ ('G !chi', 0.4),
#                        ('G F phi', 0.4),
#                        ('F psi', 0.2) ],
#             'steps': 100000 }

# # MG 3
# exp_0_3 = { 'actor_neg_entropy_regulariser_weight': 0.1,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.9,
#             'discounts': { 'hasty': 0.8,
#                            'patient': 0.9 },
#             'env': envs.mg_0,
#             'env_name': 'exp1_mg1',
#             'env_type': 'mg',
#             'epsilon': lambda e: 0.1,
#             'nat_grad_l2_regulariser_weight': 0.001,
#             'labeller': envs.l_0,
#             'learning_rates': { 'actor': ('constant', 0.05),
#                                 'critic': ('constant', 1.0),
#                                 'patient_nat_grad': ('constant', 0.1),
#                                 'hasty_nat_grad': ('constant', 0.1),
#                                 'mu': ('constant', 0.01), },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.05,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [16, 24, 16]),
#                         'critic': ('dnn', [16, 24, 16]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'sgd',
#                             'critic': 'sgd' },
#             'run_id': None,
#             'reward_weight': 10,
#             'specs': [ ('F G psi', 0.7),
#                        ('F phi', 0.3) ],
#             'steps': 500000 }


# Experiment 1
old_hps = { 'actor_neg_entropy_regulariser_weight': 0.5,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.005, # Just changed from 0
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
            'l2_regulariser_weight': 0.005,
            'labeller': None,
            'learning_rates': { 'actor': ('constant', 0.005),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.05),
                                'hasty_nat_grad': ('constant', 0.05),
                                'mu': ('constant', 0.1) },
            'local': True,
            'nat_grad_convergence_tolerance': 0.25, # Just changed from 0.25
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': { 'actor': ('dnn', [24, 32, 24]),
                        'critic': ('dnn', [24, 24, 24]) },
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd'},
            'patient_updates': True,
            'run_id': None,
            'reward_weight': 10,
            'specs': None,
            'steps': 300000 }

blank_hps = { 'actor_neg_entropy_regulariser_weight': 0.5,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.01,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.95,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': None,
            'env': None,
            'env_name': None,
            'env_type': 'mmg',
            'epsilon': None,
            'l2_regulariser_weight': 0.005,
            'labeller': None,
            'learning_rates': None,
            'local': True,
            'nat_grad_convergence_tolerance': None,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': None,
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'patient_updates': True,
            'run_id': None,
            'reward_weight': 10,
            'specs': None,
            'steps': 150000 }

best_hps = { 'actor_neg_entropy_regulariser_weight': 0.5,
            'actor_neg_variance_regulariser_weight': 5.0,
            'actor_nondeterminism_regulariser_weight': 0.01,
            'actual_state_dist': True,
            'buffers': { 'size': 1000,
                         'batch': 32 },
            'continue_prob': 0.95,
            'critic_sum_value_regulariser_weight': 0.5,
            'critic_neg_variance_regulariser_weight': 0.5,
            'discounts': {'hasty': 0.9,
                            'patient': 0.99},
            'env': None,
            'env_name': None,
            'env_type': 'mmg',
            'epsilon': lambda e: 0.01,
            'l2_regulariser_weight': 0.005,
            'labeller': None,
            'learning_rates': {'actor': ('constant', 0.05),
                                'critic': ('constant', 0.1),
                                'patient_nat_grad': ('constant', 0.1),
                                'hasty_nat_grad': ('constant', 0.1),
                                'mu': ('constant', 0.1)},
            'local': True,
            'nat_grad_convergence_tolerance': 0.25,
            'max_critic_norm': 10.0,
            'max_nat_grad_norm': 10.0,
            'models': {'actor': ('dnn', [12, 16, 12]),
                        'critic': ('dnn', [12, 12, 12])},
            'model_id': None,
            'optimisers': { 'actor': 'sgd',
                            'nat_grad': 'sgd',
                            'critic': 'sgd' },
            'patient_updates': True,
            'run_id': None,
            'reward_weight': 10,
            'specs': None,
            'steps': 500000 }

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

        best_hps['env'] = mmg
        best_hps['env_name'] = None
        best_hps['labeller'] = mmg.label
        best_hps['model_id'] = None

        spec = []
        for s in list(zip(specifications, weights)):
            f = "experiments/1/specs/" + s[0] + '.pickle'
            if path.isfile(f):
                old_spec = pickle.load(open(f, "rb"))
                spec.append(old_spec)
            else:
                new_spec = specs.Spec(s[0], s[1])
                new_spec.save("experiments/1/specs")
                spec.append(new_spec)

        best_hps['specs'] = spec

        filename = 'experiments/1/scores/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, run_num)
        # modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
        # with open(filename, 'w') as f:
        #     f.write("State size: {}\n".format(state_size))
        #     a_s_line = ', '.join([str(a) for a in action_sizes])
        #     f.write("Action sizes: {}\n".format(a_s_line))
        #     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
        #     f.write("Specs: {}\n".format(s_line))
        #     f.write("Run: {}\n\n".format(run_num))

        specs_name = 'experiments/1/prism_specs/{}-{}-{}-{}.props'.format(state_size, num_actors, num_specs, run_num)
        with open(specs_name, 'w') as f:
            if num_specs == 1:
                f.write('Pmax=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            else:
                f.write('multi( Pmax=? [ X ( ' + specifications[0] + ' ) ] , Pmax=? [ X ( ' + specifications[1] + ' ) ] )\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[1] + ' ) ]\n\n')

        weights_name = 'experiments/1/prism_specs/{}-{}-{}-{}.weights'.format(state_size, num_actors, num_specs, run_num)
        with open(weights_name, 'w') as f:
            for w in weights:
                f.write('{}\n'.format(w))

        completed, _, _ = run(best_hps, run_num, filename)

state_size = int(sys.argv[1])
num_actors = int(sys.argv[2])
num_specs = int(sys.argv[3])
hp_num = int(sys.argv[4])

seed = (state_size - 1) * 100 + (num_actors - 1) * 20 + (num_specs - 1) * 10 + (hp_num - 1) * 1
# seed = 0

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# exp1(1,1,1,1)

exp1(state_size, num_actors, num_specs, hp_num)

# Experiment 2


# Experiment 3


# Code for hyperparameter grid search for experiment 1
def create_hps():

    learning_rates = [{ 'actor': ('constant', 0.01),
            'critic': ('constant', 0.5),
            'patient_nat_grad': ('constant', 0.1),
            'hasty_nat_grad': ('constant', 0.1),
            'mu': ('constant', 0.01) },

            {'actor': ('constant', 0.001),
            'critic': ('constant', 0.1),
            'patient_nat_grad': ('constant', 0.05),
            'hasty_nat_grad': ('constant', 0.05),
            'mu': ('constant', 0.05)},

            {'actor': ('constant', 0.05),
            'critic': ('constant', 0.1),
            'patient_nat_grad': ('constant', 0.1),
            'hasty_nat_grad': ('constant', 0.1),
            'mu': ('constant', 0.1)},

            {'actor': ('constant', 0.001),
            'critic': ('constant', 0.01),
            'patient_nat_grad': ('constant', 0.01),
            'hasty_nat_grad': ('constant', 0.01),
            'mu': ('constant', 0.01)},

            {'actor': ('constant', 0.0001),
            'critic': ('constant', 0.01),
            'patient_nat_grad': ('constant', 0.01),
            'hasty_nat_grad': ('constant', 0.01),
            'mu': ('constant', 0.01)},

            {'actor': ('constant', 0.01),
            'critic': ('constant', 0.1),
            'patient_nat_grad': ('constant', 0.05),
            'hasty_nat_grad': ('constant', 0.05),
            'mu': ('constant', 0.01)}]

    # actor_neg_entropy_regulariser_weights = [0.05, 0.5, 5.0]
    # actor_neg_variance_regulariser_weights = [0.05, 0.5, 5.0]
    # actor_nondeterminism_regulariser_weights = [0.0, 0.05, 0.5, 5.0]

    # critic_sum_value_regulariser_weights = [0.5,0.05,5.0]
    # critic_neg_variance_regulariser_weights = [0.5,0.05,5.0]

    discounts = [{'hasty': 0.85,
                'patient': 0.95},
                {'hasty': 0.9,
                'patient': 0.99},
                {'hasty': 0.75,
                'patient': 0.9}]

    epsilons = [lambda s: 0.01,
                lambda s: 0.05,
                lambda s: 0.1]

    # l2_regulariser_weights = [0.001, 0.01, 0.0001]

    nat_grad_convergence_tolerances = [0.15, 0.25, 0.35]

    models = [{ 'actor': ('dnn', [24, 32, 24]),
                'critic': ('dnn', [24, 24, 24])},
                # {'actor': ('dnn', [12, 24, 12]),
                # 'critic': ('dnn', [12, 12, 12])},
                {'actor': ('dnn', [12, 16, 12]),
                'critic': ('dnn', [12, 12, 12])},
                {'actor': ('dnn', [16, 24, 16]),
                'critic': ('dnn', [12, 16, 12])},
                ]

    possible_hps = []

    for learning_rate in learning_rates:
        # for actor_neg_entropy_regulariser_weight in actor_neg_entropy_regulariser_weights:
        #     for actor_neg_variance_regulariser_weight in actor_neg_variance_regulariser_weights:
                # for actor_nondeterminism_regulariser_weight in actor_nondeterminism_regulariser_weights:
                # for critic_sum_value_regulariser_weight in critic_sum_value_regulariser_weights:
                #     for critic_neg_variance_regulariser_weight in critic_neg_variance_regulariser_weights:
        for discount in discounts:
            for epsilon in epsilons:
                # for l2_regulariser_weight in l2_regulariser_weights:
                for nat_grad_convergence_tolerance in nat_grad_convergence_tolerances:
                    for model in models:
                        # for reward_weight in reward_weights:

                        hps = copy.deepcopy(blank_hps)
                        hps['learning_rates'] = learning_rate
                        hps['discounts'] = discount
                        hps['epsilon'] = epsilon
                        hps['nat_grad_convergence_tolerance'] = nat_grad_convergence_tolerance
                        hps['models'] = model

                        possible_hps.append(hps)



    return possible_hps

def exp1_hps(state_size, num_actors, num_specs, hp_num):

    if num_specs == 1:
        seed = 4
    else:
        seed = 12

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_hps = create_hps()
    hps = all_hps[hp_num]

    hps['run_id'] = 's{}-n{}'.format(num_specs, hp_num)

    completed = False
    while not completed:

        specifications = random.sample(possible_specs, num_specs)
        weights = random.sample(possible_weights, num_specs)
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
        action_sizes = [random.randint(2,4) for r in range(num_actors)]
        mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels, sparsity=0.6, structured_labels=True, nonlinearities=0.4)

        hps['env'] = mmg
        hps['env_name'] = None
        hps['labeller'] = mmg.label
        hps['model_id'] = None

        spec = []
        for s in list(zip(specifications, weights)):
            f = "experiments/1/specs/" + s[0] + '.pickle'
            if path.isfile(f):
                old_spec = pickle.load(open(f, "rb"))
                spec.append(old_spec)
            else:
                new_spec = specs.Spec(s[0], s[1])
                new_spec.save("experiments/1/specs")
                spec.append(new_spec)

        hps['specs'] = spec

        filename = 'experiments/1/scores/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, hp_num)
        # modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
        # with open(filename, 'w') as f:
        #     f.write("State size: {}\n".format(state_size))
        #     a_s_line = ', '.join([str(a) for a in action_sizes])
        #     f.write("Action sizes: {}\n".format(a_s_line))
        #     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
        #     f.write("Specs: {}\n".format(s_line))
        #     f.write("Run: {}\n\n".format(run_num))

        specs_name = 'experiments/1/prism_specs/{}-{}-{}-{}.props'.format(state_size, num_actors, num_specs, hp_num)
        with open(specs_name, 'w') as f:
            if num_specs == 1:
                f.write('Pmax=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            else:
                f.write('multi( Pmax=? [ X ( ' + specifications[0] + ' ) ] , Pmax=? [ X ( ' + specifications[1] + ' ) ] )\n\n')
                f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
                f.write('P=? [ X ( ' + specifications[1] + ' ) ]\n\n')

        weights_name = 'experiments/1/prism_specs/{}-{}-{}-{}.weights'.format(state_size, num_actors, num_specs, hp_num)
        with open(weights_name, 'w') as f:
            for w in weights:
                f.write('{}\n'.format(w))

        completed, final_score, score_at_end = run(hps, hp_num, filename)

    hp_scores_name = 'experiments/1/hp{}_scores.csv'.format(num_specs)
    with open(hp_scores_name, 'a') as f:
        f.write('{},{},{}\n'.format(hp_num, final_score, score_at_end))
