import learners
import specs
import envs
from os import path
import pickle
import copy
import random
import sys

### DEBUGGING ###

# debug_1 = { 'actor_neg_entropy_regulariser_weight': 0.1,
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

### EXPERIMENT 1 ###



# MG 1
exp_1_1 = { 'actor_neg_entropy_regulariser_weight': 0.1,
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
exp_1_2 = { 'actor_neg_entropy_regulariser_weight': 0.1,
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
exp_1_3 = { 'actor_neg_entropy_regulariser_weight': 0.1,
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

### EXPERIMENT 2 ###

### EXPERIMENT 3 ###

### RUN ###

# labels = ['phi','psi','chi','xi']
# possible_specs = ['G F psi', 'F G (!phi)', 'G ((!phi) | (X chi))']
# possible_weights = [0.2, 0.5, 0.8]
# num_trials = 5

# mmg_experiments = dict()
# for state_size in []:
#     for num_actors in []:
#         for num_specs in []:
#             k = (state_size, num_actors, num_specs)
#             mmgs = []
#             for n in range(num_trials):
#                 specifications = random.sample(possible_specs, num_specs)
#                 weights = random.sample(possible_weights, num_specs)
#                 sum_weights = sum(weights)
#                 weights = [w / sum_weights for w in weights]
#                 action_sizes = [random.randint(2,4) for r in range(num_actors)]
#                 mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels)
#                 mmg_hp = copy.deepcopy(mmg_hps)
#                 mmg_hp['env'] = envs.MatrixMarkovGame(state_size, action_sizes, labels)
#                 mmg_hp['env_name'] = '-'.join(k)
#                 mmg_hp['labeller'] = mmg.labeller
#                 mmg_hp['model_id'] = str(n)
#                 mmg_hp['specs'] = list(zip(specifications, weights))
#                 mmgs.append(mmg)
#             mmg_experiments[k] = mmgs

# mmg = envs.MatrixMarkovGame(3, [2,2], labels, sparsity=0.7, structured_labels=True, nonlinearities=0.5)
# labeller = mmg.labeller
# speccc = [('F G psi', 0.3), ('G (!phi | X chi)', 0.7)]
# test = copy.deepcopy(mmg_hps)
# test['env'] = mmg
# test['env_name'] = 'testing'
# test['labeller'] = mmg.labeller
# test['specs'] = speccc
# test['local'] = True


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

# state_size = int(sys.argv[1])
# num_actors = int(sys.argv[2])
# num_specs = int(sys.argv[3])
# run_num = int(sys.argv[4])

state_size = 1
num_actors = 1
num_specs = 1
run_num = 5

completed = False

while not completed:

    specifications = random.sample(possible_specs, num_specs)
    weights = random.sample(possible_weights, num_specs)
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]
    action_sizes = [random.randint(2,4) for r in range(num_actors)]
    mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels, sparsity=0.6, structured_labels=True, nonlinearities=0.4)

    # print("### SPECIFICATION ###")
    # print(specifications)

    mmg_hps['env'] = mmg
    mmg_hps['env_name'] = None
    mmg_hps['labeller'] = mmg.labeller
    mmg_hps['model_id'] = None
    mmg_hps['specs'] = list(zip(specifications, weights))

    filename = 'results/scores/mmg/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, run_num)
    # modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
    # with open(filename, 'w') as f:
    #     f.write("State size: {}\n".format(state_size))
    #     a_s_line = ', '.join([str(a) for a in action_sizes])
    #     f.write("Action sizes: {}\n".format(a_s_line))
    #     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
    #     f.write("Specs: {}\n".format(s_line))
    #     f.write("Run: {}\n\n".format(run_num))

    specs_name = 'specs/mmg/{}-{}-{}-{}.props'.format(state_size, num_actors, num_specs, run_num)
    with open(specs_name, 'w') as f:
        if num_specs == 1:
            f.write('Pmax=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
        else:
            f.write('multi( Pmax=? [ X ( ' + specifications[0] + ' ) ] , Pmax=? [ X ( ' + specifications[1] + ' ) ] )\n\n')
            f.write('P=? [ X ( ' + specifications[0] + ' ) ]\n\n')
            f.write('P=? [ X ( ' + specifications[1] + ' ) ]\n\n')

    weights_name = 'specs/mmg/{}-{}-{}-{}.weights'.format(state_size, num_actors, num_specs, run_num)
    with open(weights_name, 'w') as f:
        for w in weights:
            f.write('{}\n'.format(w))

    completed = run(mmg_hps, run_num, filename)

    # for speckle in possible_specs:
    #     s = specs.Spec(speckle, 1.0)
    #     s.save()

    # import time
    # start_time = time.time()

    # elapsed_time = time.time() - start_time
    # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))