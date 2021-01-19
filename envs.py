### Environments ###

import random
import torch.tensor as tt
from torch.distributions import Categorical
from torch.nn.functional import one_hot as one_hot
import pickle
from environments.test_envs import mg0, mg1, mg2, mg3
from environments.overcooked_maps import oc0
import torch
import utils
import numpy as np
import specs

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

# Use GPU if available
if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")


# Environment wrapper
class EnvWrapper:

    def __init__(self, name, kind, model):

        if not(kind != 'mg' or kind!= 'mmg' or kind != 'overcooked'):
            print("Error: Environment kind must be \'mg\' or \'mmg\' or \'overcooked\'")
            return
        else:
            self.name = name
            self.kind = kind
            self.model = model
            self.state = model.state

    def reset(self):

        # if self.kind == 'mg':
        #     self.state = self.model.reset()
        # elif self.kind == 'mmg':
        #     self.state = self.model.reset()
        # elif self.kind == 'overcooked':
        #     self.state = self.model.reset()
        self.state = self.model.reset()

        return self.state
        
    def step(self, joint_action):

        if self.kind == 'mg':
            model_joint_action = tuple([action_space[a.int()] for action_space, a in zip(self.model.action_spaces, joint_action)])
            # self.state = self.model.step(model_joint_action)
            # done = False
        elif self.kind == 'mmg':
            model_joint_action = tuple([int(a) for a in joint_action])
            # self.state = self.model.step(model_joint_action)
            # done = False
        elif self.kind == 'overcooked':
            model_joint_action = [self.model.action_space[joint_action[0]], self.model.action_space[joint_action[1]]]
            # self.state = self.model.step(model_joint_action)
            # done = False

        self.state = self.model.step(model_joint_action)
        done = False

        return self.state, done

    def featurise(self, state):

        if self.kind == 'mg':
            features = self.model.featurise(state)
        elif self.kind == 'mmg':
            features = state
        elif self.kind == 'overcooked':
            features = self.model.featurise(state)

        return features

    def label(self, state):

        if self.kind == 'mg':
            labels = self.model.label(state)
        elif self.kind == 'mmg':
            labels = self.model.label(state)
        elif self.kind == 'overcooked':
            label_dict = self.model.label(state)
            labels = []
            for k in label_dict.keys():
                if label_dict[k][0]:
                    labels.append('0_' + k)
                if label_dict[k][1]:
                    labels.append('1_' + k)

        return labels

    def get_obs_size(self):

        if self.kind == 'mg':
            obs_size = self.model.num_states
        elif self.kind == 'mmg':
            obs_size = self.model.state_size
        elif self.kind == 'overcooked':
            # obs_size = len(featurise(self.state))
            obs_size = 62

        return obs_size

    def get_act_sizes(self):

        if self.kind == 'mg':
            act_sizes = [len(a) for a in self.model.action_spaces]
        elif self.kind == 'mmg':
            act_sizes = self.model.action_sizes
        elif self.kind == 'overcooked':
            act_sizes = [6,6]

        return act_sizes

    def get_name(self):

        return self.name

    def get_kind(self):

        return self.kind

    def save(self):

        with open("models/{}/{}.pickle".format(self.kind, self.name), 'wb') as f:
            pickle.dump(self, f)


# Markov game class
class MarkovGame:
    
    def __init__(self, num_players, state_space, action_spaces, transition, initial, labeller):

        self.num_players = num_players
        self.state_space = state_space
        self.num_states = len(state_space)
        self.action_spaces = action_spaces
        self.transition = transition
        self.initial = initial
        self.labeller = labeller
        self.state = initial(state_space)

    def step(self, joint_action):

        for i in range(self.num_players):
            assert joint_action[i] in self.action_spaces[i] 
        self.state = self.transition(self.state, joint_action)
        return self.state

    def reset(self):

        self.state = self.initial(self.state_space)
        return self.state
    
    def featurise(self, state):

        return one_hot(self.state, self.num_states)
    
    def label(self, state):

        return self.labeller(state)

    def print(self):

        print("Current state: ", int(self.state))

    def test(self, steps=20):

        for t in range(steps):
            self.print()
            joint_action = tuple([random.choice(a) for a in self.action_spaces])
            print("Action: ", joint_action)
            self.step(joint_action)

mg_0 = MarkovGame(num_players=mg0.num_players,
                 state_space=mg0.state_space,
                 action_spaces=mg0.action_spaces,
                 transition=mg0.transition,
                 initial=mg0.initial,
                 labeller=mg0.labeller)

mg_1 = MarkovGame(num_players=mg1.num_players,
                 state_space=mg1.state_space,
                 action_spaces=mg1.action_spaces,
                 transition=mg1.transition,
                 initial=mg1.initial,
                 labeller=mg1.labeller)

mg_2 = MarkovGame(num_players=mg2.num_players,
                 state_space=mg2.state_space,
                 action_spaces=mg2.action_spaces,
                 transition=mg2.transition,
                 initial=mg2.initial,
                 labeller=mg2.labeller)

mg_3 = MarkovGame(num_players=mg3.num_players,
                 state_space=mg3.state_space,
                 action_spaces=mg3.action_spaces,
                 transition=mg3.transition,
                 initial=mg3.initial,
                 labeller=mg3.labeller)


# Matrix Markov game (mmg) class
class MatrixMarkovGame:

    def __init__(self, state_size, action_sizes, labels, sparsity=0.7, structured_labels=True, nonlinearities=0.3):

        self.state_size = state_size
        self.action_sizes = action_sizes
        self.num_players = len(action_sizes)
        self.labels = labels
        self.sparsity = sparsity
        self.nonlinearities = nonlinearities
        self.masks = self.create_masks(int(state_size * nonlinearities))
        self.transitions = [self.create_transitions() for r in range(int(state_size * nonlinearities) + 1)]
        self.structured_labels = structured_labels
        if state_size == 1:
            self.state_labels = [self.create_masks(1) for l in labels]
        else:
            self.state_labels = [self.create_masks(random.randint(1, state_size - 1)) for l in labels]
        self.initial = torch.distributions.bernoulli.Bernoulli(torch.rand(state_size))
        self.state = self.reset()

    def create_transitions(self):

        possible_actions = [[]]
        for action_size in self.action_sizes:
            new_possible_actions = utils.flatten([[p_a + [a] for p_a in possible_actions] for a in range(action_size)])
            possible_actions = new_possible_actions
        transitions = dict()
        matrices = [torch.rand(self.state_size, self.state_size) * torch.where(torch.rand(self.state_size, self.state_size) > self.sparsity, tt(1), tt(0)) for r in range(len(self.action_sizes))] + [torch.eye(self.state_size)]
        for p_a in possible_actions:
            transitions[tuple(p_a)] = random.choice(matrices)
        return transitions

    def create_masks(self, n):
        
        masks = set()
        while len(masks) < n:
            m = np.random.randint(0, high=2, size=self.state_size)
            if m.any():
                masks.add(tuple(m))
        return [tt(m).float() for m in masks]

    def create_state_labels(self):

        if self.structured_labels:
            return [[torch.randint(2, (self.state_size,)).float() for r in range(random.randint(1, self.state_size - 1))] for l in self.labels]
            # return [random.sample(range(self.state_size), random.randint(1, self.state_size)) for l in self.labels]
        else:
            return [set([torch.randint(2, (self.state_size,)).float() for r in range(random.randint(1, 2**self.state_size))]) for l in self.labels]

    def label(self, state):

        if self.structured_labels:
            labels = []
            for l, s in zip(self.labels, self.state_labels):
                if any([torch.dot(state, x) == torch.sum(x) for x in s]):
                    labels.append(l)
            return tuple(labels)
        else:
            return tuple([l for l, s in zip(self.labels, self.state_labels) if any([(state == x).all() for x in s])])

    def reset(self):

        return self.initial.sample()

    def step(self, joint_action):
        
        for i in range(len(self.masks)):
            if torch.dot(self.state, self.masks[i]) == torch.sum(self.masks[i]):
                dist = torch.matmul(self.state, self.transitions[i][joint_action])
                self.state = torch.distributions.bernoulli.Bernoulli(dist / utils.denom(dist.max())).sample()
                return self.state
        dist = torch.matmul(self.state, self.transitions[-1][joint_action])
        self.state = torch.distributions.bernoulli.Bernoulli(dist / utils.denom(dist.max())).sample()
        return self.state

    def print(self):
        print("State: ", self.state)
        print("Labels: ", self.label(self.state))
    
    def test(self, steps=100):

        for t in range(steps):
            self.print()
            joint_action = tuple([random.randint(0, a - 1) for a in self.action_sizes])
            print("Action: ", joint_action)
            self.step(joint_action)
    
    def create_prism_model(self, num, ldbas, policy=None, det=False, filename=None):
        
        p = '' if policy == None else '-policy'
        d = '' if det == False else '-det'
        if filename == None:
            filename = 'environments/markov_games/mmg/prism_models/{}-{}-{}-{}{}{}.prism'.format(self.state_size, len(self.action_sizes), len(ldbas), num, p, d)

        with open(filename, 'w') as f:
            
            if policy == None:
                f.write('mdp\n\n\n')
            else:
                f.write('dtmc\n\n\n')

            eps_actions = []
            epsilon_transitions = []
            for i in range(len(ldbas)):
                eps_actions += ['eps_{}_{}'.format(i,j) for j in ldbas[i].ldba.eps_actions]
                for e in ldbas[i].ldba.eps_actions:
                    eps_trans = '('
                    for a in range(self.num_players):
                        eps_trans += 'a{}=eps_{}_{}'.format(a, i, e)
                        if a != self.num_players - 1:
                            eps_trans += ' | '
                        else:
                            eps_trans += ')'
                    epsilon_transitions.append(eps_trans)
            max_action_num = max(self.action_sizes)
            any_eps_trans = ' | '.join(epsilon_transitions)

            for i in range(len(eps_actions)):
                f.write('const int {} = {};\n'.format(eps_actions[i], max_action_num + i))
            highest_action_num = max_action_num + len(eps_actions) - 1
    
            f.write('\n\nmodule INIT\n\n')
            f.write('    i : bool init false;\n')
            f.write('    [initialisation] !i -> 1.0:(i\'=true);\n')
            f.write('\nendmodule\n\n')
            f.write('\nmodule SYNC\n\n')
            f.write('    t : bool init true;\n')
            f.write('    [initialisation] !i -> 1.0:(t\' = false);\n')
            f.write('    [action] !t -> 1.0:(t\' = true);\n')
            f.write('    [transition] t -> 1.0:(t\' = false);\n')
            f.write('\nendmodule\n\n')

            masks = []
            for m in self.masks:
                nze = m.nonzero()
                mask = '('
                for i in range(len(nze)):
                    mask += 's{} = 1'.format(int(nze[i]))
                    if i != len(nze) - 1:
                        mask += ' & '
                    else:
                        mask += ')'
                masks.append(mask)

            for state in range(self.state_size):

                f.write('\nmodule STATE_{}\n\n'.format(state))
                f.write('    s{} : [0..2] init 2;\n'.format(state))
                init_dist = self.initial.probs
                p = init_dist[state]
                f.write('    [initialisation] !i -> {}:(s{}\'=0) + {}:(s{}\'=1);\n'.format((1.0-p),state,p,state))

                for i in range(len(masks) + 1):

                    if len(masks) != 0:
                        guard_mask = '[transition] '
                        if i != len(masks):
                            guard_mask += masks[i]
                            for j in range(i):
                                guard_mask += ' & !' + masks[j]
                        else:
                            guard_mask += '!' + ' & !'.join(masks)
                    else:
                        guard_mask = '[transition] true'
                    
                    for a in self.transitions[i].keys():
                        guard_action = '('
                        for k in range(len(a)):
                            guard_action += 'a{}={}'.format(k, a[k])
                            if k != len(a) - 1:
                                guard_action += ' & '
                            else:
                                guard_action += ')'

                        guard = guard_mask + ' & ' + guard_action
                        transition_probs = self.transitions[i][a]

                        probs = []
                        for l in range(self.state_size):
                            l_probs = transition_probs[:,l]
                            p = '('
                            for m in range(len(l_probs)):
                                p += '{}*s{}'.format(l_probs[m],m)
                                if m != self.state_size - 1:
                                    p += ' + '
                                else:
                                    p += ')'
                            probs.append(p)

                        max_prob = 'max(0.0, ' + ','.join(probs) + ')'   
                        q = state
                        new_line = '    ' + guard + ' -> (' + probs[q] + '/' + max_prob + '):(s{}\'=1)'.format(q) + ' + (1.0 - (' + probs[q] + '/' + max_prob + ')):(s{}\'=0);\n'.format(q)
                        f.write(new_line)

                if len(epsilon_transitions) != 0:
                    f.write('    [transition] ' + any_eps_trans + ' -> 1.0:(s{0}\'=s{0});\n'.format(state))        
            
                f.write('\nendmodule\n\n')

            for i in range(len(ldbas)):
                f.write(ldbas[i].create_prism_model(i, self.num_players))

            if policy == None:

                for i in range(len(self.action_sizes)):

                    f.write('\nmodule ACTION_{}\n\n'.format(i))
                    f.write('    a{} : [-1..{}] init -1;\n'.format(i,(highest_action_num)))

                    for j in range(self.action_sizes[i]):
                        f.write('    [action] true -> 1.0:(a{}\'={});\n'.format(i,j))
                    for e in eps_actions:
                        f.write('    [action] true -> 1.0:(a{}\'={});\n'.format(i,e))

                    f.write('\nendmodule\n\n')

            else:
                
                for i in range(len(self.action_sizes)):

                    f.write('\nmodule ACTION_{}\n\n'.format(i))
                    f.write('    a{} : [-1..{}] init -1;\n'.format(i,(highest_action_num)))

                    for k in policy.keys():

                        guard = ('    [action] ')
                        for j in range(self.state_size):
                            guard += 's{}={}'.format(j, k[j])
                            if j != self.state_size - 1:
                                guard += " & "
                  
                        remaining = k[self.state_size:]

                        for l in range(len(ldbas)):
                            n_l_s = ldbas[l].ldba.get_num_states()
                            ldba_state = remaining[:n_l_s]
                            guard += " & q{}={}".format(l, ldba_state.index(1))
                            remaining = remaining[n_l_s:]

                        action_probs = policy[k][i]
                        eps_action_probs = action_probs[self.action_sizes[i]:]

                        if not det:
                            action_choice = ''
                            for a in range(self.action_sizes[i]):
                                action_choice += '{}:(a{}\'={})'.format(action_probs[a], i, a)
                                if a != self.action_sizes[i] - 1:
                                    action_choice += ' + '
                            for b in range(len(eps_actions)):
                                action_choice += ' + {}:(a{}\'={})'.format(eps_action_probs[b], i, eps_actions[b])

                        else:
                            action = int(torch.argmax(action_probs))
                            poss_actions = list(range(self.action_sizes[i])) + eps_actions
                            action_choice = '1.0:(a{}\'={})'.format(i, poss_actions[action])
                        
                        f.write(guard + ' -> ' + action_choice + ';\n')

                    f.write('\nendmodule\n\n')

            if self.structured_labels:
                for l, s_ls in zip(self.labels, self.state_labels):

                    masks = []
                    for m in s_ls:
                        nze = m.nonzero()
                        mask = '('
                        for i in range(len(nze)):
                            mask += 's{} = 1'.format(int(nze[i]))
                            if i != len(nze) - 1:
                                mask += ' & '
                            else:
                                mask += ')'
                        masks.append(mask)

                    f.write('\nformula ' + l + ' = (' + ' | '.join(masks) + ');')
                    f.write('\nlabel "' + l + '" = ' + l + ';')

            else:
                print("Error: Converting unstructured labels to PRISM not yet supported")


class OvercookedGame:

    def __init__(self, overcooked_map, recipes):

        self.overcooked_map = overcooked_map
        self.recipes = recipes
        self.params = {"start_bonus_orders": [], "rew_shaping_params": None, }
        recipe_dicts = [{"ingredients": r["ingredients"]} for r in recipes]
        recipe_times = [r["time"] for r in recipes]
        self.params.update({"start_all_orders" : recipe_dicts, "recipe_times": recipe_times})
        self.overcooked = OvercookedEnv.from_mdp(OvercookedGridworld.from_grid(self.overcooked_map, params_to_overwrite=self.params))
        self.state = self.overcooked.state
        self.labels = {}
        self.action_space = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST, Action.INTERACT, Action.STAY]

    def reset(self):

        self.labels = {}
        self.overcooked = OvercookedEnv.from_mdp(OvercookedGridworld.from_grid(self.overcooked_map, params_to_overwrite=self.params))
        self.state = self.overcooked.state

        return self.state

    def step(self, joint_action):

        (self.state, _, _, env_info) = self.overcooked.step(joint_action)
        self.labels = env_info["event_infos"]

        return self.state
    
    def featurise(self, state):

        return self.overcooked.featurize_state_mdp(state)

    def label(self, state):

        if state != self.state:
            print("Error: Can only obtain labels for current state")
            return
        else:
            return self.labels

    def print(self):

        print(repr(self.overcooked)[:-2])
