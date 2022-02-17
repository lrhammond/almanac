### Environments ###

import itertools
import random
from torch._C import dtype
from torch import tensor as tt
from torch.distributions import Categorical
from torch.nn.functional import one_hot as one_hot
import pickle
# from environments.test_envs import mg0, mg1, mg2, mg3
# from environments.overcooked_maps import oc0
import torch
import utils
import numpy as np
import specs

# Overcooked imports
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction

# MPE imports
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

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
        
        assert kind in ['mg', 'mmg', 'overcooked', 'smg', 'mpe']
   
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
        elif self.kind == 'mmg':
            model_joint_action = tuple([int(a) for a in joint_action])
        elif self.kind == 'smg':
            model_joint_action = tuple([int(a) for a in joint_action])
        elif self.kind == 'overcooked':
            model_joint_action = [self.model.action_space[joint_action[0]], self.model.action_space[joint_action[1]]]
        elif self.kind == 'mpe':
            model_joint_action = tuple(joint_action)

        self.state = self.model.step(model_joint_action)
        done = False

        return self.state, done

    def featurise(self, state):

        if self.kind == 'mg':
            features = self.model.featurise(state)
        elif self.kind == 'mmg':
            features = state
        elif self.kind == 'smg':
            features = self.model.featurise(state)
        elif self.kind == 'overcooked':
            features = self.model.featurise(state)
        elif self.kind == 'mpe':
            features = state

        return features

    def label(self, state):

        if self.kind == 'mg':
            labels = self.model.label(state)
        elif self.kind == 'mmg':
            labels = self.model.label(state)
        elif self.kind == 'smg':
            labels = self.model.label(state)
        elif self.kind == 'overcooked':
            label_dict = self.model.label(state)
            labels = []
            for k in label_dict.keys():
                if label_dict[k][0]:
                    labels.append(k.replace('_', '') + '0')
                if label_dict[k][1]:
                    labels.append(k.replace('_', '') + '1')
        elif self.kind == 'mpe':
            labels = self.model.label(state)

        return labels

    def get_obs_size(self):

        if self.kind == 'mg':
            obs_size = self.model.num_states
        elif self.kind == 'mmg':
            obs_size = self.model.state_size
        elif self.kind == 'smg':
            obs_size = self.model.state_size
        elif self.kind == 'overcooked':
            # obs_size = len(featurise(self.state))
            obs_size = 100
        elif self.kind == 'mpe':
            obs_size = len(self.model.get_state())

        return obs_size

    def get_act_sizes(self):

        if self.kind == 'mg':
            act_sizes = [len(a) for a in self.model.action_spaces]
        elif self.kind == 'mmg':
            act_sizes = self.model.action_sizes
        elif self.kind == 'smg':
            act_sizes = self.model.action_sizes
        elif self.kind == 'overcooked':
            act_sizes = [6,6]
        elif self.kind == 'mpe':
            act_sizes = self.model.num_agents * [5]

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
        self.state = self.reset()

    def step(self, joint_action):

        for i in range(self.num_players):
            assert joint_action[i] in self.action_spaces[i] 
        self.state = self.transition(self.state, joint_action)
        return self.state

    def reset(self):

        self.state = self.initial.sample()

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

# mg_0 = MarkovGame(num_players=num_players,
#                  state_space=state_space,
#                  action_spaces=action_spaces,
#                  transition=transition,
#                  initial=initial,
#                  labeller=labeller)

# mg_1 = MarkovGame(num_players=mg1.num_players,
#                  state_space=mg1.state_space,
#                  action_spaces=mg1.action_spaces,
#                  transition=mg1.transition,
#                  initial=mg1.initial,
#                  labeller=mg1.labeller)

# mg_2 = MarkovGame(num_players=mg2.num_players,
#                  state_space=mg2.state_space,
#                  action_spaces=mg2.action_spaces,
#                  transition=mg2.transition,
#                  initial=mg2.initial,
#                  labeller=mg2.labeller)

# mg_3 = MarkovGame(num_players=mg3.num_players,
#                  state_space=mg3.state_space,
#                  action_spaces=mg3.action_spaces,
#                  transition=mg3.transition,
#                  initial=mg3.initial,
#                  labeller=mg3.labeller)

class FixedStructuredMarkovGame:

    # Based on the procedure described in http://incompleteideas.net/RandomMDPs.html

    def __init__(self, num_state_vars, num_players, num_rules, num_antecedents, deterministic=False, single_init=False,
                 sink_prob=0.1):

        assert num_rules < num_state_vars
        assert num_antecedents < num_state_vars
        print("Num rules: " + str(num_rules))
        print("Num antedecents: " + str(num_antecedents))
        self.deterministic = deterministic
        self.sink_prob = sink_prob

        self.num_state_vars = num_state_vars

        self.sink_states = []
        for state_id in range(2 ** self.num_state_vars):
            if random.uniform(0, 1) < self.sink_prob:
                self.sink_states.append(state_id)

        self.num_players = num_players
        self.labels = ['l{}'.format(i) for i in range(num_state_vars)]
        self.rules = self.make_rules(num_rules, num_antecedents)
        # if deterministic:
        #     self.transitions = self.make_transitions()
        round_to = 1 if single_init else 3
        self.initial = tuple([round(random.uniform(0, 1), round_to) for _ in range(num_state_vars)])
        # self.initial = tuple([random.choice([0,1]) for _ in range(state_size)])
        self.state = self.reset()

    def make_rules(self, num_rules, num_antecedents):

        rules = {}
        for a in itertools.product(*[[0, 1] for _ in self.num_state_vars]):  # for each joint action
            rules[a] = {}
            """if random.uniform(0,1) < self.sink_prob:
                continue"""

            state_variables = random.sample(range(self.num_state_vars), k=num_rules)
            antecedents = random.sample(range(self.num_state_vars), k=num_antecedents)
            """print(self.state_size)
            print(state_variables)
            print(antecedents)
            input()"""
            for i in state_variables:
                rules[a][i] = {'antecedents': antecedents}
                for x in itertools.product(*[[0, 1] for _ in range(self.num_state_vars)]):


                """
                for x in itertools.product(*[[0, 1] for _ in range(num_antecedents)]):  # for each possible preceding set of state vars
                    # state_id = int("".join(str(bit) for bit in x), 2)

                    if self.deterministic:
                        rules[a][i][x] = random.choice([0, 1])
                    else:
                        rules[a][i][x] = round(random.uniform(0, 1), 3)
                """

        return rules

    def make_transitions(self):

        transitions = {}
        for s in itertools.product(*[[0, 1] for _ in range(self.state_size)]):
            s_1 = tuple(s)
            transitions[s_1] = {}
            for a in itertools.product(*[range(a_s) for a_s in self.action_sizes]):
                s_2 = list(s_1)
                for i in self.rules[a].keys():
                    x = tuple([s_1[y] for y in self.rules[a][i]['antecedents']])
                    s_2[i] = self.rules[a][i][x]
                transitions[s_1][a] = tuple(s_2)

        return transitions

    def reset(self):

        self.state = tuple([int(random.uniform(0, 1) > self.initial[i]) for i in range(self.num_state_vars)])

        return self.state

    def label(self, state):

        return tuple([self.labels[i] for i in range(self.state_size) if state[i] == 1])

    def featurise(self, state):

        return torch.tensor(state, dtype=float)

    def step(self, joint_action):

        # if self.deterministic:
        #     self.state = self.transitions[self.state][joint_action]
        # else:
        print("hiiiii")
        if self.state not in self.sink_states:
            new_state = list(self.state)

            for k_1 in self.rules[joint_action].keys():
                k_2 = tuple([self.state[a] for a in self.rules[joint_action][k_1]['antecedents']])
                new_state[k_1] = 1 if random.uniform(1, 0) > self.rules[joint_action][k_1][k_2] else 0
            self.state = tuple(new_state)
        else:
            print("SINK")
        return self.state

    def print(self):

        print("State: ", self.state)
        print("Labels: ", self.label(self.state))

    def create_prism_model(self, num, ldbas, location, policy=None, det=False):

        p = '' if policy == None else '-policy'
        d = '' if det == False else '-det'
        filename = location + '/prism_models/{}-{}-{}-{}{}{}.prism'.format(self.state_size, len(self.action_sizes),
                                                                           len(ldbas), num, p, d)

        with open(filename, 'w') as f:

            if policy == None:
                f.write('mdp\n\n\n')
            else:
                f.write('dtmc\n\n\n')

            eps_actions = []
            epsilon_transitions = []
            for i in range(len(ldbas)):
                eps_actions += ['eps_{}_{}'.format(i, j) for j in ldbas[i].ldba.eps_actions]
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

            for state in range(self.state_size):

                f.write('\nmodule STATE_{}\n\n'.format(state))
                f.write('    s{} : [0..2] init 2;\n'.format(state))
                p = self.initial[state]
                f.write('    [initialisation] !i -> {}:(s{}\'=0) + {}:(s{}\'=1);\n'.format(p, state, (round(1.0 - p, 3)),
                                                                                       state))

                for a in itertools.product(*[range(a_s) for a_s in self.action_sizes]):

                    action_guard = ' & '.join(['a{}={}'.format(i, a[i]) for i in range(self.num_players)])

                    if state not in self.rules[a].keys():
                        continue
                    else:
                        antecedents = self.rules[a][state]['antecedents']

                    for x in itertools.product(*[[0, 1] for _ in range(len(antecedents))]):
                        state_guard = ' & '.join(['s{}={}'.format(i, j) for (i, j) in zip(antecedents, x)])
                        p = self.rules[a][state][x]
                        new_state = '{}:(s{}\'=0) + {}:(s{}\'=1)'.format(p, state, round(1.0 - p, 3), state)
                        f.write('    [transition] ' + action_guard + ' & ' + state_guard + ' -> ' + new_state + ';\n')

                        """else:
                            print(state)
                            new_state = '{}:(s{}\'={})'.format(1, state, state)"""

                if len(epsilon_transitions) != 0:
                    f.write('    [transition] ' + any_eps_trans + ' -> 1.0:(s{0}\'=s{0});\n'.format(state))

                f.write('\nendmodule\n\n')

            for i in range(len(ldbas)):
                f.write(ldbas[i].create_prism_model(i, self.num_players))

            if policy == None:

                for i in range(len(self.action_sizes)):

                    f.write('\nmodule ACTION_{}\n\n'.format(i))
                    f.write('    a{} : [-1..{}] init -1;\n'.format(i, (highest_action_num)))

                    for j in range(self.action_sizes[i]):
                        f.write('    [action] true -> 1.0:(a{}\'={});\n'.format(i, j))
                    for e in eps_actions:
                        f.write('    [action] true -> 1.0:(a{}\'={});\n'.format(i, e))

                    f.write('\nendmodule\n\n')

            else:

                for i in range(len(self.action_sizes)):

                    f.write('\nmodule ACTION_{}\n\n'.format(i))
                    f.write('    a{} : [-1..{}] init -1;\n'.format(i, (highest_action_num)))

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

            for i in range(self.state_size):
                f.write('\nformula l{0} = (s{0}=1);'.format(i))
                f.write('\nlabel "l{0}" = l{0};'.format(i))


class StructuredMarkovGame:

    # Based on the procedure described in http://incompleteideas.net/RandomMDPs.html

    def __init__(self, state_size, action_sizes, num_rules, num_antecedents, deterministic=False, single_init=False, sink_prob=0.1):

        assert num_rules < state_size
        assert num_antecedents < state_size
        print("Num rules: " + str(num_rules))
        print("Num antedecents: " + str(num_antecedents))
        self.deterministic = deterministic
        self.sink_prob = sink_prob

        self.state_size = state_size

        self.sink_states = []
        for x in range(self.state_size):
            if random.uniform(0, 1) < self.sink_prob:
                self.sink_states.append(x)




        self.action_sizes = action_sizes
        self.num_players = len(action_sizes)
        self.labels = ['l{}'.format(i) for i in range(state_size)]
        self.rules = self.make_rules(num_rules, num_antecedents)
        # if deterministic:
        #     self.transitions = self.make_transitions()
        round_to = 1 if single_init else 3
        self.initial = tuple([round(random.uniform(0,1),round_to) for _ in range(state_size)])
        # self.initial = tuple([random.choice([0,1]) for _ in range(state_size)])
        self.state = self.reset()

    def make_rules(self, num_rules, num_antecedents):


        rules = {}
        for a in itertools.product(*[range(a_s) for a_s in self.action_sizes]): # for each joint action
            rules[a] = {}

            state_variables = random.sample(range(self.state_size), k=num_rules)
            antecedents = random.sample(range(self.state_size), k=num_antecedents)
            """print(self.state_size)
            print(state_variables)
            print(antecedents)
            input()"""
            for i in state_variables:
                rules[a][i] = {'antecedents': antecedents}
                for x in itertools.product(*[[0,1] for _ in range(num_antecedents)]): # for each possible preceding set of state vars
                    #state_id = int("".join(str(bit) for bit in x), 2)

                    if self.deterministic:
                        rules[a][i][x] = random.choice([0,1])
                    else:
                        rules[a][i][x] = round(random.uniform(0, 1),3)

        return rules

    def make_transitions(self):

        transitions = {}
        for s in itertools.product(*[[0,1] for _ in range(self.state_size)]):
            s_1 = tuple(s)
            transitions[s_1] = {}
            for a in itertools.product(*[range(a_s) for a_s in self.action_sizes]):
                s_2 = list(s_1)
                for i in self.rules[a].keys():
                    x = tuple([s_1[y] for y in self.rules[a][i]['antecedents']])
                    s_2[i] = self.rules[a][i][x]
                transitions[s_1][a] = tuple(s_2)

        return transitions

    def reset(self):

        self.state = tuple([int(random.uniform(0,1) > self.initial[i]) for i in range(self.state_size)])

        return self.state

    def label(self, state):

        return tuple([self.labels[i] for i in range(self.state_size) if state[i] == 1])

    def featurise(self, state):

        return torch.tensor(state, dtype=float)
                
    def step(self, joint_action):
        
        # if self.deterministic:
        #     self.state = self.transitions[self.state][joint_action]
        # else:
        print("hiiiii")
        if self.state not in self.sink_states:
            new_state = list(self.state)

            for k_1 in self.rules[joint_action].keys():
                k_2 = tuple([self.state[a] for a in self.rules[joint_action][k_1]['antecedents']])
                new_state[k_1] = 1 if random.uniform(1,0) > self.rules[joint_action][k_1][k_2] else 0
            self.state = tuple(new_state)
        else:
            print("SINK")
        return self.state

    def print(self):

        print("State: ", self.state)
        print("Labels: ", self.label(self.state))

    def create_prism_model(self, num, ldbas, location, policy=None, det=False):

        p = '' if policy == None else '-policy'
        d = '' if det == False else '-det'
        filename = location + '/prism_models/{}-{}-{}-{}{}{}.prism'.format(self.state_size, len(self.action_sizes), len(ldbas), num, p, d)

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

            for state in range(self.state_size):

                f.write('\nmodule STATE_{}\n\n'.format(state))
                f.write('    s{} : [0..2] init 2;\n'.format(state))
                p = self.initial[state]
                f.write('    [initialisation] !i -> {}:(s{}\'=0) + {}:(s{}\'=1);\n'.format(p,state,(round(1.0-p,3)),state))

                for a in itertools.product(*[range(a_s) for a_s in self.action_sizes]):
    
                    action_guard = ' & '.join(['a{}={}'.format(i,a[i]) for i in range(self.num_players)])

                    if state not in self.rules[a].keys():
                        continue
                    else:
                        antecedents = self.rules[a][state]['antecedents']

                    for x in itertools.product(*[[0,1] for _ in range(len(antecedents))]):

                        state_guard = ' & '.join(['s{}={}'.format(i,j) for (i,j) in zip(antecedents,x)])
                        p = self.rules[a][state][x]
                        new_state = '{}:(s{}\'=0) + {}:(s{}\'=1)'.format(p,state,round(1.0-p,3),state)
                        f.write('    [transition] ' + action_guard + ' & ' + state_guard + ' -> ' + new_state + ';\n')

                        """else:
                            print(state)
                            new_state = '{}:(s{}\'={})'.format(1, state, state)"""



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

            for i in range(self.state_size):
                f.write('\nformula l{0} = (s{0}=1);'.format(i))    
                f.write('\nlabel "l{0}" = l{0};'.format(i))
    

# Matrix Markov game (mmg) class
class MatrixMarkovGame:

    def __init__(self, state_size, action_sizes, labels, sparsity=0.7, structured_labels=True, nonlinearities=0.3):

        self.state_size = state_size
        self.action_sizes = action_sizes
        self.num_players = len(action_sizes)
        self.labels = labels
        self.labeller = lambda s: self.label(s) 
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
                if any([torch.equal(torch.dot(state, x), torch.sum(x)) for x in s]):
                    labels.append(l)
            return tuple(labels)
        else:
            return tuple([l for l, s in zip(self.labels, self.state_labels) if any([(state == x).all() for x in s])]) #check this works with == etc. TODO

    def reset(self):

        self.state = self.initial.sample()

        return self.state

    def step(self, joint_action):
        
        for i in range(len(self.masks)):
            if torch.equal(torch.dot(self.state, self.masks[i]), torch.sum(self.masks[i])):
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
    
    def create_prism_model(self, num, ldbas, location, policy=None, det=False):
        
        p = '' if policy == None else '-policy'
        d = '' if det == False else '-det'
        filename = location + '/prism_models/{}-{}-{}-{}{}{}.prism'.format(self.state_size, len(self.action_sizes), len(ldbas), num, p, d)

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
    
            f.write('module INIT\n\n')
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
        self.overcooked = OvercookedEnv.from_mdp(OvercookedGridworld.from_grid(self.overcooked_map, params_to_overwrite=self.params), info_level=0)
        self.state = self.overcooked.state
        self.labels = {}
        self.action_space = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST, Action.INTERACT, Action.STAY]

    def reset(self):

        self.labels = {}
        self.overcooked = OvercookedEnv.from_mdp(OvercookedGridworld.from_grid(self.overcooked_map, params_to_overwrite=self.params), info_level=0)
        self.state = self.overcooked.state

        return self.state

    def step(self, joint_action):

        _, mdp_infos = self.overcooked.mdp.get_state_transition(self.overcooked.state, joint_action, False, self.overcooked.mp)
        (self.state, _, _, _) = self.overcooked.step(joint_action)
        self.labels = mdp_infos["event_infos"]

        return self.state
    
    def featurise(self, state):
        
        original_features = self.overcooked.featurize_state_mdp(state)

        return tt(np.concatenate([original_features[0], original_features[1][-4:]]))

    def label(self, state):

        if state != self.state:
            print("Error: Can only obtain labels for current state")
            i = 4
            p = i[2]
            print(p)
            return
        else:
            return self.labels

    def print(self):

        print(repr(self.overcooked)[:-2])


class MPE:


    class Scenario(BaseScenario):

        def make_world(self, num_agents, num_landmarks, moving_landmarks, collisions):

            world = World()
            # add agents
            world.agents = [Agent() for _ in range(num_agents)]
            for i, agent in enumerate(world.agents):
                agent.name = 'agent_{}'.format(i)
                agent.collide = collisions
                agent.silent = True
            # add landmarks
            world.landmarks = [Landmark() for _ in range(num_landmarks)]
            for i, landmark in enumerate(world.landmarks):
                landmark.name = 'landmark %d' % i
                landmark.collide = collisions
                landmark.movable = moving_landmarks

            return world

        def reset_world(self, world, np_random):

            for _, agent in enumerate(world.agents):
                agent.color = np.array([0.25, 0.25, 0.25])
            for _, landmark in enumerate(world.landmarks):
                landmark.color = np.array([0.75, 0.75, 0.75])
            # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
            for agent in world.agents:
                agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        def reward(self, agent, world):
  
            return 0

        def observation(self, agent, world):

            entity_pos = []
            entity_vel = []
            for landmark in world.landmarks:
                entity_pos.append(landmark.state.p_pos)
                if landmark.movable:
                    entity_vel.append(landmark.state.p_vel)
            for agent in world.agents:
                entity_pos.append(agent.state.p_pos)
                entity_vel.append(agent.state.p_vel)

            return np.concatenate(entity_pos + entity_vel)

    class raw_env(SimpleEnv):
        
        def __init__(self, name=None, num_agents=0, num_landmarks=0, moving_landmarks=False, collisions=False, max_cycles=1000000):
            scenario = MPE.Scenario()
            world = scenario.make_world(num_agents, num_landmarks, moving_landmarks, collisions)
            super().__init__(scenario, world, max_cycles)
            self.metadata['name'] = name

    def __init__(self, name, num_agents, num_landmarks, moving_landmarks=False, collisions=False):

        self.name = name
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.env = make_env(MPE.raw_env)(name=name, num_agents=num_agents, num_landmarks=num_landmarks, moving_landmarks=moving_landmarks, collisions=collisions)
        self.state = self.reset()
        self.labels = ['at{}_{}'.format(a,l) for (a, l) in itertools.product(range(num_agents), range(num_landmarks))] + ["meets{}_{}".format(a1,a2) for (a1, a2) in itertools.product(range(num_agents), range(num_agents)) if a1 != a2]

    def get_state(self):

        return torch.tensor(self.env.last()[0])

    def reset(self):

        self.env.reset()

        return self.get_state()

    def step(self, joint_action):
        
        for a in joint_action:
            self.env.step(a)

        return self.get_state()

    def label(self, state, tolerance=0.2):

        labels = []
        for a in range(self.num_agents):
            for l in range(self.num_landmarks):
                l_pos = state[2*l:2*(l+1)]
                a_pos = state[2*(self.num_landmarks+a):2*(self.num_landmarks+a+1)]
                if abs(l_pos[0] - a_pos[0]) <= tolerance and abs(l_pos[1] - a_pos[1]) <= tolerance:
                    labels.append("{}at{}".format(a,l))
                for a2 in range(self.num_agents):
                    if a2 != a:
                        a2_pos = state[2*(self.num_landmarks+a2):2*(self.num_landmarks+a2+1)]
                        if abs(a_pos[0] - a2_pos[0]) <= tolerance and abs(a_pos[1] - a2_pos[1]) <= tolerance:
                            labels.append("{}meets{}".format(a,a2))
            
        return labels


### MARKOV GAME 0 ###
# Note that this game (and its variants) is for debugging purposes and so is very simple

### MARKOV GAME 0.0 ###
# Single player, single spec

# Number of players
num_players = 1

# State space
state_space = [tt(i) for i in range(2)]

# Action spaces
action_spaces = [('a','b')]

# Dists
d0 = Categorical(tt([ 0.5, 0.5 ]))
d1 = Categorical(tt([ 0.7, 0.3 ]))

# Initial state dist
initial = Categorical(tt([ 0.5, 0.5 ]))

# Transition function
def transition(s, a):
    if torch.equal(s,tt(0)):
        if a == ('a',):
            return s
        else:
            return tt(1) - s
    else:
        if a == ('a',):
            return tt(1) - s
        else:
            return s

# Labelling function
def labeller(s):
    if torch.equal(s, tt(0)):
        return ('phi',)
    else:
        return ('psi',)

# # Reward functions
# def reward_1(s_1, a, s_2):

#     sas = (s_1, a, s_2)
#     if utils.sas_eq((tt(0), ('a',), tt(0)), (s_1, a, s_2)):
#         return 2.0
#     elif utils.sas_eq((tt(0), ('a',), tt(1)), (s_1, a, s_2)):
#         return 4.0
#     elif torch.equal(s_1, tt(0)) and a == ('b',):
#         return -1.0
#     elif torch.equal(s_1, tt(1)):
#         return 0.0

# def reward_2(s_1, a, s_2):

#     if a == ('a',):
#         return 1.0
#     else:
#         return 2.0

### MARKOV GAME 0.1 ###
# Two players, single spec

# # Number of players
# num_players = 2

# # State space
# state_space = [tt(i) for i in range(2)]

# # Action spaces
# action_spaces = [('a','b'),('x','y')]

# # Dists
# d0 = Categorical(tt([ 0.7, 0.3 ]))
# d1 = Categorical(tt([ 0.3, 0.7 ]))

# # Initial state dist
# initial = Categorical(tt([ 1.0, 0 ]))

# # Transition function
# def transition(s, a):
#     if torch.equal(s,tt(0)):
#         if a == ('a','x'):
#             return s
#         else:
#             return tt(1) - s
#     else:
#         if a == ('b','y'):
#             return tt(1) - s
#         else:
#             return s

# # Labelling function
# def labeller(state):
#     if state == tt(0):
#         return ('phi',)
#     else:
#         return ('psi',)

# # Reward functions
# def reward_1(s_1, _, s_2):

#     if torch.equal(s_1, tt(0)) and torch.equal(s_2, tt(0)):
#         return 1.0
#     elif torch.equal(s_1, tt(1)) and torch.equal(s_2, tt(1)):
#         return 3.0
#     else:
#         return 0.0

# def reward_2(s_1, a, s_2):
#     sas = (s_1, a, s_2)
#     if utils.sas_eq((tt(0), ('b','y'), tt(1)), (s_1, a, s_2)):
#         return -2.0
#     elif utils.sas_eq((tt(1), ('a','x'), tt(0)), (s_1, a, s_2)):
#         return -2.0
#     else:
#         return 0.0   

### MARKOV GAME 0.2 ###
# Single player, two specs

# # Number of players
# num_players = 1

# # State space
# state_space = [tt(i) for i in range(2)]

# # Action spaces
# action_spaces = [('a','b')]

# # Dists
# d0 = Categorical(tt([ 0.5, 0.5 ]))
# d1 = Categorical(tt([ 0.7, 0.3 ]))

# # Initial state dist
# def initial(states):
#     return d1.sample()

# # Transition function
# def transition(s, a):
#     if s == tt(0):
#         if a == ('a',):
#             return s
#         else:
#             return d0.sample()
#     else:
#         if a == ('a',):
#             return s
#         else:
#             return d0.sample()

# # Labelling function
# def labeller(state):
#     if state == tt(0):
#         return ('phi',)
#     else:
#         return ('psi',)

mg_0 = MarkovGame(num_players=num_players,
                 state_space=state_space,
                 action_spaces=action_spaces,
                 transition=transition,
                 initial=initial,
                 labeller=labeller)