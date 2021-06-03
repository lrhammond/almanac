### Learners ###

import torch
import torch.tensor as tt
import random
from torch import pow as power
import time
from collections import namedtuple, deque
import torch.nn as nn
from torch import optim, ones_like
from torch.distributions import Categorical
from torch.nn.functional import one_hot as one_hot
import cProfile
import itertools
import numpy as np
from os import path
import pickle
import specs
import envs
import utils


# Use GPU if available
if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")


# Basic DNN module
class DNN(nn.Module):

    def __init__(self, in_size, out_sizes, output, hidden, bias=True):

        super(DNN,self).__init__()

        self.dims = [in_size] + hidden + [sum(out_sizes)]
        self.output = output
        self.out_sizes = out_sizes
        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1], bias=bias))

    def forward(self, x):

        # Apply to ReLU activations to all layers apart from the last
        x = x.float()
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))

        # Compute last layer
        x = x.float()
        x = self.layers[-1](x)

        # Return output of DNN as a tuple
        if self.output == 'linear':
            return x
        elif self.output == 'softmax':
            d = 0 if len(x.size()) == 1 else 1
            return [nn.functional.softmax(y) for y in torch.split(x, self.out_sizes, dim=d)]
        elif self.output == 'beta':
            pass
        elif self.output == 'gaussian':
            pass
        else:
            print("Error: DNN output must be \'linear\' or \'softmax\' or \'beta\' or \'gaussian\'")


# Critic buffer module
class CriticBuffer:

    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", 
                                        field_names=["time",
                                                    "state",
                                                    "joint_action",
                                                    "reward",
                                                    "next_state",
                                                    "gamma",
                                                    "done"])     

    def add(self, time, state, joint_action, reward, next_state, gamma, done):
        
        e = self.experiences(time, state, joint_action, reward, next_state, gamma, done)
        self.memory.append(e)
        
    def sample(self, sample_all=False):

        if len(self.memory) == 0:
            return None
        elif sample_all or len(self.memory) < self.batch_size:
            experiences = self.memory
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        times = tt([e.time for e in experiences if e is not None]).view(-1,1).float().to(device)
        states = torch.stack([e.state for e in experiences if e is not None]).float().to(device)
        joint_actions = [tt([e.joint_action[i] for e in experiences if e is not None]).view(-1,1).float().to(device) for i in range(len(experiences[0].joint_action))]
        rewards = tt([e.reward for e in experiences if e is not None]).view(-1,1).float().to(device)
        next_states = torch.stack([e.next_state for e in experiences if e is not None]).float().to(device)
        gammas = tt([e.gamma for e in experiences if e is not None]).view(-1,1).float().to(device)
        dones = tt([int(e.done) for e in experiences if e is not None]).view(-1,1).float().to(device)
        
        return (times, states, joint_actions, rewards, next_states, gammas, dones)
    
    def clear(self):

        for m in self.memory:
            m.clear()
    
    def __len__(self):
        return len(self.memory)


# Actor buffer module
class ActorBuffer:

    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", 
                                        field_names=["time",
                                                    "state",
                                                    "joint_action",
                                                    "reward",
                                                    "next_state",
                                                    "discount",
                                                    "gamma",
                                                    "done",
                                                    "true_state"])     

    def add(self, time, state, joint_action, reward, next_state, discount, gamma, done, true_state):
        
        e = self.experiences(time, state, joint_action, reward, next_state, discount, gamma, done, true_state)
        self.memory.append(e)
        
    def sample(self, sample_all=False):

        if len(self.memory) == 0:
            return None
        elif sample_all or len(self.memory) < self.batch_size:
            experiences = self.memory
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        times = tt([e.time for e in experiences if e is not None]).view(-1,1).float().to(device)
        states = torch.stack([e.state for e in experiences if e is not None]).float().to(device)
        joint_actions = [tt([e.joint_action[i] for e in experiences if e is not None]).view(-1,1).float().to(device) for i in range(len(experiences[0].joint_action))]
        rewards = [tt([e.reward[j] for e in experiences if e is not None]).view(-1,1).float().to(device) for j in range(len(experiences[0].reward))]
        next_states = torch.stack([e.next_state for e in experiences if e is not None]).float().to(device)
        gammas = [tt([e.gamma[j] for e in experiences if e is not None]).view(-1,1).float().to(device) for j in range(len(experiences[0].gamma))]
        discounts = [tt([e.discount[j] for e in experiences if e is not None]).view(-1,1).float().to(device) for j in range(len(experiences[0].discount))]
        dones = tt([int(e.done) for e in experiences if e is not None]).view(-1,1).float().to(device)
        true_states = torch.stack([e.true_state for e in experiences if e is not None]).float().to(device)

        return (times, states, joint_actions, rewards, next_states, gammas, discounts, dones, true_states)
    
    def clear(self):

        for m in self.memory:
            m.clear()
    
    def __len__(self):
        return len(self.memory)


# Set up actor functions 
def set_up_actors(parametrization, obs_size, act_sizes, hidden):

    if parametrization == 'dnn':
        actors = []
        for i in range(len(act_sizes)):
            actor = DNN(obs_size, [act_sizes[i]], 'softmax', hidden)
            actors.append(actor)

    elif parametrization == 'linear':
        actors = []
        for i in range(len(act_sizes)):
            actor = DNN(obs_size, [act_sizes[i]], 'softmax', [], bias=False)
            actors.append(actor)

    else:
        print("Error: Actor must be \'dnn\' or \'linear\')")

    for actor in actors:
        actor.to(device)

    return actors


# Set up critic functions 
def set_up_critics(parametrization, obs_size, num_specs, hidden):

    if parametrization == 'dnn':
        critics = [DNN(obs_size, [1], 'dnn', hidden) for j in range(num_specs)]

    elif parametrization == 'linear':
        critics = [DNN(obs_size, [1], 'linear', [], bias=False) for j in range(num_specs)]

    else:
        print("Error: Critic must be \'dnn\' or \'linear\')")

    for critic in critics:
        critic.to(device)

    return critics


# Set up optimisers for given networks
def set_up_optimisers(networks, optimiser, l2=0.0):

    if optimiser == 'adam':
        optimisers = [optim.Adam(n.parameters(), lr=1.0, weight_decay=l2) for n in networks]
    elif optimiser == 'sgd':
        optimisers = [optim.SGD(n.parameters(), lr=1.0, weight_decay=l2) for n in networks]
    elif optimiser == 'rmsprop':
        optimisers = [optim.RMSprop(n.parameters(), lr=1.0, weight_decay=l2) for n in networks] 
    else:
        print("Error: Optimiser must be \'adam\' or \'sgd\' or \'rmsprop\')")

    return optimisers


# Set up learning rates
def set_up_lrs(learning_rates):

    lrs = dict()

    for k in learning_rates.keys():
        (method, param) = learning_rates[k]
        if method == 'robbinsmonro':
            if not (0.5 < param) and (param < 1.0):
                print("Error: Robbins-Monro decay parameter p must satisfy 0.5 < p < 1.0")
                return
            lr_lambda = lambda e: e ** (-param)
        elif method == 'exponential':
            if not (0.0 < param) and (param < 1.0):
                print("Error: Exponential decay parameter p must satisfy 0.0 < p < 1.0")
                return
            lr_lambda = lambda e: param ** e
        elif method == 'constant':
            if not (0.0 < param):
                print("Error: Contant decay parameter p must satisfy 0.0 < p")
                return
            lr_lambda = lambda e: param
        else:
            print("Error: Learning rate must be \'robbinsmonro\' or \'exponential\' or \'constant\'")

        lrs[k] = lr_lambda    

    return lrs


# Almanac module
class Almanac:

    def __init__(self, obs_size, act_sizes, num_rewards, num_objectives, hps):

        # Parameters
        self.act_sizes = act_sizes
        self.hps = hps
        self.lrs = set_up_lrs(self.hps['lrs'])
        self.num_objectives = num_objectives
        self.num_players = len(self.act_sizes)
        self.num_rewards = num_rewards
        self.obs_size = obs_size
        self.gammas = self.num_rewards * [1.0]
        self.u = [0.0 for _ in range(num_objectives - 1)]
        self.recent_losses = [deque(maxlen=25) for i in range(num_objectives)]
       
        # Networks
        self.actors = set_up_actors(self.hps['models']['actor'][0], self.obs_size, self.act_sizes, hidden=self.hps['models']['actor'][1])
        self.critics = set_up_critics(self.hps['models']['critic'][0], self.obs_size, self.num_rewards, hidden=self.hps['models']['critic'][1])
        
        # Optimisers
        self.actor_optimisers = set_up_optimisers(self.actors, self.hps['optimisers']['actor'])
        self.critic_optimisers = set_up_optimisers(self.critics, self.hps['optimisers']['critic'])

        # Buffers
        self.critic_buffers = [CriticBuffer(buffer_size=self.hps['buffers']['critic_size'], batch_size=self.hps['buffers']['critic_batch']) for c in self.critics]
        self.actor_buffer = ActorBuffer(buffer_size=self.hps['buffers']['actor_size'], batch_size=self.hps['buffers']['actor_batch'])

        # Lagrange multipliers
        self.lagrange_multipliers = (self.num_objectives - 1) * [tt(0.0)]
        
    def update_critic(self, j, e, data, continue_prob, until_converged=True, num_updates=None):

        times, states, _, rewards, next_states, gammas, dones = data
        self.critics[j].train()

        # Form state dists for weighting samples
        dist = self.state_dist(continue_prob, times, gammas).to(device)

        # Compute target
        with torch.no_grad():
            target = rewards.to(device) + (self.discounts[j] * self.critics[j](next_states.to(device)) * (1-dones)).to(device)

        # Update networks
        finished = False
        temp_recent_losses = deque(max_len=25)
        if num_updates == None:
            num_updates = len(times)
        e = 0
        while not finished:

            # Form loss
            prediction = self.critics[j](states.to(device)).to(device)
            loss = self.lrs["alpha"](e) * ((dist * (prediction - target)**2).mean()).to(device)

            # Backpropagate
            self.critic_optimisers[j].zero_grad()
            loss.backward()
            self.critic_optimisers[j].step()
            self.critics[j].eval()

            # Check whether to finish updating
            e += 1
            if (not until_converged and e > num_updates) or utils.losses_converged(temp_recent_losses):
                finished = True
     
    def update_actors(self, data, e, continue_prob, objectives, until_converged=True, num_updates=None):

        times, states, joint_actions, rewards, next_states, gammas, discounts, dones, true_states = data
        for i in range(self.num_players):
            self.actors[i].train()

        # Form state dists for weighting samples
        dists = [self.state_dist(continue_prob, times, gamma).to(device) for gamma in gammas]
        obj_dists = [sum([o[j] * dists[j] for j in range(self.num_rewards)]) for o in objectives]

        with torch.no_grad():

            # Form advantages
            advantages = [rewards[j].to(device) + (discounts[j] * self.critics[j](next_states.to(device)) * (1-dones)).to(device) - self.critics[j](states.to(device)) for j in range(self.num_rewards)]
            obj_advantages = [sum([o[j] * advantages[j] for j in range(self.num_rewards)]) for o in objectives]

            # Form old probability factors
            action_dists = self.policy(true_states)
            old_log_probs = sum([action_dists[i].log_prob(joint_actions[i]) for i in range(self.num_players)])
        
        # Update networks
        finished = False
        temp_recent_losses = deque(max_len=25)
        if num_updates == None:
            num_updates = len(times)
        e = 0
        while not finished:

            # Form weights
            objective_range = self.k + 1 if self.k != None else self.num_objectives
            objective_weights =  self.compute_weights(objective_range)
            relative_kl_weights = [self.kl_weight * objective_weights[i] / sum(objective_weights) for i in range()] + [0.0 for _ in range(objective_range, self.num_objectives)]

            # Form loss
            new_log_probs = sum([action_dists[i].log_prob(joint_actions[i]) for i in range(self.num_players)])
            kl_penalty = (new_log_probs - old_log_probs).to(device)
            ratio = torch.exp(kl_penalty)
            loss = sum([(objective_weights[k] * obj_dists[k] * (ratio * obj_advantages[k] - self.kl_weight * kl_penalty) for k in range(len(objective_weights))).mean()])

            # Backpropagate
            for i in range(self.num_players):
                self.actor_optimisers[i].zero_grad()
                loss.backward()
                self.actor_optimisers[i].step()
                self.actors[i].eval()

            # Update KL weight term as in the original PPO paper
            if kl_penalty.mean() < self.hps["kl_target"] / 1.5:
                self.kl_weight *= 0.5
            else:
                self.kl_weight *= 2

            # Update Lagrange multipliers
            for k in range(self.num_objectives):
                self.recent_losses[k].append((ratio * obj_advantages[k] - relative_kl_weights[k] * kl_penalty).mean())
            self.update_lagrange_multipliers(e)

            # Check whether to finish updating
            e += 1
            self.recent_actor_losses.append(loss)
            if until_converged:
                if self.k == None:
                    finished = utils.losses_converged(temp_recent_losses)
                elif self.k == self.num_objectives - 1 and utils.losses_converged(self.recent_losses[self.k]):
                    finished = True
            elif e > num_updates:
                finished = True

    # Udpate Lagrange multipliers
    def update_lagrange_multipliers(self, e, max_lm=100.0):
        
        # Save relevant loss information for updating Lagrange parameters
        if self.k != None:
            if not utils.converged(self.recent_losses[self.k]):
                if self.k != self.num_objectives - 1:
                    self.u[self.k] = -torch.tensor(self.recent_losses[self.k]).mean()
            else:
                self.k = 0 if self.k == self.num_objectives - 1 else self.k + 1
        else:
            for i in range(self.num_objectives - 1):
                self.u[i] = -torch.tensor(self.recent_losses[i]).mean()

        # Update Lagrange parameters
        r = self.k if self.k != None else self.num_objectives - 1
        for i in range(r):
            self.lagrange_multipliers[i] += self.lrs["eta"][i](e) * (self.u[i] - self.recent_losses[i][-1])
            self.lagrange_multipliers[i] = min(max(self.lagrange_multipliers[i], 0.0), max_lm)

    # Compute weights for lexicographic objectives
    def compute_weights(self, objective_range):

        objective_weights = []
        for k in range(objective_range - 1):
            if self.k != None:
                w = self.lrs["beta"][objective_range - 1] * self.lagrange_multipliers[k]
            else:
                w = self.lrs["beta"][k] + self.lagrange_multipliers[k] * sum([self.lrs["beta"][kk] for kk in range(k+1, objective_range)])
            objective_weights.append(w)
        objective_weights.append(self.lrs["beta"][objective_range - 1])

        return objective_weights

    # Output policy distribution at a particular state
    def policy(self, state, probs=False):

        action_dists = [actor(state) for actor in self.actors]
        if probs:
            return [a[0] for a in action_dists]
        else:
            return [Categorical(a[0]) for a in action_dists]

    # Perform (epsilon-greedy) joint action
    def act(self, state, epsilon=0.0):

        if random.random() < epsilon:
            joint_action = [tt(random.choice(range(act_size))) for act_size in self.act_sizes]
        else:
            joint_action = [actions.sample() for actions in self.policy(state)]

        return joint_action

    # Take a learning step
    def step(self, info, augment_data, continue_prob):

        # Reset Gamma terms if needed
        if info["t"] == 0:
            self.gammas = self.num_rewards * [1.0]

        # Form product states
        prod_state = torch.cat([info["f(s)"]] + info["f(q)"], 0)
        new_prod_state = torch.cat([info["f(s')"]] + info["f(q')"], 0)

        # Form scalar rewards
        if info["is_e_t"]:
            scalar_rewards = [0.0 for _ in info["R"]]
            scalar_discounts = [1.0 for _ in info["R"]]
        else:
            scalar_rewards = [r_f(info["s"], info["a"], info["s'"]) for r_f in info["R"]]
            scalar_discounts = info["D"]

        for j in range(self.num_critics):

            # Temporarily save product state
            self.Z[j][tuple(prod_state.tolist())] = (info["t"], prod_state, info["a"])
        
            # Perform patient updates if possible
            add_stored_experiences = False
            if (j < self.num_specs and info["F"][j]):
                self.gammas[j] *= self.hps['gamma_Phi']
                r = self.hps['spec_reward']
                add_stored_experiences = True
            elif (j >= self.num_specs and not info["is_e_t"]):
                self.gammas[j] *= info["D"][j]
                r = info["R"][j - self.num_specs](info["s"], info["a"], info["s'"])
                add_stored_experiences = True
            elif self.critics[j](new_prod_state) == 0.0:
                r = 0.0
                add_stored_experiences = True
            if add_stored_experiences:
                for k in self.Z[j].keys():
                    (t, s, a) = self.Z[j][k]
                    self.critic_buffers[j].add(t, s, a, r, new_prod_state, self.gammas[j], info["d"])
                self.Z[j] = dict()

        # Store (augmented) data for actors
        for q_1, q_2, f_s in info["f(q)_s"], info["f(q')_s"], info["F_s"]:
            s_q_1 = torch.cat([info["f(s)"]] + q_1, 0)
            s_q_2 = torch.cat([info["f(s')"]] + q_2, 0)
            r = [self.hps['spec_reward'] * f for f in f_s] + scalar_rewards
            d_s = [1 - f * (1 - self.hps['gamma_Phi']) for f in f_s] + scalar_discounts
            self.actor_buffer.add(info["t"], s_q_1, info["a"], r, s_q_2, d_s, self.gammas, info["d"], prod_state)

            # Store augmented data for critics when possible
            if augment_data:
                # Avoid oversampling the true transition
                if torch.eq(s_q_1, prod_state).all():
                    continue
                # Add data for alternative transitions
                else:
                    for j in range(self.num_critics):
                        add_experience = False
                        if (j < self.num_specs and info["F"][j]):
                            r_j = self.hps['spec_reward']
                            add_experience = True
                        elif (j >= self.num_specs and not info["is_e_t"]):
                            r_j = scalar_rewards[j]
                            add_experience = True
                        elif self.critics[j](s_q_2) == 0.0:
                            r_j = 0.0
                            add_experience = True
                        if add_experience:
                            self.critic_buffers[j].add(info["t"], s_q_1, info["a"], r_j, s_q_2, self.gammas[j], info["d"])
                        
        # If enough data has been collected update the critics
        for j in range(self.num_critics):
            if len(self.critic_buffers[j]) > self.hps['update_after']['critics']:
                data = self.critic_buffers[j].sample(sample_all=True)
                self.update_critic(j, self.update["critic"][j], data, continue_prob)
                self.critic_buffers[j].clear()
        
        # If enough data has been collected update the actors and the Lagrange multipliers
        if len(self.actor_buffer) > self.hps['update_after']['actors']:
            data = self.actor_buffer.sample(sample_all=True)
            self.update_actors(data, self.update["actors"])
            self.actor_buffer.clear()











    def train(self, steps, env, specs, actual_dist, patient_updates, train_constants, run_id, filename, score_interval=100):
        
        best = 0.0
        last_score_interval = deque(maxlen=score_interval)

        if self.env_kind == 'mmg':
            first_50 = []
            on2awinner = False

        if filename == None:
            filename = 'scores/{}/almanac-{}-{}.txt'.format(self.env_kind, self.env_name, run_id)

        reward_multiplier = train_constants['reward_weight']
        tolerance = train_constants['nat_grad_tolerance']
        neg_ent_reg = train_constants['neg_ent_reg']
        non_det_reg = train_constants['non_det_reg']
        sum_val_reg = train_constants['sum_val_reg']
        c_neg_var_reg = train_constants['c_neg_var_reg']
        a_neg_var_reg = train_constants['a_neg_var_reg']
        continue_prob = train_constants['continue_prob']
        epsilon = train_constants['epsilon']
        max_nat_grad_norm = train_constants['max_nat_grad_norm']
        max_critic_norm = train_constants['max_critic_norm']

        # Check for dumb mistakes
        if env.get_name() != self.env_name:
            print("Error: Environment mismatch")
            return

        if run_id == None:
            run_id = time.strftime("%Y%m%d-%H%M%S")

        s = 1
        game_score = 0.0
        e = -1
        recent_game_score = 0.0

        # Run episodes
        while s < steps:
            
            # Initialise environment
            game_state = env.reset()
            spec_states_rewards = [(spec.ldba.reset(), 0.0) for spec in self.specs]
            done = False
            t = 0
            gammas = [1.0 for j in range(self.num_specs)]
            Z = [dict() for j in range(self.num_specs)]
            e += 1
            # print("End of episode, resetting...")

            while not random.random() > continue_prob and not done:

                # Form state vectors 
                spec_state_vectors = [one_hot(tt(spec_states_rewards[j][0]), self.specs[j].ldba.get_num_states()) for j in range(self.num_specs)]
                prod_state = torch.cat([env.featurise(game_state)] + spec_state_vectors, 0)
                
                # Perform joint action
                if random.random() < epsilon(e):
                    joint_action = [tt(random.choice(range(act_size))) for act_size in self.act_sizes]
                else:
                    joint_action = [actions.sample() for actions in self.policy(prod_state)]

                # If an epsilon transition is made the game (and possibly spec) state remains the same 
                is_e_t, j, e_t = self.is_epsilon_transition(joint_action)
                if is_e_t:
                    new_game_state, done = game_state, done

                    # If the chosen epsilon transition is available in this state then it is made
                    if e_t != None:
                        spec_states_rewards[j] = self.specs[j].ldba.step(None, epsilon=e_t)
                        label_set = env.label(new_game_state)
                        for k in range(len(self.specs)):
                            if k != j:
                                spec_states_rewards[k] = self.specs[k].ldba.step(label_set)
                        new_spec_states_rewards = spec_states_rewards
                    else:
                        label_set = env.label(new_game_state)
                        new_spec_states_rewards = [spec.ldba.step(label_set) for spec in self.specs]

                # Otherwise a standard action is performed in the MG
                else:
                    new_game_state, done = env.step(joint_action)
                    label_set = env.label(new_game_state)
                    new_spec_states_rewards = [spec.ldba.step(label_set) for spec in self.specs]
                
                # Temporarily save new product state
                if patient_updates:
                    for j in range(self.num_specs): 
                        Z[j][tuple(prod_state.tolist())] = (t, prod_state, joint_action)

                # Compute rewards etc. from environment and automata
                unweighted_rewards = [reward_multiplier * s_r[1] for spec, s_r in zip(self.specs, new_spec_states_rewards)]
                discounts = [self.discounts['patient'] if r > 0.0 else 1.0 for r in unweighted_rewards]
                gammas = [g * d for g, d in zip(gammas, discounts)]
                new_spec_state_vectors = [one_hot(tt(new_spec_states_rewards[j][0]), self.specs[j].ldba.get_num_states()) for j in range(self.num_specs)]
                new_prod_state = torch.cat([env.featurise(new_game_state)] + new_spec_state_vectors, 0)

                for j in range(self.num_specs):
                    
                    # Store patient critic experience when possible
                    if patient_updates:
                        if unweighted_rewards[j] > 0.0 or self.critics[j](new_prod_state) == 0.0:
                            for k in Z[j].keys():
                                (p_t, p_s, p_a) = Z[j][k]
                                self.patient_buffer.add(j, p_t, p_s, p_a, unweighted_rewards[j], new_prod_state, gammas[j], done)
                            Z[j] = dict()
                    # else:
                    #     self.patient_buffer.add(j, t, prod_state, joint_action, unweighted_rewards[j], new_prod_state, gammas[j], done)

                    # Store hasty critic experience
                    self.hasty_buffer.add(j, t, prod_state, joint_action, unweighted_rewards[j], new_prod_state, gammas[j], done)
                    
                # Update variables for next step
                game_state = new_game_state
                spec_states_rewards = new_spec_states_rewards

                game_score += sum([r * spec.weight for r, spec in zip(unweighted_rewards, self.specs)])
                recent_game_score += sum([r * spec.weight for r, spec in zip(unweighted_rewards, self.specs)])
                t += 1
                s += 1

                # Save and occasionally print (average) score
                if s % score_interval == 0:
                    
                    average_game_score = game_score / s

                    print("Score ({}/{}): {} (recent)   {} (average)".format(int(s / score_interval), int(steps / score_interval), (recent_game_score / score_interval), average_game_score))

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

            # If episode has ended use stored data for learning
            hasty_data = [self.hasty_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
            if patient_updates:
                patient_data = [self.patient_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
                data = patient_data + hasty_data
            else:
                patient_data = hasty_data
                data = hasty_data

            # Compute new score functions
            for d in data:
                if d != None:
                    for i in range(len(self.actors)):
                        data_state_actions = torch.unique(torch.cat((d[1],d[2][i]), dim=1), dim=0)
                        keys = [tuple(tens.tolist()) for tens in data_state_actions]
                        new_pairs = [(k, tens) for k, tens in zip(keys, data_state_actions) if k not in self.scores[i].keys()]
                        self.scores[i].update(self.get_scores(i, new_pairs))
                    
            # Update critics and natural gradients
            self.update_critics(e, patient_data, hasty_data, continue_prob, sum_val_reg, c_neg_var_reg, max_critic_norm, actual_dist)
            self.update_nat_grads(e, hasty_data, continue_prob, max_nat_grad_norm, actual_dist)
            self.update_lam(e)

            # If natural gradients have converged, update the policy and clear buffers and old scores
            for i in range(len(self.actors)):
                if self.nat_grads_converged(i, tolerance):                
                    # print("Natural gradient {} converged!".format(i))
                    self.update_actors(i, e, neg_ent_reg, non_det_reg, a_neg_var_reg)
                    self.patient_buffer.clear()
                    self.hasty_buffer.clear()
                    self.scores[i].clear()

                    #Additional Info when using cuda
                    # if device.type == 'cuda':
                    #     print(torch.cuda.get_device_name(0))
                    #     print('Memory Usage:')
                    #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,5), 'MB')
                    #     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**2,5), 'MB')

        return True


    def state_dist(self, continue_prob, ts, gammas):

        return power(continue_prob * ones_like(ts), -ts) * gammas if self.actual_dist else power(continue_prob * ones_like(ts), -ts)

    def save(self):

        pass

    def get_policy_dists(self, possible_states, possible_state_tensors):

        return dict([(p, self.policy(t, probs=True)) for p,t in zip(possible_states, possible_state_tensors)])