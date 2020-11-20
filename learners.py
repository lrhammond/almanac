import torch
import torch.tensor as tt
import random
from torch import pow as power
import time
from collections import namedtuple, deque
import torch.nn as nn
from torch import optim, ones_like
from torch.distributions import Categorical
# import numpy as np
from torch.nn.functional import one_hot as one_hot
import cProfile
import itertools
import numpy as np

# import learners
import specs
import envs
from os import path
import pickle
import utils





if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

### BASE ELEMENTS ###


class DNN(nn.Module):

    def __init__(self, in_size, out_sizes, output, hidden, bias=True):

        super(DNN,self).__init__()

        self.dims = [in_size] + hidden + [sum(out_sizes)]
        self.output = output
        self.out_sizes = out_sizes
        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1], bias=bias))

    # @profile 
    def forward(self, x):

        # Apply to ReLU activations to all layers apart from the last
        # x = x.view(x.size(0), -1)
        x = x.float()
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))

        # Compute last layer
        x = x.float()
        x = self.layers[-1](x)

        
        # Return output of DNN as a tuple
        if self.output == 'linear':
            # return torch.split(x, self.out_sizes, dim=1)
            return x
        elif self.output == 'softmax':
            d = 0 if len(x.size()) == 1 else 1
            return [nn.functional.softmax(y) for y in torch.split(x, self.out_sizes, dim=d)]
        else:
            print("Error: DNN output must be \'linear\' or \'softmax\'")
        
        # if type(self.out_size) == int:
        #     x = x.view(x.size(0),self.out_size)
        # else:
        #     x = x.view((x.size(0),) + self.out_size)
        
        # return x


class NaturalGradient(nn.Module):

    def __init__(self, param_sizes):

        super(NaturalGradient,self).__init__()

        self.xs = nn.ParameterList([nn.Parameter(torch.zeros(p, requires_grad=True, dtype=torch.float)) for p in param_sizes])

        for p in self.parameters():
            p.requires_grad = True
        
    def forward(self, scores):

        flat_xs = torch.cat([torch.flatten(x) for x in self.xs])

        return torch.sum(flat_xs * scores, dim=1)

        # predictions = []

        # for score in scores:
        #     predictions.append(sum([torch.sum(x * p.to(device)).to(device) for x, p in zip(self.xs, score)]).to(device))

        # return torch.stack(predictions).to(device)


class ReplayBuffer:

    def __init__(self, num_specs, buffer_size, batch_size):

        self.memory = [deque(maxlen=buffer_size) for j in range(num_specs)]
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", 
                                        field_names=["time",
                                                    "state",
                                                    "joint_action",
                                                    "reward",
                                                    "next_state",
                                                    "gamma",
                                                    "done"])     

    def add(self, j, time, state, joint_action, reward, next_state, gamma, done):
        
        e = self.experiences(time, state, joint_action, reward, next_state, gamma, done)
        self.memory[j].append(e)
        
    def sample(self, j, sample_all=False):

        if len(self.memory[j]) == 0:
            return None
        elif sample_all or len(self.memory[j]) < self.batch_size:
            experiences = self.memory[j]
        else:
            experiences = random.sample(self.memory[j], k=self.batch_size)
        
        times = tt([e.time for e in experiences if e is not None]).view(-1,1).float().to(device)
        # times = torch.cat([e.time for e in experiences if e is not None], dim=0).float()

        states = torch.stack([e.state for e in experiences if e is not None]).float().to(device)
        # torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().to(device)

        joint_actions = [tt([e.joint_action[i] for e in experiences if e is not None]).view(-1,1).float().to(device) for i in range(len(experiences[0].joint_action))]
        # joint_actions = [torch.from_numpy(np.vstack([e.joint_action[i] for e in experiences if e is not None])).float().to(device) for i in range(len(experiences[0].joint_action))]
       
        rewards = tt([e.reward for e in experiences if e is not None]).view(-1,1).float().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        next_states = torch.stack([e.next_state for e in experiences if e is not None]).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state.cpu() for e in experiences if e is not None])).float().to(device)

        gammas = tt([e.gamma for e in experiences if e is not None]).view(-1,1).float().to(device)
        # gammas = torch.from_numpy(np.vstack([e.gamma for e in experiences if e is not None])).float().to(device)

        dones = tt([int(e.done) for e in experiences if e is not None]).view(-1,1).float().to(device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (times, states, joint_actions, rewards, next_states, gammas, dones)
    
    def clear(self):

        for m in self.memory:
            m.clear()
    
    def __len__(self, j):
        return len(self.memory[j])


### SETUP FUNCTIONS ###


def set_up_actors(parametrization, local, obs_size, act_sizes, hidden):

    if parametrization == 'dnn':
        if local:
            actors = []
            for i in range(len(act_sizes)):
                actor = DNN(obs_size, [act_sizes[i]], 'softmax', hidden)
                actors.append(actor)
        else:
            actors = [DNN(obs_size, act_sizes, 'softmax', hidden)]

    elif parametrization == 'linear':
        if local:
            actors = []
            for i in range(len(act_sizes)):
                actor = DNN(obs_size, [act_sizes[i]], 'softmax', [], bias=False)
                actors.append(actor)
        else:
            actors = [DNN(obs_size, act_sizes, 'softmax', [], bias=False)]

    else:
        print("Error: Actor must be \'dnn\' or \'linear\')")

    for actor in actors:
        actor.to(device)

    return actors


def set_up_critics(parametrization, obs_size, num_specs, hidden):

    if parametrization == 'dnn':
        critics = [DNN(obs_size, [1], 'linear', hidden) for j in range(num_specs)]

    elif parametrization == 'linear':
        critics = [DNN(obs_size, [1], 'linear', [], bias=False) for j in range(num_specs)]

    else:
        print("Error: Critic must be \'dnn\' or \'linear\')")


    for critic in critics:
        critic.to(device)

    return critics


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
        # elif method == 'onplateau_min':
        #     if not (0.0 < param) and (param < 1.0):
        #         print("Error: Plateau decay parameter p must satisfy 0.0 < p < 1.0")
        #         return
        #     schedulers = [optim.lr_scheduler.ReduceLROnPlateau(o, mode='min', factor=param, patience=10, threshold=0.001) for o in optimisers]
        # elif method == 'onplateau_max':
        #     if not (0.0 < param) and (param < 1.0):
        #         print("Error: Plateau decay parameter p must satisfy 0.0 < p < 1.0")
        #         return
        #     schedulers = [optim.lr_scheduler.ReduceLROnPlateau(o, mode='max', factor=param, patience=10, threshold=0.001) for o in optimisers]
        else:
            print("Error: Learning rate must be \'robbinsmonro\' or \'exponential\' or \'constant\'")

        lrs[k] = lr_lambda    

    return lrs


### ALMANAC ###


class Almanac:

    def __init__(self, env, specifications, optimisers, buffers, models, local, model_constants, model_id):

        # Parameters
        self.obs_size = env.get_obs_size()
        self.act_sizes = env.get_act_sizes()
        self.lrs = set_up_lrs(model_constants['lrs'])
        self.discounts = model_constants['discounts']
        self.num_specs = len(specifications)
        self.num_players = len(self.act_sizes)
        self.local = local
        self.models = models
        self.buffers = buffers
        self.env_name = env.get_name()
        self.env_kind = env.get_kind()
        self.model_id = random.randint(0,int(1e6)) if model_id == None else model_id
        self.l2_reg = model_constants['l2_reg']

        # Automata
        self.specs = specifications
        self.ldba_state_sizes = [spec.ldba.get_num_states() for spec in self.specs]
        self.epsilon_act_sizes = [spec.ldba.get_num_eps_actions() for spec in self.specs]
       
        # Networks
        self.actors = set_up_actors(self.models['actor'][0], local, self.obs_size + sum(self.ldba_state_sizes), [a_s + sum(self.epsilon_act_sizes) for a_s in self.act_sizes], hidden=self.models['actor'][1])
        self.patient_critics = set_up_critics(self.models['critic'][0], self.obs_size + sum(self.ldba_state_sizes), self.num_specs, hidden=self.models['critic'][1])
        self.hasty_critics = set_up_critics(self.models['critic'][0], self.obs_size + sum(self.ldba_state_sizes), self.num_specs, hidden=self.models['critic'][1])

        # Natural gradients
        self.param_shapes = []
        for actor in self.actors:
            model_params = filter(lambda p: p.requires_grad, actor.parameters())
            self.param_shapes.append(tuple([p.shape for p in model_params]))
        self.nat_grads = [NaturalGradient(params) for params in self.param_shapes]
        for nat_grad in self.nat_grads:
            nat_grad.to(device)
        self.scores = [dict() for i in range(len(self.actors))]
        self.current_patient_losses = [deque(maxlen=10) for a in self.nat_grads]
        self.current_hasty_losses = [deque(maxlen=10) for a in self.nat_grads]
        
        # Optimisers
        self.actor_optimisers = set_up_optimisers(self.actors, optimisers['actor'])
        self.patient_critic_optimisers = set_up_optimisers(self.patient_critics, optimisers['critic'], l2=self.l2_reg)
        self.hasty_critic_optimisers = set_up_optimisers(self.hasty_critics, optimisers['critic'], l2=self.l2_reg)
        self.nat_grad_optimisers = set_up_optimisers(self.nat_grads, optimisers['nat_grad'], l2=self.l2_reg)

        # Replay buffers
        self.patient_buffer = ReplayBuffer(self.num_specs, buffer_size=self.buffers['size'], batch_size=self.buffers['batch'])
        self.hasty_buffer = ReplayBuffer(self.num_specs, buffer_size=self.buffers['size'], batch_size=self.buffers['batch'])

        # Lagrange multipliers
        self.mu = [tt(0.0) for a in self.actors]

        possible_game_states = list(itertools.product([0, 1], repeat=self.obs_size))
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


    def update_critics(self, e, patient_experiences, hasty_experiences, continue_prob, sum_val_reg, c_neg_var_reg, max_critic_norm, actual_dist):
        
        if len(self.scores[0]) != 0:
            o = self.obs_size + sum(self.ldba_state_sizes)
            unique_states = torch.stack([tt(state)[0:o] for state in self.scores[0].keys()]).to(device)

        for j in range(self.num_specs):

            if patient_experiences[j] != None:

                ts, states, _, rewards, next_states, gammas, dones = patient_experiences[j]
                self.patient_critics[j].train()

                # Form state dists and state-dependent discounts
                dist = self.state_dist('patient', actual_dist, continue_prob, ts, gammas).to(device)
                patient_discounts = torch.ones_like(rewards).to(device) - ((1.0 - self.discounts['patient']) / utils.denom(torch.max(rewards))) * rewards.to(device)

                # Compute losses
                patient_prediction = self.patient_critics[j](states.to(device)).to(device)
                with torch.no_grad():
                    patient_target = rewards.to(device) + (patient_discounts * self.patient_critics[j](next_states.to(device)) * (1-dones)).to(device)

                values = self.patient_critics[j](unique_states)
                t_v_r = (sum_val_reg * torch.sum(torch.square(values))).to(device)
                if len(unique_states) == 1:
                    patient_loss = self.lrs['critic'](e) * (t_v_r + (dist * (patient_prediction - patient_target)**2).mean()).to(device)
                else:
                    n_v_r = -(c_neg_var_reg * torch.var(values))
                    patient_loss = self.lrs['critic'](e) * (t_v_r + n_v_r + (dist * (patient_prediction - patient_target)**2).mean()).to(device)
                
                # PRINTER
                # print("Patient_Loss:", float(patient_loss))

                # Update networks
                self.patient_critic_optimisers[j].zero_grad()
                patient_loss.backward()

                nn.utils.clip_grad_norm_(self.patient_critics[j].parameters(), max_critic_norm)

                self.patient_critic_optimisers[j].step()
                self.patient_critics[j].eval()


            if hasty_experiences[j] != None:

                ts, states, _, rewards, next_states, _, dones = hasty_experiences[j]
                self.hasty_critics[j].train()

                # Form state dists and state-dependent discounts
                dist = self.state_dist('hasty', actual_dist, continue_prob, ts, None).to(device)

                # Compute losses
                hasty_prediction = self.hasty_critics[j](states.to(device)).to(device)
                with torch.no_grad():
                    hasty_target = rewards.to(device) + (self.discounts['hasty'] * self.hasty_critics[j](next_states.to(device)) * (1-dones)).to(device)

                values = self.hasty_critics[j](unique_states)
                t_v_r = (sum_val_reg * torch.sum(torch.abs(values))).to(device)
                if len(unique_states) == 1:
                    hasty_loss = self.lrs['critic'](e) * (t_v_r + (dist * (hasty_prediction - hasty_target)**2).mean()).to(device)
                else:
                    n_v_r = -(c_neg_var_reg * torch.var(values))
                    hasty_loss = self.lrs['critic'](e) * (t_v_r + n_v_r + (dist * (hasty_prediction - hasty_target)**2).mean()).to(device)
                
                # PRINTER
                # print("Hasty_Loss:", float(hasty_loss))

                # Update networks
                self.hasty_critic_optimisers[j].zero_grad()
                hasty_loss.backward()

                nn.utils.clip_grad_norm_(self.hasty_critics[j].parameters(), max_critic_norm)

                self.hasty_critic_optimisers[j].step()
                self.hasty_critics[j].eval()

    # @profile
    def update_nat_grads(self, e, hasty_experiences, continue_prob, max_nat_grad_norm, actual_dist):

        # ts, states, joint_actions, rewards, next_states, gammas, dones = experiences

        # td_errors = 

        # actual_dist = False

        for i in range(len(self.nat_grads)):

            self.nat_grads[i].train()

            patient_losses = []
            hasty_losses = []

            for j in range(self.num_specs):

                if hasty_experiences[j] != None:

                    ts, states, joint_actions, rewards, next_states, gammas, dones = hasty_experiences[j]

                    # Form state dists and state-dependent discounts
                    patient_dist = self.state_dist('patient', actual_dist, continue_prob, ts, gammas).to(device)
                    hasty_dist = self.state_dist('hasty', actual_dist, continue_prob, ts, None).to(device)
                    patient_discounts = (torch.ones_like(rewards).to(device) - ((1.0 - self.discounts['patient']) / utils.denom(torch.max(rewards))) * rewards.to(device)).to(device)

                    # Retrieve saved scores
                    if self.local:
                        data_state_actions = torch.cat((states,joint_actions[i]), dim=1)
                    else:
                        data_state_actions = torch.cat((states,torch.cat(joint_actions, dim=1)), dim=1)
                    scores = torch.stack([self.scores[i][tuple(d_s_a.tolist())] for d_s_a in data_state_actions])

                    # scores = [torch.autograd.grad(outputs=l_p, inputs=self.actors[i].parameters(), retain_graph=True)[0] for l_p in self.policy(states.to(device))[i].log_prob(torch.transpose(joint_actions[i],0,1))[0]]

                    prediction = self.nat_grads[i](scores).to(device)

                    # print(prediction)

                    with torch.no_grad():
                        patient_target = rewards.to(device) + (patient_discounts * self.patient_critics[j](next_states.to(device)) * (1-dones)).to(device) - self.patient_critics[j](states.to(device)).to(device)
                        hasty_target = rewards.to(device) + (self.discounts['hasty'] * self.hasty_critics[j](next_states.to(device)) * (1-dones)).to(device) - self.hasty_critics[j](states.to(device)).to(device)

                    patient_losses.append((self.specs[j].weight * (patient_dist * torch.square(prediction - torch.flatten(patient_target))).mean()).to(device))
                    hasty_losses.append((self.specs[j].weight * (hasty_dist * torch.square(prediction - torch.flatten(hasty_target))).mean()).to(device))

            # Form losses
            l = tt(0.0) if len(self.current_patient_losses[i]) == 0 else tt(self.current_patient_losses[i]).float().mean()
            loss_1 = sum(patient_losses)
            loss_2 = sum(hasty_losses) + max(self.mu[i] * (loss_1 - l), tt(0))
            self.current_patient_losses[i].append(loss_1)
            self.current_hasty_losses[i].append(loss_2)
            loss = (self.lrs['patient_nat_grad'](e) * loss_1) + (self.lrs['hasty_nat_grad'](e) * loss_2).to(device)

            # if len(hasty_losses) != 0:
            #     loss = sum(hasty_losses).to(device)
            # else:
            #     loss = sum(hasty_losses)

            # self.current_patient_losses[i].append(loss)
            # self.current_hasty_losses[i].append(loss)

            # print(float(loss))

            # if len(hasty_losses) != 0:
            #     loss = (self.lrs['hasty_nat_grad'](e) * loss).to(device)
            # else:
            #     loss = (self.lrs['hasty_nat_grad'](e) * loss)

            
            # Update natural gradient
            if loss != 0.0:
                self.nat_grad_optimisers[i].zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.nat_grads[i].parameters(), max_nat_grad_norm)

                self.nat_grad_optimisers[i].step()

            self.nat_grads[i].eval()

            # PRINTER
            # print("Nat_Grad_Loss:", float(loss_1))

            # Clip weights to prevent exploding natural gradient updates
            # with torch.no_grad():
            #     for p in self.nat_grads[i].parameters():
            #         p = torch.clamp(p, -10.0, 10.0)
        
    def update_actors(self, i, e, neg_ent_reg, non_det_reg, a_neg_var_reg):

        # for i in range(len(self.actors)):

        self.actors[i].train()
        self.actor_optimisers[i].zero_grad()
        
        # with torch.no_grad():
        #     for p in self.nat_grads[i].parameters():
        #         print(p)
        # neg_entropy_loss = 0.0
        # regulariser_loss = 0.0
        if len(self.scores[i]) > 1:
            o = self.obs_size + sum(self.ldba_state_sizes)
            states = torch.stack([tt(state)[0:o] for state in self.scores[i].keys()])
            if self.local:
                action_dist = self.actors[i](states)[0]
                actual_mean_action_dist = action_dist.mean(dim=0)
                mean_action_dist = (1e-10 * torch.ones_like(actual_mean_action_dist)) + actual_mean_action_dist
                neg_entropy = torch.dot(mean_action_dist, torch.log(mean_action_dist))
                non_determinism = torch.abs((0.5 * torch.ones_like(action_dist)) - action_dist).mean()
                variance = torch.var(action_dist, dim=0).mean()
            else:
                # mean_action_dists = [torch.clamp(self.actors[0](states)[i][0].mean(dim=0), min=tt(1e-20)) for i in range(self.num_players)]
                joint_action_dists = self.actors[0](states)
                action_dists = [joint_action_dists[i] for i in range(self.num_players)]
                mean_action_dists = [1e-50 + joint_action_dists[i].mean(dim=0) for i in range(self.num_players)]
                neg_entropy = sum([torch.dot(a.to(device), torch.log(a).to(device)).to(device) for a in mean_action_dists]).to(device)
                non_determinism = sum([torch.square((0.5 * torch.ones_like(a_d)) - a_d).mean() for a in action_dists]).to(device)
                # TODO variance regularisation
            regulariser_loss = 0.1 * ((neg_ent_reg * neg_entropy) - (non_det_reg * non_determinism) - (a_neg_var_reg * variance))
            regulariser_loss.backward()

            # print(float(regulariser_loss))

        # Form normalised natural gradients
        with torch.no_grad():
            # xs = [- (self.lrs['actor'](e) * ones_like(p) * p) / utils.denom(torch.sqrt(torch.sum(ones_like(p) * p * p)))  for p in self.nat_grads[i].parameters()]
            xs = [(- self.lrs['actor'](e) * ones_like(p) * p).to(device) for p in self.nat_grads[i].parameters()]

            # print(xs)

        for p, x in zip(self.actors[i].parameters(), xs):
            if p.size() !=  x.size():
                print("Error: Mismatch between natural gradient and score dimensions!")
                return None
            # p.grad = torch.autograd.Variable(p.data.new(p.size()).zero_())

            if p.grad != None:
            # p.grad = utils.remove_nans(p.grad)
                p.grad = utils.remove_nans(p.grad + x).to(device)


            # if regulariser_loss != 0.0:
                
            # else:
            #     p.grad = utils.remove_nans(x).to(device)

            # print(p.grad)   

        self.actor_optimisers[i].step()
        self.actors[i].eval()

        # Clip policy weights to prevent exploding natural gradient updates
        # with torch.no_grad():
        #     for p in self.actors[i].parameters():
        #         p = torch.clamp(p, -10.0, 10.0)

    def update_mu(self, e, max_mu=100.0):

        with torch.no_grad():
            for i in range(len(self.nat_grads)):
                l = tt(0.0) if len(self.current_patient_losses[i]) == 0 else tt(self.current_patient_losses[i]).float().mean()
                self.mu[i] += self.lrs['mu'](e) * (self.current_patient_losses[i][-1] - l)
                self.mu[i] = min(max(self.mu[i], tt(0.0)), max_mu)

    # @ profile
    def policy(self, state, probs=False):

        if self.local: 
            action_dists = [actor(state) for actor in self.actors]
            if probs:
                return [a[0] for a in action_dists]
            else:
                return [Categorical(a[0]) for a in action_dists]
        else:
            joint_action_dists = self.actors[0](state)
            action_dists = [joint_action_dists[i] for i in range(self.num_players)]
            return [Categorical(a) for a in action_dists]

        

    # @profile
    def train(self, steps, env, specs, actual_dist, patient_updates, train_constants, run_id, filename):

        best = 0.0

        first_50 = []
        on2awinner = False
        last_100 = deque(maxlen=100)

        if filename == None:
            filename = 'results/scores/{}/almanac-{}-{}-{}.txt'.format(self.env_kind, self.env_name, self.model_id, run_id)

        
        possible_game_states = list(itertools.product([0, 1], repeat=self.obs_size))
        possible_spec_states = []
        for s_s in self.ldba_state_sizes:
            spec_states = [[0 for i in range(s_s)] for j in range(s_s)]
            for k in range(s_s):
                spec_states[k][k] = 1
            possible_spec_states.append([tuple(s_s) for s_s in spec_states])
        possible_product_states = itertools.product(possible_game_states, *possible_spec_states)
        possible_states = torch.stack([tt(tuple(utils.flatten(p_s))).float() for p_s in possible_product_states])





        
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

        # bb = 0
        recent_game_score = 0.0

        # statez = tt([[1., 0., 1., 0., 0., 1., 0., 0.],
        #             [1., 0., 0., 1., 0., 1., 0., 0.],
        #             [1., 0., 0., 0., 1., 1., 0., 0.],
        #             [0., 1., 1., 0., 0., 1., 0., 0.],
        #             [0., 1., 0., 1., 0., 1., 0., 0.],
        #             [0., 1., 0., 0., 1., 1., 0., 0.],
        #             [1., 0., 1., 0., 0., 0., 1., 0.],
        #             [1., 0., 0., 1., 0., 0., 1., 0.],
        #             [1., 0., 0., 0., 1., 0., 1., 0.],
        #             [0., 1., 1., 0., 0., 0., 1., 0.],
        #             [0., 1., 0., 1., 0., 0., 1., 0.],
        #             [0., 1., 0., 0., 1., 0., 1., 0.]])

        # statez = tt([[1., 0., 1., 0., 0.],
        #             [1., 0., 0., 1., 0.],
        #             [1., 0., 0., 0., 1.],
        #             [0., 1., 1., 0., 0.],
        #             [0., 1., 0., 1., 0.],
        #             [0., 1., 0., 0., 1.]])

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

                is_e_t, j, e_t = self.is_epsilon_transition(joint_action)
                # If an epsilon transition is made the game (and possibly spec) state remains the same 
                if is_e_t:
                    new_game_state, done = game_state, done
                    # new_spec_states_rewards = spec_states_rewards
                    # new_spec_states_rewards = [(state, 0.0) for (state, reward) in spec_states_rewards]
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

                if patient_updates:
                    # Temporarily save new product state
                    for j in range(self.num_specs): 
                        Z[j][tuple(prod_state.tolist())] = (t, prod_state, joint_action)

                # Compute rewards etc. from environment and automata
                unweighted_rewards = [reward_multiplier * s_r[1] for spec, s_r in zip(self.specs, new_spec_states_rewards)]
                discounts = [self.discounts['patient'] if r > 0.0 else 1.0 for r in unweighted_rewards]
                gammas = [g * d for g, d in zip(gammas, discounts)]
                new_spec_state_vectors = [one_hot(tt(new_spec_states_rewards[j][0]), self.specs[j].ldba.get_num_states()) for j in range(self.num_specs)]
                new_prod_state = torch.cat([env.featurise(new_game_state)] + new_spec_state_vectors, 0)

                for j in range(self.num_specs):

                    if patient_updates:
                        # Store patient critic experience when possible
                        if unweighted_rewards[j] > 0.0 or self.patient_critics[j](new_prod_state) == 0.0:
                            for k in Z[j].keys():
                                (p_t, p_s, p_a) = Z[j][k]
                                self.patient_buffer.add(j, p_t, p_s, p_a, unweighted_rewards[j], new_prod_state, gammas[j], done)

                                # print("Patient experience ({}):\nt: {}, s: {}, a: {}, r: {}, s': {}, gamma: {}".format(j, p_t, p_s, int(p_a[0]), rewards[j], new_prod_state, gammas[j]))
                            Z[j] = dict()
                    # else:
                    #     self.patient_buffer.add(j, t, prod_state, joint_action, unweighted_rewards[j], new_prod_state, gammas[j], done)

                    # Store hasty critic experience
                    self.hasty_buffer.add(j, t, prod_state, joint_action, unweighted_rewards[j], new_prod_state, gammas[j], done)
                    
                # print("Hasty experience:\nt: {}, s: {}, a: {}, r: {}, s': {}, gamma: {}".format(t, prod_state, int(joint_action[0]), rewards, new_prod_state, gammas))

                # Update variables for next step
                game_state = new_game_state
                spec_states_rewards = new_spec_states_rewards

                game_score += sum([r * spec.weight for r, spec in zip(unweighted_rewards, self.specs)])
                # bb += sum([r * spec.weight for r, spec in zip(unweighted_rewards, self.specs)])
                recent_game_score += sum([r * spec.weight for r, spec in zip(unweighted_rewards, self.specs)])

                t += 1
                s += 1

                # print(s)

                # Save and occasionally print (average) score
                if s % 100 == 0:
                    
                    average_game_score = game_score / s

                    print("Score ({}/{}): {} (recent)   {} (average)".format(int(s / 100), int(steps / 100), (recent_game_score / 100), average_game_score))

                    if len(first_50) < 60:
                        first_50.append(recent_game_score/100)
                    else:
                        if not on2awinner:
                            if np.var(first_50[10:]) < 0.01:
                                return False
                            else:
                                on2awinner = True
                    
                    last_100.append(recent_game_score/100)

                    
                    with open(filename, 'a') as f:
                        f.write('{}, {}, {}\n'.format(int(s / 100), average_game_score, recent_game_score / 100))
                    recent_game_score = 0

                    
                    if s > 50000 and average_game_score > best + 0.05:
                        self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                        best = average_game_score


                if len(last_100) == 100:
                    if np.var(last_100) < 0.001:
                        average_game_score = game_score / s
                        if average_game_score > best + 0.05:
                            self.policy_dists = dict([(p, self.policy(t, probs=True)) for p,t in zip(self.possible_states, self.possible_state_tensors)])
                        s = steps + 1   

                # if s % 100 == 0:

                    

                # if s % 10000 == 0:
                #     print("Losses (hasty then patient):")
                #     print(self.current_hasty_losses[0])
                #     print(self.current_patient_losses[0])
                #     print("Critic 1 (hasty then patient):")
                #     print(self.hasty_critics[0](possible_states))
                #     print(self.patient_critics[0](possible_states))
                #     print("Critic 2 (hasty then patient):")
                #     print(self.hasty_critics[1](possible_states))
                #     print(self.patient_critics[1](possible_states))
                #     print("Actor 1:")
                #     print(self.actors[0](possible_states)[0])
                #     print("Actor 2:")
                #     print(self.actors[1](possible_states)[0])

            # If episode has ended use stored data for learning (if enough stuff as been recorded)
            # if min([len(mem) for mem in self.patient_buffer.memory]) > 100:
            #     patient_data = [self.patient_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
            #     hasty_data = [self.hasty_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
            if True:

                hasty_data = [self.hasty_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
                if patient_updates:
                    patient_data = [self.patient_buffer.sample(j, sample_all=True) for j in range(self.num_specs)]
                    data = patient_data + hasty_data
                else:
                    patient_data = hasty_data
                    data = hasty_data

                # Compute new score functions
                # with torch.no_grad():
                for d in data:
                    if d != None:
                        for i in range(len(self.actors)):
                            if self.local:
                                data_state_actions = torch.unique(torch.cat((d[1],d[2][i]), dim=1), dim=0)
                            else:
                                data_state_actions = torch.unique(torch.cat((d[1],torch.cat(d[2], dim=1)), dim=1), dim=0)
                            keys = [tuple(tens.tolist()) for tens in data_state_actions]
                            new_pairs = [(k, tens) for k, tens in zip(keys, data_state_actions) if k not in self.scores[i].keys()]
                            self.scores[i].update(self.get_scores(i, new_pairs))
                        

                # statez = tt([[1., 0., 1., 0., 0.],
                #             [1., 0., 0., 1., 0.],
                #             [1., 0., 0., 0., 1.],
                #             [0., 1., 1., 0., 0.],
                #             [0., 1., 0., 1., 0.],
                #             [0., 1., 0., 0., 1.]])
                # print(self.hasty_critics[0](statez))
                # print(list(self.nat_grads[0].parameters())[0])

                # Update critics and natural gradients
                self.update_critics(e, patient_data, hasty_data, continue_prob, sum_val_reg, c_neg_var_reg, max_critic_norm, actual_dist)
                self.update_nat_grads(e, hasty_data, continue_prob, max_nat_grad_norm, actual_dist)
                self.update_mu(e)


                # print(self.current_hasty_losses[0])
                # print(self.current_patient_losses[0])
                # print(self.hasty_critics[0](statez))
                # print(self.patient_critics[0](statez))
                # print(self.actors[0](statez)[0])


                # if device.type == 'cuda':
                #     print(torch.cuda.get_device_name(0))
                #     print('Memory Usage:')
                #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,5), 'MB')
                #     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**2,5), 'MB')

            # If natural gradients have converged, update the policy and clear buffers and old scores
            for i in range(len(self.actors)):
                if self.nat_grads_converged(i, tolerance):                
                    print("Natural gradient {} converged!".format(i))
                    self.update_actors(i, e, neg_ent_reg, non_det_reg, a_neg_var_reg)
                    self.patient_buffer.clear()
                    self.hasty_buffer.clear()




                    # print(self.actors[0](possible_states))

                    # self.scores[i].clear()

                    # #Additional Info when using cuda
                    # if device.type == 'cuda':
                    #     print(torch.cuda.get_device_name(0))
                    #     print('Memory Usage:')
                    #     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,5), 'MB')
                    #     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**2,5), 'MB')

            # for p in self.actors[0].parameters():
            #     print(p)
        
        # print(self.current_hasty_losses[0])
        # print(self.hasty_critics[0](statez))
        # print(self.patient_critics[0](statez))
        # print(self.actors[0](statez)[0])

        # self.evaluate(policy=policy1)

        print("done!")

        return True

    # @profile
    def get_scores(self, i, new_pairs):

        if self.local:
            # scores = dict([(k, torch.autograd.grad(self.policy(t[0:-1])[i].log_prob(t[-1]).to(device), self.actors[i].parameters(), retain_graph=True)) for (k, t) in new_pairs])
            scores = dict()
            for (k, t) in new_pairs:
                 scores[k] = torch.cat([torch.flatten(g) for g in torch.autograd.grad(self.policy(t[0:-1])[i].log_prob(t[-1]).to(device), self.actors[i].parameters(), retain_graph=True)])
        else:
            scores = dict([(k, torch.autograd.grad(sum([self.policy(t[0:-self.num_players])[j].log_prob(t[j-self.num_players]) for j in range(self.num_players)]).to(device), self.actors[i].parameters(), retain_graph=True)) for (k, t) in new_pairs]) 

        return scores

    def state_dist(self, loss, actual_dist, continue_prob, ts, gammas):

        if loss == 'patient':
            if actual_dist:
                dist = power(continue_prob * ones_like(ts), -ts) * gammas
            else:
                dist = power(continue_prob * ones_like(ts), -ts)
        elif loss == 'hasty':
            if actual_dist:
                dist = power((self.discounts['hasty'] / continue_prob) * ones_like(ts), ts)
            else:
                dist = power(continue_prob * ones_like(ts), -ts)
        else:
            print("Error: State dist must use loss \'patient\' or \'hasty\'")
        return dist
    
    def nat_grads_converged(self, i, tolerance=0.01, bound=0.1):

        # for i in range(len(self.actors)):

        if len(self.current_patient_losses[i]) == 0 or len(self.current_hasty_losses[i]) == 0:
            return False

        pls = tt(self.current_patient_losses[i]).float()
        hls = tt(self.current_hasty_losses[i]).float()
        mean_pls = pls.mean()
        mean_hls = hls.mean()

        if mean_pls > bound or mean_hls > bound:

            if pls.min() < (1.0 - tolerance) * mean_pls or pls.max() > (1.0 + tolerance) * mean_pls:
                return False
            if hls.min() < (1.0 - tolerance) * mean_hls or hls.max() > (1.0 + tolerance) * mean_hls:
                return False
        
        return True 

    def is_epsilon_transition(self, joint_action):

        is_e_t = False
        eps_trans = []
        for i in range(len(self.act_sizes)):
            a = joint_action[i] - (self.act_sizes[i] - 1) 
            if a <= 0:
                continue
            else:
                is_e_t = True
                j = -1
                while a > 0:
                    j += 1
                    a -= self.epsilon_act_sizes[j]
                eps_trans.append((j, a + self.epsilon_act_sizes[j] - 1))
        eps_trans = [(j, e_t) for (j, e_t) in eps_trans if self.specs[j].ldba.check_epsilon(e_t)]
        if is_e_t:
            if len(eps_trans) > 0:
                (j, e_t) = random.choice(eps_trans)
            else:
                j, e_t = None, None
        else:
            j, e_t = None, None

        return is_e_t, j, e_t

    def evaluate(self, policy=None, play=False, iterations=10, steps=100, continue_prob=0.9):

        results = []

        for it in range(iterations):

            s = 1
            game_score = 0.0

            # Run episodes
            while s < steps:
                
                # Initialise environment
                game_state = env.reset()
                spec_states_rewards = [(spec.ldba.reset(), 0.0) for spec in self.specs]
                done = False
                if play or policy != None:
                    print("End of episode, resetting...")

                while not random.random() > continue_prob and not done:

                    if play or policy != None:
                        spec_states = [ssr[0] for ssr in spec_states_rewards]
                        rewards = [ssr[1] for ssr in spec_states_rewards]
                        print("Game state: {}, Spec states: {}, Rewards: {}".format(game_state, spec_states, rewards))

                    spec_state_vectors = [one_hot(tt(spec_states_rewards[j][0]), self.specs[j].ldba.get_num_states()) for j in range(self.num_specs)]
                    prod_state = torch.cat([env.featurise(game_state)] + spec_state_vectors, 0)

     
                        # Get action
                    if play:
                        joint_action = list(map(lambda x: tt(int(x)),input("Enter actions: ").strip().split()))[:self.num_players] 
                    elif policy!= None:
                        joint_action = [torch.argmax(policy(prod_state))]
                    else:
                        joint_action = [actions.sample() for actions in self.policy(prod_state)]

                    is_e_t, j, e_t = self.is_epsilon_transition(joint_action)
                    # If an epsilon transition is made the game (and possibly spec) state remains the same 
                    if is_e_t:
                        new_game_state, done = game_state, done
                        # new_spec_states_rewards = spec_states_rewards
                        new_spec_states_rewards = [(state, 0.0) for (state, reward) in spec_states_rewards]
                        # If the chosen epsilon transition is available in this state then it is made
                        if e_t != None:
                            new_spec_states_rewards[j] = self.specs[j].ldba.step(None, epsilon=e_t)
                    # Otherwise a standard action is performed in the MG
                    else:
                        new_game_state, done = env.step(joint_action)
                        label_set = env.label(new_game_state)
                        new_spec_states_rewards = [spec.ldba.step(label_set) for spec in self.specs]

                    # Compute rewards etc. from environment and automata
                    rewards = [spec.weight * s_r[1] for spec, s_r in zip(self.specs, new_spec_states_rewards)]

                    new_spec_state_vectors = [one_hot(tt(new_spec_states_rewards[j][0]), self.specs[j].ldba.get_num_states()) for j in range(self.num_specs)]
                    new_prod_state = torch.cat([env.featurise(new_game_state)] + new_spec_state_vectors, 0)

                    # Update variables for next step
                    game_state = new_game_state
                    spec_states_rewards = new_spec_states_rewards

                    game_score += sum(rewards)
                    s += 1

            results.append(game_score / s)

            # statez = tt([[1., 0., 1., 0., 0., 1., 0.],
            #             [1., 0., 0., 1., 0., 1., 0.],
            #             [1., 0., 0., 0., 1., 1., 0.],
            #             [0., 1., 1., 0., 0., 1., 0.],
            #             [0., 1., 0., 1., 0., 1., 0.],
            #             [0., 1., 0., 0., 1., 1., 0.],
            #             [1., 0., 1., 0., 0., 0., 1.],
            #             [1., 0., 0., 1., 0., 0., 1.],
            #             [1., 0., 0., 0., 1., 0., 1.],
            #             [0., 1., 1., 0., 0., 0., 1.],
            #             [0., 1., 0., 1., 0., 0., 1.],
            #             [0., 1., 0., 0., 1., 0., 1.]])

            # print(self.actors[0](statez)[0])

        print(results)

    def save(self):

        pass

        #     actor_path = 'results/models/almanac-{}-{}/actors.pth'.format(self.model_id, self.env_name)
        #     actor_list = ['actor_{}_state_dict'.format(i) for i in range(len(self.actors))]
        #     torch.save(dict(zip(actor_list, [actor.state_dict() for actor in self.actors])), actor_path)

        #     patient_critic_path = 'results/models/almanac-{}-{}/patient_critics.pth'.format(self.model_id, self.env_name)
        #     patient_critic_list = ['patient_critic_{}_state_dict'.format(i) for i in range(len(self.patient_critics))]
        #     torch.save(dict(zip(patient_critic_list, [critic.state_dict() for critic in self.patient_critics])), patient_critic_path)

        #     hasty_critic_path = 'results/models/almanac-{}-{}/hasty_critics.pth'.format(self.model_id, self.env_name)
        #     hasty_critic_list = ['hasty_critic_{}_state_dict'.format(i) for i in range(len(self.hasty_critics))]
        #     torch.save(dict(zip(hasty_critic_list, [critic.state_dict() for critic in self.hasty_critics])), hasty_critic_path)

        # TODO save nat_grads

        # TODO save the rest of these
        # # Parameters
        # self.obs_size = env.get_obs_size()
        # self.act_sizes = env.get_act_sizes()
        # self.lrs = lrs
        # self.discounts = discounts
        # self.num_specs = len(specs)
        # self.num_players = len(self.act_sizes)
        # self.local = local
        # self.env_name = env.get_name()
        # self.model_id = random.randint(1e6) if model_id == None else model_id 

        # # Automata
        # self.automata = [(LDBA(spec.formula), spec.weight) for spec in specs]
        # self.epsilon_act_sizes = [ldba.get_num_states() for (ldba, _) in self.automata]
       
        # # Networks
        # self.actors = set_up_actors(actor, local, self.obs_size, self.act_sizes + sum(self.epsilon_act_sizes), hidden=[16])
        # self.patient_critics = set_up_critics(critic, self.obs_size, self.num_specs, hidden=[16])
        # self.hasty_critics = set_up_critics(critic, self.obs_size, self.num_specs, hidden=[16])

        # # Natural gradients
        # self.param_shapes = []
        # for actor in self.actors:
        #     model_params = filter(lambda p: p.requires_grad, actor.parameters())
        #     self.param_shapes.append(tuple([p.shape for p in model_params]))
        # self.nat_grads = [NaturalGradient(params) for params in self.param_shapes]
        # if torch.cuda.is_available():
        #     for nat_grad in self.nat_grads:
        #         nat_grad.cuda()
        
        # # Optimisers
        # self.actor_optimisers = [optim.Adam(actor.parameters(), lr=self.lrs['actor'][i]) for actor in self.actors]
        # self.patient_critic_optimisers = [optim.Adam(critic.parameters(), lr=self.lrs['critic'][j]) for critic in self.patient_critics]
        # self.hasty_critic_optimisers = [optim.Adam(critic.parameters(), lr=self.lrs['critic'][j]) for critic in self.patient_critics]
        # self.nat_grad_optimisers = [optim.Adam(nat_grad.parameters(), lr=self.lrs['nat_grad'][i]) for nat_grad in self.nat_grads]

        # # Replay buffers
        # self.patient_buffer = ReplayBuffer(self.num_specs, buffer_size=1e6, batch_size=32)
        # self.hasty_buffer = ReplayBuffer(self.num_specs, buffer_size=1e6, batch_size=32)

        # # Lagrange multipliers
        # self.mu = [0.0 for a in self.actors]
        # self.current_patient_loss = [0.0 for a in self.actors]

    def get_policy_dists(self, save=False):

        # possible_game_states = list(itertools.product([0, 1], repeat=self.obs_size))

        # possible_spec_states = []
        # for s_s in self.ldba_state_sizes:
        #     spec_states = [[0 for i in range(s_s)] for j in range(s_s)]
        #     for k in range(s_s):
        #         spec_states[k][k] = 1
        #     possible_spec_states.append([tuple(s_s) for s_s in spec_states])

        # possible_product_states = itertools.product(possible_game_states, *possible_spec_states)

        # possible_states = [tuple(utils.flatten(p_s)) for p_s in possible_product_states]

        # policies = self.policy(self.possible_state_tensors, probs=True)

        # # return  dict(zip([p, self.policy(t, probs=True) for p,t in zip(self.possible_states, self.possible_state_tensors)

        # return dict(zip(self.possible_states, policies))

        return self.policy_dists
        



### TESTS ###

def policy1(state):

    state = state.long()

    if torch.equal(state, tt([1., 0., 1., 0., 0., 1., 0., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    elif torch.equal(state, tt([1., 0., 0., 1., 0., 1., 0., 0.]).long()):
        return tt([0.0, 1.0, 0.0]) #b
    elif torch.equal(state, tt([1., 0., 0., 0., 1., 1., 0., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    elif torch.equal(state, tt([0., 1., 1., 0., 0., 1., 0., 0.]).long()):
        return tt([0.0, 1.0, 0.0]) #b
    elif torch.equal(state, tt([0., 1., 0., 1., 0., 1., 0., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    elif torch.equal(state, tt([0., 1., 0., 0., 1., 1., 0., 0.]).long()):
        return tt([0.0, 1.0, 0.0]) #b
    elif torch.equal(state, tt([1., 0., 1., 0., 0., 0., 1., 0.]).long()):
        return tt([0.0, 1.0, 0.0]) #b
    elif torch.equal(state, tt([1., 0., 0., 1., 0., 0., 1., 0.]).long()):
        return tt([0.0, 1.0, 0.0]) #b
    elif torch.equal(state, tt([1., 0., 0., 0., 1., 0., 1., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    elif torch.equal(state, tt([0., 1., 1., 0., 0., 0., 1., 0.]).long()):
        return tt([0.0, 0.0, 1.0]) #epsilon
    elif torch.equal(state, tt([0., 1., 0., 1., 0., 0., 1., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    elif torch.equal(state, tt([0., 1., 0., 0., 1., 0., 1., 0.]).long()):
        return tt([1.0, 0.0, 0.0]) #a
    else:
        print("darn")


# env = envs.EnvWrapper('test', 'mg', envs.mg_0, envs.l_0)
# specs = (specs.Spec('F G psi', 0.7), specs.Spec('F phi', 0.3))
# lrs = {'critic': ('constant', 1.0),
#        'patient_nat_grad': ('constant', 0.01),
#        'hasty_nat_grad': ('constant', 0.01),
#        'mu': ('constant', 0.01),
#        'actor': ('constant', 0.0001)}   
# opt = {'actor': 'sgd',
#        'nat_grad': 'sgd',
#        'critic': 'sgd'}
# discounts = {'patient': 0.9, 'hasty': 0.85}
# almanac = Almanac(env, specs, lrs, discounts, opt, critic='dnn', actor='dnn')
# almanac.train(500000, env, specs, 0.9)

# debug_1 = { 'actor_neg_entropy_regulariser_weight': 0.01,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.9,
#             'critic_sum_value_regulariser_weight': 0.001,
#             'critic_neg_variance_regulariser_weight': 0.001,
#             'discounts': { 'hasty': 0.85,
#                            'patient': 0.95 },
#             'env': envs.mg_0,
#             'env_name': 'debug_mg0',
#             'env_type': 'mg',
#             'epsilon': lambda e: 0.01,
#             'l2_regulariser_weight': 0.001,
#             'labeller': envs.l_0,
#             'learning_rates': { 'actor': ('constant', 0.01),
#                                 'critic': ('constant', 0.1),
#                                 'patient_nat_grad': ('constant', 0.05),
#                                 'hasty_nat_grad': ('constant', 0.05),
#                                 'mu': ('constant', 0.01) },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.25,
#             'max_critic_norm': 10.0,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [16, 16, 16]),
#                         'critic': ('dnn', [8, 16, 8]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'adam',
#                             'critic': 'sgd' },
#             'patient_updates': False,
#             'run_id': None,
#             'reward_weight': 10.0,
#             'specs': [ ('F G psi', 1.0)
#                     #   , ('F phi', 0.3)
#                      ],
#             'steps': 300000 }



# def run(hp):

#     env = envs.EnvWrapper(hp['env_name'], hp['env_type'], hp['env'], hp['labeller'])
#     spec = []
#     for s in hp['specs']:
#         f = "specs/" + s[0] + '.pickle'
#         if path.isfile(f):
#             old_spec = pickle.load(open(f, "rb"))
#             spec.append(old_spec)
#         else:
#             new_spec = specs.Spec(s[0], s[1])
#             new_spec.save()
#             spec.append(new_spec)
#     model_constants = { 'discounts': hp['discounts'],
#                         'l2_reg': hp['l2_regulariser_weight'],
#                         'lrs': hp['learning_rates'] }
#     almanac = Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants, hp['model_id'])
#     train_constants = { 'continue_prob': hp['continue_prob'],
#                         'epsilon': hp['epsilon'],
#                         'nat_grad_tolerance': hp['nat_grad_convergence_tolerance'],
#                         'neg_ent_reg': hp['actor_neg_entropy_regulariser_weight'],
#                         'sum_val_reg': hp['critic_sum_value_regulariser_weight'],
#                         'c_neg_var_reg': hp['critic_neg_variance_regulariser_weight'],
#                         'max_nat_grad_norm': hp['max_nat_grad_norm'],
#                         'max_critic_norm': hp['max_critic_norm'],
#                         'reward_weight': hp['reward_weight'] }
#     almanac.train(hp['steps'], env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'])
#     if hp['env_type'] == 'mmg':
#         env.model.create_prism_model(0, spec, policy=almanac.get_policy_dists())

# run(debug_1)



















# def run(hp, num, filename=None, modelname=None):

#     env = envs.EnvWrapper(hp['env_name'], hp['env_type'], hp['env'], hp['labeller'])
#     spec = []
#     for s in hp['specs']:
#         f = "specs/" + s[0] + '.pickle'
#         if path.isfile(f):
#             old_spec = pickle.load(open(f, "rb"))
#             spec.append(old_spec)
#         else:
#             new_spec = specs.Spec(s[0], s[1])
#             new_spec.save()
#             spec.append(new_spec)
#     model_constants = { 'discounts': hp['discounts'],
#                         'l2_reg': hp['l2_regulariser_weight'],
#                         'lrs': hp['learning_rates'] }
#     almanac = Almanac(env, spec, hp['optimisers'], hp['buffers'], hp['models'], hp['local'], model_constants, hp['model_id'])
#     train_constants = { 'continue_prob': hp['continue_prob'],
#                         'epsilon': hp['epsilon'],
#                         'nat_grad_tolerance': hp['nat_grad_convergence_tolerance'],
#                         'neg_ent_reg': hp['actor_neg_entropy_regulariser_weight'],
#                         'non_det_reg': hp['actor_nondeterminism_regulariser_weight'],
#                         'sum_val_reg': hp['critic_sum_value_regulariser_weight'],
#                         'c_neg_var_reg': hp['critic_neg_variance_regulariser_weight'],
#                         'max_nat_grad_norm': hp['max_nat_grad_norm'],
#                         'max_critic_norm': hp['max_critic_norm'],
#                         'reward_weight': hp['reward_weight'] }
#     almanac.train(hp['steps'], env, spec, hp['actual_state_dist'], hp['patient_updates'], train_constants, hp['run_id'], filename)
#     if hp['env_type'] == 'mmg':
#         env.model.create_prism_model(num, spec)
#         env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists())
#         env.model.create_prism_model(num, spec, policy=almanac.get_policy_dists(), det=True)
#     if modelname != None:
#         almanac.lrs = None
#         almanac.patient_buffer = None
#         almanac.hasty_buffer = None
#         with open(modelname, 'wb') as f:
#             pickle.dump(almanac, f)


# mmg_hps = { 'actor_neg_entropy_regulariser_weight': 5.0,
#             'actor_nondeterminism_regulariser_weight': 10.0,
#             'actual_state_dist': True,
#             'buffers': { 'size': 1000,
#                          'batch': 32 },
#             'continue_prob': 0.95,
#             'critic_sum_value_regulariser_weight': 0.01,
#             'critic_neg_variance_regulariser_weight': 0.001,
#             'discounts': { 'hasty': 0.85,
#                            'patient': 0.95 },
#             'env': None,
#             'env_name': None,
#             'env_type': 'mmg',
#             'epsilon': lambda e: 0.05,
#             'l2_regulariser_weight': 0.001,
#             'labeller': None,
#             'learning_rates': { 'actor': ('constant', 0.1),
#                                 'critic': ('constant', 1.0),
#                                 'patient_nat_grad': ('constant', 1.00),
#                                 'hasty_nat_grad': ('constant', 1.00),
#                                 'mu': ('constant', 0.01) },
#             'local': True,
#             'nat_grad_convergence_tolerance': 0.25,
#             'max_critic_norm': 10.0,
#             'max_nat_grad_norm': 10.0,
#             'models': { 'actor': ('dnn', [24, 32, 32, 24]),
#                         'critic': ('dnn', [24, 24, 24]) },
#             'model_id': None,
#             'optimisers': { 'actor': 'sgd',
#                             'nat_grad': 'adam',
#                             'critic': 'sgd' },
#             'patient_updates': True,
#             'run_id': None,
#             'reward_weight': 10,
#             'specs': None,
#             'steps': 50000 }

# labels = ['phi','psi','chi','xi']
# possible_specs = ['G F psi', 'F G (!phi | !xi)', 'G ((!phi) | (X chi))', 'F xi', 'G ((!psi) | (F phi))', 'G chi', '(!xi) U psi']
# possible_weights = [0.2, 0.5, 0.8]

# state_size = int(sys.argv[1])
# num_actors = int(sys.argv[2])
# num_specs = int(sys.argv[3])
# run_num = int(sys.argv[4])

# state_size = 8
# num_actors = 4
# num_specs = 2
# run_num = 0

# specifications = random.sample(possible_specs, num_specs)
# weights = random.sample(possible_weights, num_specs)
# sum_weights = sum(weights)
# weights = [w / sum_weights for w in weights]
# action_sizes = [random.randint(2,4) for r in range(num_actors)]
# mmg = envs.MatrixMarkovGame(state_size, action_sizes, labels, sparsity=0.6, structured_labels=True, nonlinearities=0.4)

# mmg_hps['env'] = mmg
# mmg_hps['env_name'] = None
# mmg_hps['labeller'] = mmg.labeller
# mmg_hps['model_id'] = None
# mmg_hps['specs'] = list(zip(specifications, weights))

# filename = 'results/scores/mmg/almanac-{}-{}-{}-{}.txt'.format(state_size, num_actors, num_specs, run_num)
# modelname = 'results/models/mmg/almanac-{}-{}-{}-{}.pickle'.format(state_size, num_actors, num_specs, run_num)
# with open(filename, 'w') as f:
#     f.write("State size: {}\n".format(state_size))
#     a_s_line = ', '.join([str(a) for a in action_sizes])
#     f.write("Action sizes: {}\n".format(a_s_line))
#     s_line = ', '.join(['{}: {}'.format(w, s) for s, w in zip(specifications, weights)])
#     f.write("Specs: {}\n".format(s_line))
#     f.write("Run: {}\n\n".format(run_num))

# for speckle in possible_specs:
#     s = specs.Spec(speckle, 1.0)
#     s.save()

# import time
# start_time = time.time()
# run(mmg_hps, run_num, filename, modelname)
# elapsed_time = time.time() - start_time
# print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))