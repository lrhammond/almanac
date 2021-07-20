### Specifications ###

# Note, this code is primarily taken from work by Alper Kamil Bozkurt
# His original repository can be found at https://gitlab.oit.duke.edu/cpsl/csrl/ and is licensed under the Apache License 2.0 (also included for reference within this repository)

from subprocess import check_output
import random
from utils import flatten
import os
import re
from itertools import chain, combinations
import pickle


# Specification class
class Spec:

    def __init__(self, formula, weight):

        self.formula = formula
        self.weight = weight
        self.ldba = LDBA(formula)

    def save(self, location):

        # if filename == None:
        #     filename = self.formula
        with open(location + "/{}.pickle".format(self.formula), 'wb') as f:
            pickle.dump(self, f)
        
    def create_prism_model(self, spec_num, num_players, to_file=False, filename=None):

        lines = '// {}\n'.format(self.formula)
        lines += 'module SPEC_{}\n\n'.format(spec_num)
        lines += '    q{} : [0..{}] init 0;\n'.format(spec_num, self.ldba.num_states - 1)

        ls = self.ldba.labels
        
        for i in range(self.ldba.num_states):
            
            eps = self.ldba.eps[i]
            epsilon_actions  = []

            for e in eps:

                guard = '    [transition] q{}={} & '.format(spec_num, i)
                eps_trans = '('
                for a in range(num_players):
                    eps_trans += 'a{}=eps_{}_{}'.format(a, spec_num, e)
                    if a != num_players - 1:
                        eps_trans += ' | '
                    else:
                        eps_trans += ')'
                epsilon_actions.append(eps_trans)   
                lines += guard + eps_trans + ' -> 1.0:(q{}\'={});\n'.format(spec_num, e)

            if len(epsilon_actions) != 0:
                no_eps_trans = ' & !' + ' & !'.join(epsilon_actions)
            else:
                no_eps_trans = ''

            for k in self.ldba.delta[i].keys():

                guard = '    [transition] q{}={}'.format(spec_num, i)
                for l in ls:
                    if l in k:
                        guard += ' & ' + l
                    else:
                        guard += ' & !' + l
                lines += guard + no_eps_trans + ' -> 1.0:(q{}\'={});\n'.format(spec_num, self.ldba.delta[i][k])

        lines += '\nendmodule\n\n'

        if to_file:
            if filename == None:
                filename = 'specs/{}.prism'.format(self.formula)
            with open(filename, 'w') as f:
                f.write("mdp\n\n")
                f.write(lines)
        
        return lines


# Automaton class
class LDBA:
    """Transforms the LTL formula to an omega-automaton (OA) and stores the specifications.
    
    Attributes
    ----------
    q0 : int
        The initial state of the OA.

    delta : list of dicts
        The transition function of the OA. delta[q][label_set] is the number of the state that the OA makes a transition to when it consumes the label_set in the state q.
        
    eps : list of lists
        The epsilon-moves of the OA. epsilon_moves[q] is the set of states the OA can nondeterministically make a transition from state q.

    acc : array, shape (n_qs,n_pairs)
        The n_qs x n_pairs matrix that represents the accepting condition. If acc[q][i] is false then it means that q belongs to the first set of ith Rabin pair,
        if it is true, then q belongs to the second set and if it is none q doesn't belong either of them. The Buchi condition is represented by a single Rabin pair.
        
    shape : tuple
        The pair of the number of the Rabin pairs and the number of states in the OA, i.e. : (n_pairs,n_qs)

    spot_oa : spot.twa_graph
        The spot twa_graph object of the OA for visualization.
        
    Parameters
    ----------
    ltl : str
        The linear temporal logic (LTL) formula to be transformed to a OA.
        
    oa_type : str
        The type of the OA to be constructed. The default value is 'ldba'
        
    """

    def __init__(self, ltl):
        """Constructs and returns dictionaries and lists containing the specifications of an OA obtained by translation from the ltl property.
        It parses the output of ltl2ldba or ltl2dra for the ltl formula and creates a objects that store the specification of the OA.

        Parameters
        ----------
        ltl : str
            The linear temporal logic (LTL) formula to be transformed to a OA.

        Returns
        -------
        out : (q0, delta, acc, eps, shape, spot_oa)
            The tuple of the initial state q0, the list of dictionaries of transitions delta, 
            the list of dictionaries of the accepting transitions, the list of lists of epsilon-moves,
            the pair of the number of the Rabin pairs and the number of states and the spot object of the OA.
            
        """

        # Translate the LTL formula to an OA using Rabinizer 4.
        out=check_output(['rabinizer4/bin/ltl2ldba', '-e', ltl])
        
        # Split the output into two parts: the header and the body
        header, body = out.decode('utf-8').split('--BODY--\n')
        
        # Parse the initial state, the atomic propositions and the number of Rabin pairs
        for line in header.splitlines():
            if line.startswith('Start'):
                q0 = int(line[7:])  # The initial state
            elif line.startswith('AP'):
                char_map = {i:c for i,c in enumerate(re.sub("[^\w]", " ",  line[4:]).split()[1:])}  # Maps ids to atomic propositions
                ap_list = [tuple(ap) for ap in self.powerset(sorted(char_map.values()))]  # The list of all subsets of AP.
            elif line.startswith('Acceptance'):
                n_pairs = int(line.split()[1])//2  # Zero for the Buchi condition
                
        body_lines = body.splitlines()[:-1]  # Ignore the last line
        
        # Get the number of states
        n_qs = 0  # The number of states
        for line in reversed(body_lines):  # Loop over all states because the states might not be ordered.
            if line.startswith('State'):
                n_qs = max(int(line[7:]),n_qs)  # Get the maximum of them 
                
        n_qs += 2  # +1 because the index origin is 0 and +1 for the trap state
        n_i = max(1,n_pairs)  # Because n_pairs is zero for the Buchi condition
        shape = n_i, n_qs
        
        # The transition function delta[q][label] stores the next state The OA makes a transition when the it consumes 'label' at state 'q'.
        delta = [{ap:n_qs-1 for ap in ap_list} for i in range(n_qs)]  # The default target of a transition is the trap state whose index is n_qs-1
        acc = [{ap:[None]*n_i for ap in ap_list} for i in range(n_qs)]  # The default acceptance value is None, meaning the transition does not belong to any acceptance set.
        eps = [[] for i in range(n_qs)]  # The epsilon moves in the OA. eps[q] is the list of states can be reached from `q` by making an epsilon-transition.
        
        # Parse the transitions, acceptance values
        q=-1  # The state to be parsed
        for line in body_lines:
            if line.startswith('State'):
                q = int(line[7:])  # Update the state to be parsed 
            else:
                # Parse the transition into three parts
                _, _label, _dst, _, _acc_set = re.findall('(\[(.*)\])? ?(\d+) ?(\{(.*)\})?',line)[0]
                dst = int(_dst)  # Get the destination
                if not _label: # If there is no label then the transition is an epsilon-move
                    eps[q].append(dst)
                else:
                    # Get the acceptance status of the transition
                    acc_set = set([int(a) for a in _acc_set.split()])  # The set of acceptance states that the transition belongs to
                    if not n_pairs: # acc_name == 'Buchi':
                        t_acc = [True if 0 in acc_set else None]  # If it is an Buchi set, then it is True and None otherwise
                    else:
                        t_acc = [None]*n_pairs
                        for i in range(n_pairs):  # For each Rabin pairs
                            if 2*i+1 in acc_set:
                                t_acc[i] = True  # True if it belongs to the second set of the Rabin pair
                            if 2*i in acc_set:
                                t_acc[i] = False  # False if it belongs to the first set of the Rabin pair
                    
                    labels = ['']
                    _labels = re.compile('[()]').split(_label)  # The transitions might have subformulas
                    for _l in _labels:
                        labels = [l+_ll for l in labels for _ll in _l.split('|')]  # Add all the combinations

                    for label in labels:
                        if label == 't':  # Means all the transitions
                            label_acc, label_rej = set(()), set(())
                        else:
                            ls = list(filter(None,re.compile('[\s&]').split(label)))  # Get the atoms
                            label_acc = set([char_map[int(l)] for l in ls if not l.startswith('!')])  # Transitions having these atoms
                            label_rej = set([char_map[int(l[1:])] for l in ls if l.startswith('!')])  # Transitions that don't have these

                        for ap in delta[q]:  # Find all the matching transitions
                            # If matches, update the transition properties
                            if not(label_acc-set(ap)) and (label_rej-set(ap))==label_rej:  
                                delta[q][ap] = dst
                                acc[q][ap] = t_acc

        self.q0 = q0
        self.state = q0
        self.delta = [dict(zip([tuple(sorted(key)) for key in d.keys()], d.values())) for d in delta]
        self.acc = acc
        self.shape = shape
        self.eps = eps
        self.num_states = n_qs
        self.labels = set(flatten(self.delta[0].keys()))
        self.eps_actions = sorted(set(flatten(self.eps)))

    def get_num_states(self):
        
        return self.num_states

    def get_num_eps_actions(self):
        
        return len(self.eps_actions)

    def reset(self):

        self.state = self.q0
        return self.state

    def step(self, label_set, epsilon=None):

        if epsilon == None:
            l = tuple(sorted(tuple(self.labels.intersection(set(label_set)))))
            reward = 1.0 if self.acc[self.state][l][0] else 0.0
            self.state = self.delta[self.state][l]
        elif self.eps_actions[epsilon] in self.eps[self.state]:
            reward = 0.0
            self.state = self.eps_actions[epsilon]
        else:
            print("Error: Epsilon transition not available!")
        return (self.state, reward)

    def check_epsilon(self, epsilon):

        return self.eps_actions[epsilon] in self.eps[self.state]
    
    def powerset(self,a):
        """Returns the power set of the given list.

        Parameters
        ----------
        a : list
            The input list.

        Returns
        -------
        out: str
            The power set of the list.
        """
        return chain.from_iterable(combinations(a, k) for k in range(len(a)+1))

    def random_hoa_filename(self):
        """Returns a random file name.
        
        Returns
        -------
        filename: str
            A random file name.
        """
        filename = 'temp_%032x.hoa' % random.getrandbits(128)
        while os.path.isfile(filename):
            filename = 'temp_%032x.hoa' % random.getrandbits(128)
        return filename