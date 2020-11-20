### Utilities ###

from torch import clamp, isnan, tensor
import ast


# Flattens list of lists to a single list
def flatten(l):

    return [item for sublist in l for item in sublist]


# Prevents division by 0
def denom(x):

    return x if x != 0.0 else 1.0


# Clips network weights
def clip(x, min_x=-100.0, max_x=100.0):

    return clamp(x, min_x, max_x)


# Removes Nan entries from a tensor
def remove_nans(x):

    if x == None:
        return x
    if not isinstance(x, int):
        x[isnan(x)] = 0
    return x


# Writes .sh files for running mmg experiments on ARC
def write_mmg_sh_files(range_repetitions, range_actions, range_states, range_specs):

    for t in range_repetitions:    
        for s in range_states:
            for a in range_actions:
                for l in range_specs:

                    if (s + a >= 10) or (s >= 7):
                        mb = 16384
                        gb = 16
                    else:
                        mb = 4096
                        gb = 4

                    filename = 'arc/almanac-{}-{}-{}-{}.sh'.format(s,a,l,t)
                    with open(filename, 'w') as f:

                        f.write("#!/bin/bash\n\
    #SBATCH --partition=htc\n\
    #SBATCH --job-name=2almanac{0}{1}{2}{3}\n\
    #SBATCH --output=/home/hert5888/almanac/logs/{0}-{1}-{2}-{3}.out\n\
    #SBATCH --error=/home/hert5888/almanac/logs/{0}-{1}-{2}-{3}.err\n\
    #SBATCH --time=24:00:00\n\
    #SBATCH --mem={4}\n\
    module load python/anaconda3/2019.03\n\
    source activate /home/hert5888/almanac/arc_venv\n\
    python /home/hert5888/almanac/experiments.py {0} {1} {2} {3}\n\
    sleep 240\n\
    module load prism/4.4-beta\n\
    prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/matrix_markov_games/{0}-{1}-{2}-{3}.prism \
    /home/hert5888/almanac/specs/matrix_markov_games/{0}-{1}-{2}-{3}.props -prop 1 > \
    /home/hert5888/almanac/results/matrix_markov_games/{0}-{1}-{2}-{3}-true.txt\n\
    sleep 60\n\
    prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/matrix_markov_games/{0}-{1}-{2}-{3}-policy.prism \
    /home/hert5888/almanac/specs/matrix_markov_games/{0}-{1}-{2}-{3}.props -prop 2 > \
    /home/hert5888/almanac/results/matrix_markov_games/{0}-{1}-{2}-{3}-policy-0.txt\n\
    sleep 60\n\
    prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/matrix_markov_games/{0}-{1}-{2}-{3}-policy-det.prism \
    /home/hert5888/almanac/specs/matrix_markov_games/{0}-{1}-{2}-{3}.props -prop 2 > \
    /home/hert5888/almanac/results/matrix_markov_games/{0}-{1}-{2}-{3}-policy-det-0.txt\n\
    sleep 60\n".format(s,a,l,t,mb,gb))

                        if l == 2:
                            f.write("prism -cuddmaxmem {4}g -javamaxmem {4}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/matrix_markov_games/{0}-{1}-{2}-{3}-policy.prism \
    /home/hert5888/almanac/specs/matrix_markov_games/{0}-{1}-{2}-{3}.props -prop 3 > \
    /home/hert5888/almanac/results/matrix_markov_games/{0}-{1}-{2}-{3}-policy-1.txt\n\
    sleep 60\n\
    prism -cuddmaxmem {4}g -javamaxmem {4}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/matrix_markov_games/{0}-{1}-{2}-{3}-policy-det.prism \
    /home/hert5888/almanac/specs/matrix_markov_games/{0}-{1}-{2}-{3}.props -prop 3 > \
    /home/hert5888/almanac/results/matrix_markov_games/{0}-{1}-{2}-{3}-policy-det-1.txt\n".format(s,a,l,t,gb))

                        f.write("conda deactivate")

                    filenames.append(filename)

    with open('multi.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        for fn in filenames:
            f.write("sbatch " + fn + "\n")


# Get mmg experiment results 
def get_mmg_results(range_repetitions, range_states, range_actions, range_specs)

    results = dict()

    for t in range_repetitions:    
        for s in range_states:
            for a in range_actions:
                for l in range_specs:
                
                key = (s,a,l,t)         
                key_string = 'results/experiment_1/prism_evaluations/{}-{}-{}-{}'.format(s,a,l,t)
                results[key] = []

                if l == 1:
                    with open(key_string + '-true.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                entry  = float(line[8:-29])
                                results[key].append(entry)
                        if len(results[key]) == 0:
                            results[key].append(None)
                    with open(key_string + '-policy-0.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                entry  = float(line[8:-29])
                                results[key].append(entry)
                        if len(results[key]) == 1:
                            results[key].append(None)
                    with open(key_string + '-policy-det-0.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                entry  = float(line[8:-29])
                                results[key].append(entry)
                        if len(results[key]) == 2:
                            results[key].append(None)

                if l == 2:
                    with open(key_string + '-true.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                entry = ast.literal_eval(line[8:-29])
                                results[key].append(entry)
                        if len(results[key]) == 0:
                            results[key].append(None)
                    pol_1 = None
                    with open(key_string + '-policy-0.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                pol_1  = float(line[8:-29])
                    pol_2 = None
                    with open(key_string + '-policy-1.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                pol_2  = float(line[8:-29])
                    results[key].append((pol_1,pol_2))
                    det_1 = None
                    with open(key_string + '-policy-det-0.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                det_1  = float(line[8:-29])
                    det_2 = None
                    with open(key_string + '-policy-det-1.txt') as f:
                        for i, line in enumerate(f):
                            if line[:7] == 'Result:':
                                det_2  = float(line[8:-29])
                    results[key].append((det_1,det_2))

    weights = dict()

    for t in range_repetitions:    
        for s in range_states:
            for a in range_actions:
                for l in range_specs:
                    
                    key = (s,a,l,t)
                    key_string = 'results/experiment_1/specs/{}-{}-{}-{}.weights'.format(s,a,l,t)
                    with open(key_string) as f:
                        for i, line in enumerate(f):
                            if i == 0:
                                w1 = float(line)
                            if i == 1:
                                w2 = float(line)
                        weights[key] = (w1, w2)
        
    averages = dict()

    for s in range_states:
        for a in range_actions:
            for l in range_specs:

                key = (s,a,l)
                trues = [results[s,a,l,t][0] for t in range_repetitions]
                pols = [results[s,a,l,t][1] for t in range_repetitions]
                dets = [results[s,a,l,t][2] for t in range_repetitions]
                pol_errors = []
                det_errors = []

                if l == 1:
                    for t, p, d in zip(trues, pols, dets):
                        if t == None:
                            continue
                        if p != None:
                            pol_errors.append(t-p)
                        if d != None:
                            det_errors.append(t-d)
                    if len(pol_errors) == 0:
                        av_pol_err = None
                    else:
                        av_pol_err = sum(pol_errors) / len(pol_errors)
                    if len(det_errors) == 0:
                        av_det_err = None
                    else:
                        av_det_err = sum(det_errors) / len(det_errors)
                    averages[key] = (av_pol_err, av_det_err) 

                if l == 2:
                    ws = [weights[s,a,l,t] for t in [1,2,3,4,5,6,7,8]]
                    for t, p, d, w in zip(trues, pols, dets, ws):
                        if t == None:
                            continue
                        if p[0] != None and p[1] != None:
                            pol = (w[0] * p[0]) + (w[1] * p[1])
                            true = max([(w[0] * res[0]) + (w[1] * res[1]) for res in t])
                            pol_errors.append(true-pol)
                        if d[0] != None and d[1] != None:
                            det = (w[0] * d[0]) + (w[1] * d[1])
                            true = max([(w[0] * res[0]) + (w[1] * res[1]) for res in t])
                            det_errors.append(true-det)
                    if len(pol_errors) == 0:
                        av_pol_err = None
                    else:
                        av_pol_err = sum(pol_errors) / len(pol_errors)
                    if len(det_errors) == 0:
                        av_det_err = None
                    else:
                        av_det_err = sum(det_errors) / len(det_errors)

                    averages[key] = (av_pol_err, av_det_err)
   
    for k in averages.keys():
        if averages[k][0] == None:
            averages[k] = (100,100)

    best = dict([(k, round(min(averages[k][0],averages[k][1]),2)) for k in averages.keys()])

    for s in range_states:
        for a in range_actions:
            for l in range_specs:
                print(best[(s,a,l)])


# Plots data
def plot(data):

    pass