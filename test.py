
import ast

# WRITING SH FILES ##################

# filenames = []

# # -epsilon 0.0001 -maxiters 100000 -timeout 18000

# for t in [6,7,8,9,10]:    
#     for s in [1,2,3,4,5,6,7,8,9,10]:
#         for a in [1,2,3,4,5]:
#             for l in [1,2]:

#                 if (s + a >= 10) or (s >= 7):
#                     mb = 16384
#                     gb = 16
#                 else:
#                     mb = 4096
#                     gb = 4

#                 # if l == 1:
#                 #     num_props = '2'
#                 # else:
#                 #     num_props = '2,3'
                
#                 filename = 'almanac-{}-{}-{}-{}.sh'.format(s,a,l,t)
#                 with open(filename, 'w') as f:

#                     f.write("#!/bin/bash\n\
# #SBATCH --partition=htc\n\
# #SBATCH --job-name=2almanac{0}{1}{2}{3}\n\
# #SBATCH --output=/home/hert5888/almanac/logs/{0}-{1}-{2}-{3}.out\n\
# #SBATCH --error=/home/hert5888/almanac/logs/{0}-{1}-{2}-{3}.err\n\
# #SBATCH --time=24:00:00\n\
# #SBATCH --mem={4}\n\
# module load python/anaconda3/2019.03\n\
# source activate /home/hert5888/almanac/arc_venv\n\
# python /home/hert5888/almanac/experiments.py {0} {1} {2} {3}\n\
# sleep 240\n\
# module load prism/4.4-beta\n\
# prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/markov_games/mmg/prism_models/{0}-{1}-{2}-{3}.prism \
# /home/hert5888/almanac/specs/mmg/{0}-{1}-{2}-{3}.props -prop 1 > \
# /home/hert5888/almanac/results/mmg/{0}-{1}-{2}-{3}-true.txt\n\
# sleep 60\n\
# prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/markov_games/mmg/prism_models/{0}-{1}-{2}-{3}-policy.prism \
# /home/hert5888/almanac/specs/mmg/{0}-{1}-{2}-{3}.props -prop 2 > \
# /home/hert5888/almanac/results/mmg/{0}-{1}-{2}-{3}-policy-0.txt\n\
# sleep 60\n\
# prism -cuddmaxmem {5}g -javamaxmem {5}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/markov_games/mmg/prism_models/{0}-{1}-{2}-{3}-policy-det.prism \
# /home/hert5888/almanac/specs/mmg/{0}-{1}-{2}-{3}.props -prop 2 > \
# /home/hert5888/almanac/results/mmg/{0}-{1}-{2}-{3}-policy-det-0.txt\n\
# sleep 60\n".format(s,a,l,t,mb,gb))

#                     if l == 2:
#                         f.write("prism -cuddmaxmem {4}g -javamaxmem {4}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/markov_games/mmg/prism_models/{0}-{1}-{2}-{3}-policy.prism \
# /home/hert5888/almanac/specs/mmg/{0}-{1}-{2}-{3}.props -prop 3 > \
# /home/hert5888/almanac/results/mmg/{0}-{1}-{2}-{3}-policy-1.txt\n\
# sleep 60\n\
# prism -cuddmaxmem {4}g -javamaxmem {4}g -epsilon 0.0001 -maxiters 100000 -timeout 18000 /home/hert5888/almanac/environments/markov_games/mmg/prism_models/{0}-{1}-{2}-{3}-policy-det.prism \
# /home/hert5888/almanac/specs/mmg/{0}-{1}-{2}-{3}.props -prop 3 > \
# /home/hert5888/almanac/results/mmg/{0}-{1}-{2}-{3}-policy-det-1.txt\n".format(s,a,l,t,gb))

#                     f.write("conda deactivate")

#                 filenames.append(filename)

# with open('multi2.sh', 'w') as f:
#     f.write("#!/bin/bash\n")
#     for fn in filenames:
#         f.write("sbatch " + fn + "\n")



# GETTING RESULTS ##################


results = dict()


for t in [1,2,3,4,5,6,7,8]:    
    for s in [1,2,3,4,5,6,7,8,9,10]:
        for a in [1,2,3,4,5]:
            for l in [1,2]:
                
                key = (s,a,l,t)

                if key == (1,1,1,1) or key == (1,1,2,1) or key == (1,2,1,1) or key == (1,3,1,1) or key == (1,4,1,1) or key == (9,2,2,1) or key == (7,5,2,2) or key == (8,3,2,3) or key == (6,5,2,4) or key == (7,4,2,5) or key == (9,2,1,5) or key == (3,5,2,6) or key == (4,4,2,6) or key == (4,5,2,6) or key == (5,4,2,6) or key == (5,5,2,6) or key == (6,4,2,6) or key == (6,5,2,6) or key == (7,4,2,6) or key == (7,5,2,6) or key == (8,2,2,6) or key == (9,5,2,6) or key == (10,5,2,6) or key == (3,5,2,7) or key == (4,4,2,7) or key == (4,5,2,7) or key == (5,5,2,7) or key == (6,4,2,7) or key == (6,5,2,7) or key == (7,3,2,7) or key == (7,4,2,7) or key == (7,5,2,7) or key == (8,3,2,7) or key == (10,3,2,7) or key == (10,5,2,7) or key == (4,5,2,8) or key == (5,5,2,8) or key == (6,4,2,8) or key == (6,5,2,8) or key == (7,4,2,8) or key == (8,3,2,8) or key == (8,5,2,8) or key == (9,1,2,8) or key == (9,5,2,8) or key == (10,2,2,8) or key == (10,5,2,8):
                    results[key] = [None, None, None]
                    continue
                        

                key_string = 'final_results/{}-{}-{}-{}'.format(s,a,l,t)

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

print("done!")

weights = dict()

for t in [1,2,3,4,5,6,7,8]:    
    for s in [1,2,3,4,5,6,7,8,9,10]:
        for a in [1,2,3,4,5]:
            for l in [2]:
                
                key = (s,a,l,t)

                if key == (1,1,1,1) or key == (1,1,2,1) or key == (1,2,1,1) or key == (1,3,1,1) or key == (1,4,1,1) or key == (9,2,2,1) or key == (7,5,2,2) or key == (8,3,2,3) or key == (6,5,2,4) or key == (7,4,2,5) or key == (9,2,1,5) or key == (3,5,2,6) or key == (4,4,2,6) or key == (4,5,2,6) or key == (5,4,2,6) or key == (5,5,2,6) or key == (6,4,2,6) or key == (6,5,2,6) or key == (7,4,2,6) or key == (7,5,2,6) or key == (8,2,2,6) or key == (9,5,2,6) or key == (10,5,2,6) or key == (3,5,2,7) or key == (4,4,2,7) or key == (4,5,2,7) or key == (5,5,2,7) or key == (6,4,2,7) or key == (6,5,2,7) or key == (7,3,2,7) or key == (7,4,2,7) or key == (7,5,2,7) or key == (8,3,2,7) or key == (10,3,2,7) or key == (10,5,2,7) or key == (4,5,2,8) or key == (5,5,2,8) or key == (6,4,2,8) or key == (6,5,2,8) or key == (7,4,2,8) or key == (8,3,2,8) or key == (8,5,2,8) or key == (9,1,2,8) or key == (9,5,2,8) or key == (10,2,2,8) or key == (10,5,2,8):
                    weights[key] = (0, 0)
                    continue


                key_string = 'final_weights/{}-{}-{}-{}.weights'.format(s,a,l,t)
                with open(key_string) as f:
                    for i, line in enumerate(f):
                        if i == 0:
                            w1 = float(line)
                        if i == 1:
                            w2 = float(line)
                    weights[key] = (w1, w2)
     

averages = dict()

for s in [1,2,3,4,5,6,7,8,9,10]:
    for a in [1,2,3,4,5]:
        for l in [1,2]:

            key = (s,a,l)


            trues = [results[s,a,l,t][0] for t in [1,2,3,4,5,6,7,8]]
            pols = [results[s,a,l,t][1] for t in [1,2,3,4,5,6,7,8]]
            dets = [results[s,a,l,t][2] for t in [1,2,3,4,5,6,7,8]]

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
                
print("done!")


for k in averages.keys():
    if averages[k][0] == None:
        averages[k] = (100,100)

best = dict([(k, round(min(averages[k][0],averages[k][1]),2)) for k in averages.keys()])

print("now")

for l in [1,2]:
    for a in [1,2,3,4,5]:
        for s in [1,2,3,4,5,6,7,8,9,10]:
            print(best[(s,a,l)])