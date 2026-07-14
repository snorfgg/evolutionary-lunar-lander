import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()#num de cores do pc [cuidado com o numero de processos, pode ser que o pc nao aguente muitos processos rodando ao mesmo tempo]
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
#^^nao mexer no que esta a cima, posso mudar o shape
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 250
NUMBER_OF_GENERATIONS = 200
TOURNAMENT_SIZE = 5
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.05


ELITE_SIZE = 4

"""Metricas passadas"""
LIM_X = 0.20
LIM_TH = np.deg2rad(7)
LIM_VY_FALL = -0.55
LIM_VX = 0.10
LIM_Y_NEAR = 0.25
LIM_VX_WIND = 0.10
LIM_TH_SAFE = np.deg2rad(10)


def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False


def p_landed(obs):
    return obs[6] == 1 and obs[7] == 1

def p_too_left(obs):
    return obs[0] < -LIM_X

def p_too_right(obs):
    return obs[0] > LIM_X

def p_tilted_left(obs):
    return obs[4] > LIM_TH

def p_tilted_right(obs):
    return obs[4] < -LIM_TH

def p_falling_fast(obs):
    return obs[3] < LIM_VY_FALL

def p_near_ground(obs):
    return obs[1] < LIM_Y_NEAR

def p_moving_right(obs):
    return obs[2] > LIM_VX

def p_moving_left(obs):
    return obs[2] < -LIM_VX

def p_drifting_right(obs):
    return obs[2] > LIM_VX_WIND and abs(obs[4]) < LIM_TH_SAFE

def p_drifting_left(obs):
    return obs[2] < -LIM_VX_WIND and abs(obs[4]) < LIM_TH_SAFE

def objective_function(observation_history):
    final_obs = observation_history[-1]

    x = final_obs[0]
    vx = final_obs[2]
    vy = final_obs[3]
    theta = final_obs[4]
    left_leg = final_obs[6]
    right_leg = final_obs[7]

    fitness = 100.0

    # Estado final simples
    fitness -= abs(x) * 70.0
    fitness -= abs(vx) * 35.0
    fitness -= max(0, -vy) * 55.0
    fitness -= abs(theta) * 60.0

    # Recompensa por tocar com as pernas
    fitness += (left_leg + right_leg) * 40.0

    # Última parte do voo: aterragem controlada
    last_part = observation_history[int(len(observation_history) * 0.65):]

    avg_x = sum(abs(obs[0]) for obs in last_part) / len(last_part)
    avg_vx = sum(abs(obs[2]) for obs in last_part) / len(last_part)
    avg_fall = sum(max(0, -obs[3]) for obs in last_part) / len(last_part)
    avg_theta = sum(abs(obs[4]) for obs in last_part) / len(last_part)

    fitness -= avg_x * 50.0
    fitness -= avg_vx * 35.0
    fitness -= avg_fall * 35.0
    fitness -= avg_theta * 35.0

    # penatli que obriga ao uso do motor
    wrong_drift = 0.0
    for obs in last_part:
        x_t = obs[0]
        vx_t = obs[2]

        # com s e vx igauis estamos a afastar-nos
        if x_t * vx_t > 0:
            wrong_drift += abs(x_t * vx_t)

    fitness -= wrong_drift * 120.0

    # Penalização moderada se estiver longe do centro perto do chão
    near_ground_bad = 0.0
    for obs in observation_history:
        if obs[1] < 0.35:
            near_ground_bad += abs(obs[0]) * 2.0
            near_ground_bad += abs(obs[2]) * 2.0
            near_ground_bad += max(0, -obs[3]) * 2.0

    fitness -= near_ground_bad

    success = check_successful_landing(final_obs)

    if success:
        fitness += 500.0

        # distinguir os bons dos que tiveram sorte
        fitness += max(0, 1.0 - abs(x) * 5.0) * 80.0
        fitness += max(0, 1.0 - abs(vx) * 5.0) * 60.0

    return fitness, success


def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        fit, success = simulate(ind['genotype'], seed=None, env=env)
        ind['fitness'] = fit
        ind['success'] = success
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    best = max(tournament, key=lambda x: x['fitness'])
    return copy.deepcopy(best)

def crossover(p1, p2):
    c1, c2 = sorted(random.sample(range(1, GENOTYPE_SIZE), 2))
    g1 = p1['genotype']
    g2 = p2['genotype']
    child_genotype = g1[:c1] + g2[c1:c2] + g1[c2:]
    return {'genotype': child_genotype, 'fitness': None}

def mutation(p):
    for i in range(len(p['genotype'])):
        if random.random() < PROB_MUTATION:
            p['genotype'][i] += random.gauss(0, STD_DEV)
    p['fitness'] = None
    return p  
    
def survival_selection(population, offspring):#Guardar as melhores, e ir substituindo as piores, ou seja, fazer uma junção das duas populações e escolher os melhores
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    stats = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        avg_fitness = sum(ind['fitness'] for ind in population) / len(population)
        success_rate = sum(ind['success'] for ind in population) / len(population)
        best_fitness = population[0]['fitness']

        stats.append((gen, best_fitness, avg_fitness, success_rate))
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Gen {gen}: best={best_fitness:.3f}, avg={avg_fitness:.3f}, success_rate={success_rate:.2%}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests, stats

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests


def plot_stats(stats, run_id):
    generations = [s[0] for s in stats]
    best_fitness = [s[1] for s in stats]
    avg_fitness = [s[2] for s in stats]
    success_rate = [s[3] for s in stats]

    plt.figure()
    plt.plot(generations, best_fitness, label="Best fitness")
    plt.plot(generations, avg_fitness, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness evolution - Run {run_id}")
    plt.legend()
    plt.savefig(f"fitness_run_{run_id}.png")
    plt.close()

    plt.figure()
    plt.plot(generations, success_rate)
    plt.xlabel("Generation")
    plt.ylabel("Success rate")
    plt.title(f"Success rate evolution - Run {run_id}")
    plt.savefig(f"success_run_{run_id}.png")
    plt.close()


if __name__ == '__main__':

    # Pick a setting from below
    # --to evolve the controller--
    evolve = True
    render_mode = None

    # --to test the evolved controller without visualisation--
    # evolve = False
    # render_mode = None

    # --to test the evolved controller with visualisation--
    # evolve = False
    # render_mode = 'human'

    if evolve:
        n_runs = 5
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]

        for i in range(n_runs):
            random.seed(seeds[i])

            bests, stats = evolution()

            # log com os melhores individuos
            log_path = os.path.join(os.getcwd(), f'log_run_{i}.txt')
            with open(log_path, 'w', encoding='utf-8') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

            # log com estatisticas por geracao
            stats_path = os.path.join(os.getcwd(), f'stats_run_{i}.csv')
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("generation,best_fitness,avg_fitness,success_rate\n")
                for gen, best_fit, avg_fit, succ_rate in stats:
                    f.write(f"{gen},{best_fit},{avg_fit},{succ_rate}\n")

            # graficos
            plot_stats(stats, i)

            print(f'Guardado: {log_path}')
            print(f'Guardado: {stats_path}')
            print(f'Guardado: fitness_run_{i}.png')
            print(f'Guardado: success_run_{i}.png')

    else:
        filename = 'log_run_0.txt'

        bests = load_bests(filename)
        b = bests[-1]

        SHAPE = b[1]
        ind = b[2]
        ind = {'genotype': ind, 'fitness': None}

        ntests = TEST_EPISODES

        fit, success = 0, 0
        for i in range(1, ntests + 1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed=None)
            fit += f
            success += s

        print("Average fitness:", fit / ntests)
        print("Success rate:", success / ntests)