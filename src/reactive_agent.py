import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
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
        print("Aterragem bem sucedida!")
        return True

    print("Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
# observation = [x, y, vx, vy, theta, vtheta, left_leg, right_leg]


LIM_X = 0.20
LIM_TH = np.deg2rad(7)
LIM_VY_FALL = -0.55

#detetar mais fast oq se esta a passar com o vento
LIM_VX = 0.10

LIM_Y_NEAR = 0.25
LIM_VX_WIND = 0.10
LIM_TH_SAFE = np.deg2rad(10)



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

# percepções relacionadas com o vento
def p_drifting_right(obs):
    return obs[2] > LIM_VX_WIND and abs(obs[4]) < LIM_TH_SAFE


def p_drifting_left(obs):
    return obs[2] < -LIM_VX_WIND and abs(obs[4]) < LIM_TH_SAFE



# Actions
BOOST_MAIN_SIDE = 0.12
BOOST_MAIN_NEAR = 0.18
BOOST_WIND = 0.15
MAIN_FOR_FALL = 0.88

def a_idle():
    return np.array([-1.0, 0.0], dtype=np.float32)

def a_main(level=1.0):
    level = float(np.clip(level, 0.0, 1.0))
    return np.array([level, 0.0], dtype=np.float32)

def a_push_left(boost=0.0):
    return np.array([np.clip(boost, -1.0, 1.0), -1.0], dtype=np.float32)

def a_push_right(boost=0.0):
    return np.array([np.clip(boost, -1.0, 1.0), 1.0], dtype=np.float32)


def reactive_agent(observation):
    # 1- terminou 
    if p_landed(observation):
        return a_idle()

    # 2- muito perto do chão e a cair depressa
    if p_near_ground(observation) and p_falling_fast(observation):
        if p_tilted_left(observation):
            return a_push_right(BOOST_MAIN_NEAR)
        if p_tilted_right(observation):
            return a_push_left(BOOST_MAIN_NEAR)
        return a_main(MAIN_FOR_FALL)

    # 3- cair fast
    if p_falling_fast(observation):
        if p_tilted_left(observation):
            return a_push_right(BOOST_MAIN_SIDE)
        if p_tilted_right(observation):
            return a_push_left(BOOST_MAIN_SIDE)
        return a_main(MAIN_FOR_FALL)

    # 4- corrige a inclinação
    if p_tilted_left(observation):
        return a_push_right(BOOST_MAIN_SIDE)

    if p_tilted_right(observation):
        return a_push_left(BOOST_MAIN_SIDE)
    
    #5- corrigir deriva do vento
    if p_drifting_right(observation):
        return a_push_left(BOOST_WIND)

    if p_drifting_left(observation):
        return a_push_right(BOOST_WIND)

    # 6- corrigir horizontal em deriva
    if p_moving_right(observation):
        return a_push_left(BOOST_MAIN_SIDE)

    if p_moving_left(observation):
        return a_push_right(BOOST_MAIN_SIDE)

    # 7- corrigir horizontal por posição acumulada
    if p_near_ground(observation) and p_too_right(observation):
        return a_push_left(BOOST_MAIN_NEAR)

    if p_near_ground(observation) and p_too_left(observation):
        return a_push_right(BOOST_MAIN_NEAR)

    if p_too_right(observation):
        return a_push_left(BOOST_MAIN_SIDE)

    if p_too_left(observation):
        return a_push_right(BOOST_MAIN_SIDE)

    # 8- nada a fazer
    return a_idle()

    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    

success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)