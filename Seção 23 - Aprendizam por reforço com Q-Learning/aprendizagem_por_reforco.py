from time import time
import gym
import random
import numpy as np

env = gym.make("Taxi-v3").env
env.reset()
env.render()

# Movimento: 0 = Sul, 1 = Norte, 2 = Leste, 3 = Oeste, 4 = Pegar, 5 = Deixar tem na documentação. 
print(env.action_space)

# 4 destinos
print(env.observation_space) #Ações disponiveis

#Ações possiveis
len(env.P)
#Executar as ações possiveis, em cada estado. 
env.P[484]
# {0: [(1.0, 484, -1, False)],
#  1: [(1.0, 384, -1, False)],
#  2: [(1.0, 484, -1, False)],
#  3: [(1.0, 464, -1, False)],
#  4: [(1.0, 484, -10, False)],
#  5: [(1.0, 484, -10, False)]}


# -------------TREINAMENTO --------------
# Formula da diferença temporal (Q-learning): Qt(s,a) = Qt-1(s,a) + alfaTDt(a,s)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table.shape

# 1-10% 3-90%
# exploration / exploitation
#%%time :Quando executar, tira essa comentario. Essa linha serve para imprimir os episodios
from IPython.display import clear_output

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(100000):
    estado = env.reset()

    penalidades, recompensa = 0, 0
    done = False
    while not done:
        # Exploração
        if random.uniform(0,1) < epsilon:
            acao = env.action_space.sample()
        # Exploitation
        else:
            acao = np.argmax(q_table[estado])

        proximo_estado, recompensa, done, info = env.step(acao)

        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])

        q_novo = (1- alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo

        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado

    if i % 100 == 0:
        clear_output(wait=True)
        print("Episodio: ", i)

print("Treinamento, concluido" ) 

q_table[68] #Atualizada e treinada 

env.reset()
env.render()

env.encode(0, 3, 2, 0) #Descobrindo o estado de momento do taxi

env.step(0)
env.render()

# ------------ AVALIAÇÃO -----------
total_penalidades = 0
episodios = 50
frames = []

for _ in range (episodios):
    estado = env.reset()
    penalidades, recompensas = 0, 0
    done = False
    while not done:
        acao = np.argmax(q_table[estado])
        estado, recompensa, done, info = env.step(acao)

        if recompensa == -10:
            penalidades += 1

        frames.append({
            'frame': env.render(mode="ansi"),
            "state": estado,
            "action": acao,
            "reward": recompensa
        })
    total_penalidades += penalidades
print(f"Episódios {episodios}")
print(f"Penalidades {total_penalidades}")

frames[0]

from time import sleep
for frame in frames:
    clear_output(wait=True)
    print(frame["frame"])
    print("Estado", frame["state"])
    print("Ação", frame["action"])
    print("Recopença", frame["reward"])
    sleep(.5)
# %%
