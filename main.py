import gym                                                                                                              #importando a biblioteca gym (para instalar, usar: pip install gym
from stable_baselines3 import PPO                                                                                       #importando o Algoritmo de Aprendizado Reforçado A2C do stable_baselines3 (para instalar, usar: pip install stable-baselines3

aprendendo = False
total_timesteps=10000                                                                                                   #definindo os timesteps fora da função para que possa ser possivel a referenciação posterior
if aprendendo:
    env = gym.make('ALE/MsPacman-v5',)                                                                                  #Criando o environment do MsPacman, utilizando a versão 5 providenciada pelo ALE e pelo gym
    # Criando o modelo de machine learning utilizando o algoritmo PPO, bem como o environment criado anteriormente, os parametros foram definidos usando o paper disponivel em https://arxiv.org/abs/1707.06347
    model = PPO("MlpPolicy", env, verbose=1,
                device="cuda",
                gamma=0.99,
                n_epochs=3,
                batch_size=32*8,
                gae_lambda=0.95,
                vf_coef=1,
                ent_coef=0.01,)
    model.learn(total_timesteps=total_timesteps)                                                                        #Treinando o modelo pelos timesteps definidos acima
    model.save("mspacman_models/ppo_mspacman"+str(total_timesteps))                                                     #Salvando o modelo
else:
    env = gym.make('ALE/MsPacman-v5',render_mode='human')
    model = PPO.load("mspacman_models/ppo_mspacman"+str(total_timesteps))                                               #Carregando o modelo para a visualização do resultado
    dones= False
    obs = env.reset()                                                                                                   #Resetando o environment
    while not dones:                                                                                                         #Deixando o modelo jogar o jogo até o seu fim
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
