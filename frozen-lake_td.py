import gym
import numpy as np
#import matplotlib.pyplot as plt

from IRL.agents.TemporalDifferenceLearning import SARSA, QLearning, ExpectedSARSA, DoubleQLearning

def runExperiment(nEpisodes, env, agent):
  """Train and test the agent in the given Environment for the given Episodes

    Args:
        nEpisodes (int): number of Episoeds to train
        env (gym env): Evironment to train/test the agent in
        agent (agent): the agent to train

    Returns:
        list : reward_sums
        list: episodesvstimesteps 
        list: actionValueTable_history
  """
  reward_sums = []
  episodesvstimesteps = []
  #timesteps = 0
  actionValueTable_history = []
  for e in range(nEpisodes):
    timesteps = 0
    if(e%100==0):
      print(agent.getName(), "Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state) 
    done = False
    reward_sums.append(0.0)
    while not done:
      timesteps += 1
      
      experiences = [{}]
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done, info = env.step(action)

      if e%20 == 0:
        env.render()
      
      #print("NEW STATE",new_state, reward, done)
      new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)
      
      agent.update(experiences[-2:])
    
      state = new_state
      
      if(agent.getName()=="SARSA"):
        action = new_action
      else:
        action = agent.selectAction(state)
      
      #episodesvstimesteps.append([e,timesteps])
      reward_sums[-1] += reward
    episodesvstimesteps.append([e,timesteps])

    
    #store table data
    if e%50==0:
        if agent.getName()=='Double Q-Learning':
          avg_action_table = np.mean( np.array([ agent.actionValueTable_1.copy(),agent.actionValueTable_2.copy() ]), axis=0 )
          actionValueTable_history.append(avg_action_table.copy())
        else:
          actionValueTable_history.append(agent.actionValueTable.copy())

    if e%100==0:
        title = agent.getName()+' Episode:'+str(e)
        print(title, 'reward_sums=',reward_sums[-1])
        #env.render(name = title)

      
  return reward_sums, np.array(episodesvstimesteps), actionValueTable_history


def main(env, nStates, nActions):
    """Main to test agents in environments
    """
    nExperiments = 1
    nEpisodes = 101

    # Agent
    alpha_SARSA = 0.1 
    gamma_SARSA = 0.9

    alpha_Q = 0.1
    gamma_Q = 0.9

    epsilon_SARSA = 0.01
    epsilon_Q = 0.01

    #env.render()
    for idx_experiment in range(1, nExperiments+1): #TODO Add parameter to store path or names
        
        agent_SARSA = SARSA(nStates, nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
        reward_sums_SARSA, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_SARSA)   
        
        agent_Q = QLearning(nStates, nActions, alpha_Q, gamma_Q, epsilon=epsilon_Q)
        reward_sums_Q, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_Q)
        
        agent_DQ = DoubleQLearning(nStates, nActions, alpha_Q, gamma_Q, epsilon=epsilon_Q)
        reward_sums_Q, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_DQ)


if __name__ == '__main__':
    #Setup Environment
    '''
    'desc': Used to specify custom map for frozen lake. For example
    'is_slippery': True/False. If True will move in intended direction with
        probability of 1/3 else will move in either perpendicular direction with
        equal probability of 1/3 in both directions.
            For example, if action is left and is_slippery is True, then:
            - P(move left)=1/3
            - P(move up)=1/3
            - P(move down)=1/3

    '''
    desc = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
  ]
    
    nStates = 8**2
    nActions = 4

    env = gym.make('FrozenLake-v1', is_slippery=True, desc=desc)
    env.reset()
    render_mode = 'human'

    main(env, nStates, nActions)


    #TODO exercises
    # How to compare the agents performance
    # How to analyze the training process
    # How can different maps created with increasing complexity

