import gym
import numpy as np
#import matplotlib.pyplot as plt

from IRL.agents.TemporalDifferenceLearning import nStepSARSA, nStepTreeBackup

def runExperiment_NStep(nEpisodes, env, agent):
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
  actionValueTable_history = []
  for e in range(nEpisodes):
    timesteps = 0
    if(e%10==0):
      print("Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state)    
    done = False
    experiences = [{}]
    reward_sums.append(0.0)
    while not done:
      timesteps += 1
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done, info = env.step(action)
      
      #print("State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
      
      if e%20 == 0:
        env.render()

      new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['state'] = new_state
      xp['reward'] = reward
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)
      
      agent.update(experiences[-2:])
      
      if(agent.getName()=="SARSA"):
        action = new_action
      else:
        action = agent.selectAction(new_state)
      
      state = new_state
      
      reward_sums[-1] += reward
    episodesvstimesteps.append([e,timesteps])

     #store table data
    if e%10==0:
        actionValueTable_history.append(agent.actionValueTable.copy())

    if e%10==0:
        title = agent.getName()+' Episode:'+str(e)
        print(title, 'reward_sums=',reward_sums[-1])
        #env.render(name = title)

      
  return reward_sums, np.array(episodesvstimesteps), actionValueTable_history


def main(env, nStates, nActions):
    """Main to test agents in environments
    """
    nExperiments = 1
    nEpisodes = 1001

    # Agent
    alpha_nStepSARSA = 0.1
    gamma_nStepSARSA = 0.9 
    n_nStepSARSA = 5  
    epsilon_nStepSARSA = 0.1

    #env.render()
    for idx_experiment in range(1, nExperiments+1):
        agent = nStepSARSA(nStates, nActions, alpha_nStepSARSA, gamma_nStepSARSA, n_nStepSARSA, epsilon=epsilon_nStepSARSA)
        reward_sums, evst, actionValueTable_history = runExperiment_NStep(nEpisodes, env, agent)
        

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

    env = gym.make('FrozenLake-v1', is_slippery=False, desc=desc)
    env.reset()
    render_mode = 'human'

    main(env, nStates, nActions)


    #TODO exercises
    # How to compare the agents performance
    # How to analyze the training process
    # How can different maps created with increasing complexity

