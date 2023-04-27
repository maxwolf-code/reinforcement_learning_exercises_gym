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
    state = process_observations_to_states(state)
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

      new_state = process_observations_to_states(new_state)

      if e%100 == 0:
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
    nEpisodes = 1001

    # Agent
    alpha_SARSA = 0.1 
    gamma_SARSA = 0.9

    alpha_Q = 0.1
    gamma_Q = 0.9

    epsilon_SARSA = 0.1
    epsilon_Q = 0.1

    #env.render()
    for idx_experiment in range(1, nExperiments+1):
        
        agent_SARSA = SARSA(nStates, nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
        reward_sums_SARSA, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_SARSA)   
        
        agent_Q = QLearning(nStates, nActions, alpha_Q, gamma_Q, epsilon=epsilon_Q)
        reward_sums_Q, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_Q)
        
        agent_DQ = DoubleQLearning(nStates, nActions, alpha_Q, gamma_Q, epsilon=epsilon_Q)
        reward_sums_Q, evst_SARSA, actionValueTable_history = runExperiment(nEpisodes, env, agent_DQ)


def process_observations_to_states(state):
   #TODO continous to state
   threshold_vel = 0
   threshold_pos = -0.5

   raw_pos, raw_vel = state

   pos = 0
   if raw_pos > threshold_pos:
      pos = 1
    
   vel = 0
   if raw_vel > threshold_vel:
      vel = 1

   state_str = str(pos) + str(vel)
   state_dict = {
      '00': 0,
      '01': 1,
      '10': 2,
      '11': 3,
   }
   discrete_state = state_dict[state_str]
   #print(state, discrete_state)
   return discrete_state 

if __name__ == '__main__':
    
    '''
    Action Space        Discrete(3); Accelerate : left, none, right

    Observation Shape   (2,)
    Observation High    [0.6 0.07]
    Observation Low     [-1.2 -0.07]
    0:  position of the car along the x-axis | -1.2; 0.6 
    1:  velocity of the car | -0.07; 0.07

    start position: 
    The position of the car is assigned 
    a uniform random value in [-0.6 , -0.4]. 
    The starting velocity of the car is always assigned to 0.
    
    '''

    env = gym.make('MountainCar-v0')
    env.reset()
    render_mode = 'human'

    #TODO quantisize continous inputs to create discrete states; test tabular methods

    nStates = 2**2 #depends on preprocessing of continuing states in process_observations
    nActions = 3 #Accelerate : left, none, right

    main(env, nStates, nActions)

