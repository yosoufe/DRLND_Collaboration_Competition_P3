# DRLND Collaboration and Competition, P3
Project 3, Collaboration and Competition, Deep Reinforcement Learning ND, Udacity

This is my submission to the 3rd project of [Deep Reinforcement Learning 
Nanodegree by Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project:
### Goal: 
The goal is to train an RL agent to receive the highest reward in the given environment.

### Environment:
he environment that is used here is custom version of 
[Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 
of [Unity ML-Agents Toolkit](https://unity3d.com/machine-learning). You can 
find the links to download the environment from 
[readme file in Udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

* The environment has two agents which are interacting with each other in form of tennis.
* Rewards:
    * Each agent that hits the ball over the net receives +0.1.
    * Each agent that lets the ball hit the ground or out of the bounds receives -0.01.
* Observation: The observation for each agent is a vector of 8 elements containing 
the position and velocity of the ball and the racket.
* Actions: Continuous actions of size 2 in range of [-1,1]. One for back and forthe, the other for jumping.
* Although the Udacity's project page says that the problem is considered to be solved when the agent is able to receive 
an average of maximum of reward over 100 episodes of at least +0.5. I consider the project solved when the average of 
all rewards is at least +0.5 which is a bit harder.

### Getting Started:

#### Python Dependencies:
* numpy: `pip install numpy`
* pytorch: [Installation Manual](https://pytorch.org/get-started/locally/)
* Jupyter notebook
* tqdm: `pip install tqdm` as progress bar.
* ipywidgets: `pip install ipywidgets` [Manual](https://ipywidgets.readthedocs.io/en/latest/user_install.html) 
for better visulaization of progress.
* mlagents_envs: `pip install mlagents-envs` or from source:
```bash
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents-envs
pip install .
```

#### Clone this Repo:
```bash
git clone https://github.com/yosoufe/DRLND_Collaboration_Competition_P3.git

# or with ssh setup:
git clone git@github.com:yosoufe/DRLND_Collaboration_Competition_P3.git
```

#### Prepare the Environment:
You need to download the environment from the links
[here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started)
according to your OS and pass the path of the environment in the following jupyter notebooks. 

#### How to train:
`Train Models DDPG.ipynb` and `Train Models MADDPG.ipynb` are the notebook that does the training 
and save the trained model using two algorithm `DDPA` and `MADDPG`.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64", no_graphics=True)
``` 

#### How to run the trained model:
`Demo Trained DDPG Model.ipynb` and `Demo Trained MADDPG.ipynb` are loading the agents 
and run it in the environment with graphics.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64", no_graphics=False)
``` 

### Report:
You can find the report [here](report.md).