# DRLND_Continuous_Control_P2
Project 2, Continuous Control, Deep Reinforcement Learning ND, Udacity

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

* The environment has twenty independent agents.
* Twenty Double-jointed arm which can move to target locations.
* Rewards:
    * +0.1 Each step agent's hand is in goal location.
* Observation: The observation for each agent is a vector of 33 elements containing 
the position, rotation, velocity and angular velocity of two arms and target location.
* Actions: Continuous actions of size 4. Two torque values for each joint.
* The problem is considered to be solved when the agent is able to receive 
an average reward (over 100 episodes, and over all 20 agents) of at least +30.

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
git clone https://github.com/yosoufe/DRLND_Continuous_Control_P2.git

# or with ssh setup:
git clone git@github.com:yosoufe/DRLND_Continuous_Control_P2.git
```

#### Prepare the Environment:
You need to download the environment from the links
[here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started)
according to your OS and pass the path of the environment in the following jupyter notebooks. 
I am using `Version 2: Twenty (20) Agents` environment.

#### How to train:
`Train Models DDPG.ipynb` is the notebook that does the training and save the trained model.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Reacher_Linux_2/Reacher.x86_64", no_graphics=True)
``` 

#### How to run the trained model:
`Demo Trained DDPG Model.ipynb` is loading the agent and run it in the environment with graphics.
You need to replace the pass of the simulator in one of the cells like bellow to point to where you 
placed your simulator:
```python
env = UnityEnvironment(file_name="../Reacher_Linux_2/Reacher.x86_64", no_graphics=False)
``` 

### Report:
You can find the report [here](report.md).