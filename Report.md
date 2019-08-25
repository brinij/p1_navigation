[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Learning Algorithm : DQN

In 2015. a breakthrough algorithm designed by DeepMind was released. It is a learning algorith where agent plays video games better than a human player when it was given only a raw pixel data as an input. This agent is called Deep Q-network because at the heart of an agent is Deep Neural Network that acts as a function approximator. Following image ([source](https://www.nature.com/articles/nature14236)) illustrates the architecture of such a network where the first two layers are convolutional layers followed by a ReLu activation function, then one fully connected layer followed by a ReLu and last fully connected linear output layer that produced the vector of action values. 

<p align="center">
<img src="https://github.com/brinij/p1_navigation/blob/master/dnn_structure_dqn_nature.jpg" width="600">
</p>

Training such a network requires a lot of training data and even then it is not guaranteed to converge to the optimal value function. Network weights can oscilate or diverge due to high correlation between actions and states. In order to prevent this, researchers have came up with several techniques, where these two have shown the most effective: 

- **Experience Replay** : it is the act of sampling a small batch of tuples from the replay buffer in order to learn. The replay buffer contains a collection of experience tuples (S, A, R, S_next). The tuples are gradually added to the buffer as we are interacting with the environment. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.
- **Fixed Q Targets** : Using a separate network to estimate the TD (temporal difference) target. This target network has the same architecture as the function approximator but with frozen parameters. Every T steps (a hyperparameter) the parameters from the Q network are copied to the target network. This leads to more stable training because it keeps the target function fixed (for a while).

There are two main processes in the algorithm: 
- (SAMPLE) sample the environment by peroforming actions and store away the observed experienced tuples in a replay memory.
- (LEARN) randomly select the small batch of tuples from this memory and learn from that batch using a gradient descent update step

These two processes are not directly dependant on each other, so it is possible to perform multiple sampling steps, then one learning step. The rest of the algorithm supports these two steps. At the beginning it is necessary to initialize: an empty replay memory which has finite capacity N (circular queue) and weights of both neural networks (Q and target) with random values. Also, it is not possible to run learning step until there is a minimum number of samples in the replay memory. Detailed algorithm pseudocode is shown in the picture below:

<p align="center">
<img src="https://github.com/brinij/p1_navigation/blob/master/dqn_algorithm.png" width="600">
</p>


### Implementation Details

In the project file there are two Python files defining three classes. 
- In `model.py` is Python class `QNetwork` which defines the structure of Neural Network used in this project for solving the Banana environment. It has three linear layers where first two are followed by ReLu activation functions and the last one is linear. First two hidden layers have 64 nodes and the last one has the size of the action space which is 4.
- In `dqn_agent.py` 
