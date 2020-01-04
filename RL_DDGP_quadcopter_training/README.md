# Teach A Quadcopter How To Fly

A semi-guided project from Udacity's Machine Learning Engineer Nanodegree Program.

## Project Overview

The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.
Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.
But it also comes at a price – the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!

### Deep Deterministic Policy Gradients (DDPG)
You can use one of many different algorithms to design your agent, as long as it works with continuous state and action spaces. One popular choice is Deep Deterministic Policy Gradients or DDPG. It is actually an actor-critic method, but the key idea is that the underlying policy function used is deterministic in nature, with some noise added in externally to produce the desired stochasticity in actions taken.
Let's develop an implementation of the algorithm presented in the original paper:

> Lillicrap, Timothy P., et al., 2015. Continuous Control with Deep Reinforcement Learning. [pdf](https://arxiv.org/pdf/1509.02971.pdf)

The two main components of the algorithm, the actor and critic networks can be implemented using most modern deep learning libraries, such as Keras or TensorFlow.

**DDPG: Actor (Policy) Model**
Note that the raw actions produced by the output layer are in a [0.0, 1.0] range (using a sigmoid activation function). So, we add another layer that scales each output to the desired range for each action dimension. This produces a deterministic action for any given state vector. A noise will be added later to this action to produce some exploratory behavior.
Another thing to note is how the loss function is defined using action value (Q value) gradients. These gradients will need to be computed using the critic model  and fed in while training. Hence it is specified as part of the "inputs" used in the training function.

**DDPG: Critic (Value) Model**
It is simpler than the actor model in some ways, but there some things worth noting. Firstly, while the actor model is meant to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the input layers.
These two layers can first be processed via separate "pathways" (mini sub-networks), but eventually need to be combined. This can be achieved, for instance, using the Add layer type in Keras (see Merge Layers).
The final output of this model is the Q-value for any given (state, action) pair. However, we also need to compute the gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model. This step needs to be performed explicitly, and a separate function needs to be defined to provide access to these gradients.

**DDPG: Agent**
We are now ready to put together the actor and policy models to build our DDPG agent. Note that we will need two copies of each model - one local and one target. This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, and is used to decouple the parameters being updated from the ones that are producing target values.
Notice that after training over a batch of experiences, we could just copy our newly learned weights (from the local model) to the target model. However, individual batches can introduce a lot of variance into the process, so it's better to perform a soft update, controlled by the parameter tau.

**Ornstein–Uhlenbeck Noise**
One last piece you need for all this to work properly is an appropriate noise model. We'll use a specific noise process that has some desired properties, called the Ornstein–Uhlenbeck process. It essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such that two consecutive samples are more likely to be closer together than further apart. In this sense, the process in Markovian in nature.
Why is this relevant to us? We could just sample from Gaussian distribution, couldn't we? Yes, but remember that we want to use this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly. Otherwise, we may not actually get anywhere! Imagine flicking a controller up-down, left-right randomly!
Besides the temporally correlated nature of samples, the other nice thing about the OU process is that it tends to settle down close to the specified mean over time. When used to generate noise, we can specify a mean of zero, and that will have the effect of reducing exploration as we make progress on learning the task.

**Replay Buffer**
Most modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples.

## Troubleshooting

**How long should I expect my agent to train?**
Depending on the random initialization of parameters, sometimes your agent may learn a task in the first 20-50 episodes, but you can expect most algorithms to be able to learn these tasks in 500-1000 episodes. It is also possible for them to get stuck in local minima, and never make it out (or make it out after a long time). It's possible for your training algorithm to take longer, e.g. depending on the learning rate parameter you choose.

If you see no significant progress by 1500-2000 episodes, then go back to the drawing board. First make sure your agent is able to solve a simpler continuous action problem, like Mountain Car or Pendulum (yes, your model sizes will be different based on the state/action space sizes, but this will at least verify your algorithm implementation is working).

Next, tweak the state and/or action space definitions in the task to make sure you're giving the agent all the information it needs (but not too much!). Finally (this is the part that takes the most number of iterations), try to change the reward function to give a better and more distinctive indication of what you want the agent to achieve. Play with the weights of the individual metrics, add in an extra bonus / penalty, try a non-linear mapping of the metrics (e.g. squared or log distance), etc. You may also want to scale or clip the rewards to some standard range, like -1.0 to +1.0. This avoids instability in training due to exploding gradients.

**My agent seems to be learning, but even after a lot of episodes, it still doesn't impress me. What should I do?**
Getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. It'll try to learn the optimal policy according to the reward function you've specified, but that may not address all aspects of the behavior desired by you. E.g. in the Takeoff task, if you don't penalize the agent for twisting around, and only care how high it's going, then it's perfectly fine for it to whirl around while going up!

Sometimes, your algorithm might not be tuned correctly for the environment. E.g. some environments may need more exploration or noise, some may need less. Try changing the these variables, and see if you get any improvement. But even if you have a stellar algorithm, it can sometimes take days or even weeks to learn a complex task!

This is why we ask you to plot the episode rewards. As long as it shows that the average reward per episode is generally increasing (even with some noise), that's okay. The final behavior of the agent doesn't need to be perfect.

**Am I allowed to use a Deep Q Network to solve this problem?**
Yes! But ... note that a Deep Q Network (DQN) is meant to solve discrete action space problems, whereas the quadcopter control problem has a continuous action space. So a discrete space algorithm is not advised here, but you could still use one by mapping the final output of your neural network model to the continuous action space appropriately. This may make it harder for the network to understand how its actions are related to each other, but it may also simplify the learning algorithm needed to train it.



