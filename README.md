# Policy Gradient Implementation Using Gym and Tensorflow
Policy gradient network is implemented using popular atari game, Pong Game.  "Policy gradients method involves running a policy for a while, seeing what actions lead to high rewards, increasing their probability through backpropagating gradients". If there is a large scale problems which is aimed to solve, a type of function approximator should be used. In this problem, a neural network is used as function approximator. There are too many states and/or actions to store in memory, so look up table can not be used.

![policies](https://user-images.githubusercontent.com/10358317/37866273-95673610-2f99-11e8-989f-98b2b77eef54.png)


Andrej Karpathy (Deep Reinforcement Learning: Pong from Pixels): http://karpathy.github.io/2016/05/31/rl/

Policy Gradient Neural Network, based on Andrej’s solution, will do:

- take in images from the game and "preprocess" them (remove color, background, etc).
- use the TF NN to compute a probability of moving up or down.
- sample from that probability distribution and tell the agent to move up or down.
- if the round is over, find whether you won or lost.
- when the episode has finished, pass the result through the backpropagation algorithm to compute the gradient for weights.
- after each episodes have finished, sum up the gradient and move the weights in the direction of the gradient.
- repeat this process until weights are tuned to the point

# PongGame Experiment Results
After a period time, scores are getting better.

![ponggamescoreep_1](https://user-images.githubusercontent.com/10358317/37867469-1fcd5144-2faa-11e8-9ac6-665ef332519e.png)


![ep110_630_result](https://user-images.githubusercontent.com/10358317/37867530-00dc106c-2fab-11e8-85be-fd41107abc5c.png)


**References**: 

Policy Gradients Method: http://www.scholarpedia.org/article/Policy_gradient_methods

Policy Gradients from David Silver: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf

Pong Game Open AI Gym: https://gym.openai.com/envs/Pong-v0/

Open AI Gym: https://gym.openai.com/docs/

https://github.com/llSourcell/policy_gradients_pong

https://github.com/mrahtz/tensorflow-rl-pong

https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0






