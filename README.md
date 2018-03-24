# Policy Gradient Implementation Using Gym
Policy gradient network is implemented using popular atari game, Pong Game.  "Policy gradients method involves running a policy for a while, seeing what actions lead to high rewards, increasing their probability through backpropagating gradients"

"Our Policy Gradient Neural Network, based heavily on Andrej’s solution, will do:

- take in images from the game and preprocess them (remove color, background, downsample etc.).
- use the Neural Network to compute a probability of moving up.
- sample from that probability distribution and tell the agent to move up or down.
- if the round is over (you missed the ball or the opponent missed the ball), find whether you won or lost.
- when the episode has finished(someone got to 21 points), pass the result through the backpropagation algorithm to compute the gradient for our weights.
- after 10 episodes have finished, sum up the gradient and move the weights in the direction of the gradient.
- repeat this process until our weights are tuned to the point where we can beat the computer. That’s basically it! Let’s start looking at how our code achieves this."


References: 

Policy Gradients Method: http://www.scholarpedia.org/article/Policy_gradient_methods

Policy Gradients from David Silver: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf

Pong Game Open AI Gym: https://gym.openai.com/envs/Pong-v0/

Open AI Gym: https://gym.openai.com/docs/

Andrej Karpathy (Deep Reinforcement Learning: Pong from Pixels): http://karpathy.github.io/2016/05/31/rl/

https://github.com/llSourcell/policy_gradients_pong

https://github.com/mrahtz/tensorflow-rl-pong

https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0



# PongGame

![ponggamescr](https://user-images.githubusercontent.com/10358317/37865988-2e288d68-2f95-11e8-8c38-762a156c20f4.png)
