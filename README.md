# Balance
Cart and pole Reinforcement Learning (using `TensorFlow` and `OpenAI`). The goal is to balance the pole upright by only controlling the cart with left or right momentum. A policy gradient and reward function are used to penalise poor behaviour (falling over), and reward the desired behaviour (balancing in place).


#### 1. First Step

Primitive learning strategy

  1. If pole angle is negative, move left
  2. If pole angle is positive, move right

![cart pole alt text](https://github.com/lukexyz/Balance-RL/blob/master/img/001_left-right.gif?raw=true)

#### 2. Reinforcement Learning
* ```First attempt using a shallow neural net```

The cart initially starts well with quick error corrections, however once a terminal pole angle is reached the policy goes into an unstable cycle where it cannot correct itself.

![cart pole alt text](https://github.com/lukexyz/Balance-RL/blob/master/img/002_very_shallow_network.gif?raw=true)

#### 3. Improving the RL Network

* Add deeper layers to training network
* Use Google Colab platform for TITAN GPU support.

</P>

(more to come)
