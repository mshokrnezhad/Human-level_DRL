# Human-level_DRL
The repository contains a Q-Learning agent which learns using the Deep Neural Network. 

The agent plays a simple game, named "PongNoFrameskip-v4", from the GYM Atari library.

The agent samples the env using a convolutional neural network, and learns through the approach provided in this paper: Human-level control through deep reinforcement
learning (doi:10.1038/nature14236) 

To run the learning procedure, run the main.py file, and to play a sample game using the trained model, run the test.py.

All of the dependencies can be installed using apt-get and pip. Just GYM should be cloned and installed form its git (https://github.com/openai/gym) instead of using pip. Its pip version has a bug in video-recorder.py which is corrected in its git repository. 
