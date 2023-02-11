[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

**Status:** Not really active (not a lot of time to develop and support this repo, breaking changes may occur). The program is checked in the case of `n_agents=2`

# DeepGame

Here you can see implementations of the following single- and multi-agent learning algorithms:
1) Vanilla Q-learning algorithm (Every agent behaves in optimal Q-value way)
2) Nash Q-learning (Agents try to achieve Nash-equilibrium)


## Prerequisites:
- Python>=3.6 
- Numpy>=1.14.1
- Nashpy>=0.0.17
- Matplotlib>=2.2.2

Also it is based on https://github.com/ml3705454/mapr2 repo and you have to install `mapr2` module:

1. Clone rllrb
  
 ```shell
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
sudo pip3 install -e .
 ```

 2. Intsall other dependencies
   
 ```shell
sudo pip3 install joblib,path.py,gtimer,theano,keras,tensorflow,gym, tensorflow_probability
 ```

 3. Intsall maci
   
 ```shell
cd maci
sudo pip3 install -e .
 ```


## Instructions for running the program
Run the file **run_grid_game.py** with the following command and default parameters:
```
python run_grid_game.py
```

If your want to change the size of the grid world, number of iterations or some hyperparametes you should look through these arguments of `run_grid_game.py`:

- `--grid_size`, the size of the grid world. Default one is 3,
- `--gamma`, the value of the Gamma in Bellman equation. Default one is 0.95,
- `--epsilon`, the size of the epsilon. Default one is 0.5,
- `--iterations`, the number of steps in the grid World. Default one is 1000,
- `--learning_rate`, the learning rate (alpha) value for Q-learning method. Default one is 0.9
- `--method`, the method to choose (Q-learning, Nash-Q). Default one is 'Q'
