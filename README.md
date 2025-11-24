prerequies:

install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
sudo apt update && sudo apt install libeigen3-dev libboost-all-dev liboctomap-dev
sudo apt install python-is-python3

uv venv

# build repository (to be able to use tools on src folder)
uv pip install -e .
```

launch Jupyter notebook :
```bash
source .venv/bin/activate

jupyter lab
```

or

```bash
uv run jupyter lab
```


## H1 experimentations

- `1- h1_launch_visualisation.ipynb`  
  Load h1 model in Meshcat and load a trajectory to be displayed. It helps me to create Python classes to abstract trajectory data treatment and loading robot.

- `2- h1_squat_following_given_trajectory.ipynb`  
  Load a trajectory and create an OCP problem with trajectory data as an input. The idea is to create an as simple as possible OCP that can make possible such trajectory with regard to model constrains.

- `3- h1_squat.ipynb`  
  Perform a squat using a simple OCP resolution problem. It is done with CoM following a given trajectory.


- `4- h1_squat_MCP.ipynb`  
  Implementation of MCP with loop on a vanishing horizon. I want it to be a first step to then use it as a controller in MuJoCo.


  ## launch H1 on Mujoco

  I used script provided by Unitree to launch the robot in Mujoco. Here are the steps to launch the simulation with a controller
  ```bash
  # First windows : launch controller.
  # go to 
  ```