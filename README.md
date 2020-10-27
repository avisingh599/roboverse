# roboverse
A set of environments utilizing [pybullet](https://github.com/bulletphysics/bullet3) was simulation of robotic manipulation tasks. 

## Setup
I recommend using [conda](https://docs.anaconda.com/anaconda/install/) for setup:

```
conda create -n roboverse python=3.6
source activate roboverse
pip install -r requirements.txt
```

To test if things are working by visualizing a scripted robot policy, run the following command:

`python scripts/scripted_collect.py -n 100 -t 30 -e Widow250DoubleDrawerOpenNeutral-v0 -pl drawer_open_transfer -a drawer_opened_success --noise=0.1 --gui`

## If you want to dig into the code, start here:
`python roboverse/envs/widow250.py`

## Credit
Primary developers: [Avi Singh](https://www.avisingh.org/), Albert Yu, Jonathan Yang, [Michael Janner](https://people.eecs.berkeley.edu/~janner/), Huihan Liu, Gaoyue Zhou