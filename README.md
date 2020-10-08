# Start here

`python roboverse/envs/widow250.py`

## Scripted data collection

`python scripts/scripted_collect_parallel.py -n 5000 -p 20 -t 25 -e Widow250PickPlace-v0 -pl grasp -a grasp_success -d testing`

`python scripts/scripted_collect.py -n 100 -t 25 -e Widow250PickPlace-v0 -pl grasp -a grasp_success --gui`

`python scripts/scripted_collect.py -n 100 -t 30 -e Widow250DoubleDrawerOpenNeutral-v0 -pl drawer_open_transfer -a drawer_opened_success --noise=0.1 --gui`

## Pulling all submodules (like bullet-objects)

First time: `git submodule update --init --recursive`

Subsequent updates: `git submodule update --recursive --remote`

## TODO
- [ ] DrawerOpenSuboptimal is actually just DrawerClose. This should be renamed, and the code should be shared between the two scripts (since they only differ in one action).  
- [ ] Add obstacle object and box/tray to DoubleDrawer env