from roboverse.policies import *

PICK_PLACE_ENVS = ["Widow250PickPlace-v0"]
DRAWER_OPENING_ENVS = ["Widow250DrawerOpen-v0"]


env_to_policy_map = {
    frozenset(PICK_PLACE_ENVS): PickPlace,
    frozenset(DRAWER_OPENING_ENVS): DrawerOpen,
}


def instantiate_policy_class(env_name, env):
    for env_group in env_to_policy_map.keys():
        if env_name in env_group:
            return env_to_policy_map[env_group](env)
    raise NotImplementedError
