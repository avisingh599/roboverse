from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TEST_OBJECTS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS)

import numpy as np


class MultiObjectEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 num_objects=1,
                 possible_objects=TRAIN_OBJECTS[:10],
                 **kwargs):
        assert isinstance(possible_objects, list)
        self.possible_objects = np.asarray(possible_objects)
        super().__init__(**kwargs)
        self.num_objects = num_objects

    def reset(self):
        chosen_obj_idx = np.random.randint(0, len(self.possible_objects),
                                           size=self.num_objects)
        self.object_names = tuple(self.possible_objects[chosen_obj_idx])

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_name in self.object_names:
            self.object_orientations[object_name] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_name] = OBJECT_SCALINGS[object_name]
        self.target_object = self.object_names[0]
        return super().reset()
