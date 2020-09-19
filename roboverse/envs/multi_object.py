from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, TEST_OBJECTS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS)

import numpy as np


class MultiObjectEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 # *args,
                 num_objects=1,
                 use_test_objects=False,
                 possible_train_objects=TRAIN_OBJECTS[:10],
                 possible_test_objects=TEST_OBJECTS[:10],
                 **kwargs):

        self.use_test_objects = use_test_objects  # True when doing evaluation

        assert isinstance(possible_train_objects, list)
        assert isinstance(possible_test_objects, list)

        if self.use_test_objects:
            self.possible_objects = np.asarray(possible_test_objects)
        else:
            self.possible_objects = np.asarray(possible_train_objects)

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
