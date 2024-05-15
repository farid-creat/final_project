import pybullet as p
import numpy as np

class WindVisualizer:
    def __init__(self, scale=0.5):
        # Create a cylinder representing the wind direction
        self.cylinder_id = p.createVisualShape(p.GEOM_CYLINDER,
                                               radius=0.05 * scale,
                                               length=1.0 * scale,
                                               rgbaColor=[1, 0, 0, 1],  # Red color
                                               visualFramePosition=[0, 0, 0.5 * scale])

        # Create a cone representing the wind direction
        self.cone_id = p.createVisualShape(p.GEOM_MESH,
                                            fileName="cone.obj",  # Cone mesh file
                                            rgbaColor=[1, 0, 0, 1],  # Red color
                                            meshScale=[0.1 * scale, 0.1 * scale, 0.3 * scale],  # Scale of the cone
                                            visualFramePosition=[0.5 * scale, 0, 0])  # Position of the cone

        # Combine the cylinder and cone into a single compound shape
        self.compound_id = p.createMultiBody(baseVisualShapeIndex=self.cylinder_id,
                                              basePosition=[0, 0, 0],
                                              baseOrientation=[0, 0, 0, 1])

    def update_wind_direction(self, wind_direction):
        # Convert wind direction vector to quaternion
        orientation = p.getQuaternionFromEuler([wind_direction[0], wind_direction[1], wind_direction[2]])

        # Update the position and orientation of the compound shape
        p.resetBasePositionAndOrientation(self.compound_id, [0, 0, 0], orientation)