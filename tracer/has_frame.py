# Represents an object that is locatable in 3D space with 6 degrees of freedom.

import numpy as N
import pivy.coin as coin

class HasFrame(object):
    """
    Each surface has a location vector and a rotation matrix. Together they
    define the frame of reference of the object. The rotation and location are
    accessible from the _rot and _loc attributes, respectively, or from the
    appropriate places in the homogenous transformation stored in _transform
    and kept in sync with _rot and _loc.
    
    A surface may construct its own local coordinates from the vectors of the
    rotation matrix.
    The rotation matrix is from the global to the local, so its columns are the
    basis of the local coordinates, written in the global coordinates. See [1]_ p. 25

    Tentative transformations are held in _temp_frame, which allows to preserve the
    relative transform while holding a global transform.
    
    .. [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005.
    """
    def __init__(self,  location=None,  rotation=None):
        # default location and rotation:
        if location is None:
            location = N.zeros(3)
        if rotation is None:
            rotation = N.eye(3)

        self._transform = N.zeros((4,4))
        self._transform[3,:] = N.r_[0, 0, 0, 1]
        self.set_location(location)
        self.set_rotation(rotation)
        self._temp_frame = self._transform
        
        # TODO for compatibility with Coin3D we might need to store these rotations
        # internally as quaternions... hmmm...???

    def get_location(self):
        return self._loc

    def get_rotation(self):
        return self._rot

    def set_location(self,  location):
        """Sets the location within the object"""
        if location.shape != (3, ):
            raise ValueError("location must be a 1D 3-component array")
        self._loc = location
        self._transform[:3,3] = location
    
    def set_rotation(self,  rotation):
        """Sets the rotation within the object"""
        if  rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 array")
        self._rot = rotation
        self._transform[:3,:3] = rotation

    def set_transform(self, transform):
        """Defines the transformation matrix the puts the surface into the coordinates of the
        object containing the surface (the parent object).
        
        Arguments:
        transform - a 2D array defining the 4 by 4 transformation matrix."""
        self._transform = transform
        self._loc = self._transform[:3,3]
        self._rot = self._transform[:3,:3]

    def get_transform(self):
        return self._transform

    def transform_frame(self, transform):
        """Updates the transformation matrix that puts the surface into the global
        coordinates. I.e., if the object the surface is in is rotated, than the surface
        is also rotated.  It then defines a temporary rotated frame for use of
        calculations."""
        self._temp_frame = N.dot(transform, self._transform)

    def get_scene_graph_transform(self):
        """
        Create the Coin3D transform to translate and rotate this frame, in
        accordance with the self._rot and self._loc matrices held in this class.
        """
        n = coin.SoSeparator()
        if N.any(self._loc != N.array((0,0,0))):
            tr = coin.SoTranslation()
            x,y,z = self._loc
            tr.translation = coin.SbVec3f((x,y,z))
            #print "tr.translation = ",tr.translation.getValue().getValue()
            n.addChild(tr)

        if not N.all(N.equal(self._rot, N.eye(3))):
            m = self._rot
            # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            tr = m[0,0] + m[1,1] + m[2,2]
            if tr > 0:
              S = N.sqrt(tr + 1.0)*2
              qw = 0.25 * S
              qx = (m[2,1] - m[1,2]) / S
              qy = (m[0,2] - m[2,0]) / S
              qz = (m[1,0] - m[0,1]) / S;
            elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
              S = N.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])*2
              qw = (m[2,1] - m[1,2]) / S
              qx = 0.25 * S
              qy = (m[0,1] + m[1,0]) / S
              qz = (m[0,2] + m[2,0]) / S
            elif m[1,1] > m[2,2]:
              S = N.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
              qw = (m[0,2] - m[2,0]) / S
              qx = (m[0,1] + m[1,0]) / S
              qy = 0.25 * S
              qz = (m[1,2] + m[2,1]) / S
            else:
              S = N.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
              qw = (m[1,0] - m[0,1]) / S
              qx = (m[0,2] + m[2,0]) / S
              qy = (m[1,2] + m[2,1]) / S
              qz = 0.25 * S
            ro = coin.SoRotation()
            ro.rotation = (qx,qy,qz,qw)
            n.addChild(ro)

        return n

# vim: et:ts=4
