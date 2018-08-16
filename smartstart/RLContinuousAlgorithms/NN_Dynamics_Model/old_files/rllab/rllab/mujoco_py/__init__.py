import os

from .mjconstants import *
from .mjcore import MjModel
from .mjcore import register_license
from .mjviewer import MjViewer

register_license(os.path.join(os.path.dirname(__file__),
                              '../../vendor/mujoco/mjkey.txt'))
