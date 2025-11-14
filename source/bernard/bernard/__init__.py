# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
Python module with the BERNARD robot model and configuration.
"""

# Register Gym environments.
from .bernard import BERNARD_CFG, BERNARD_WITH_CAMERA_CFG

__all__ = ["BERNARD_CFG", "BERNARD_WITH_CAMERA_CFG"]
