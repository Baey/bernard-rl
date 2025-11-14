from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.core.robots import Robot

robot = Robot(
    prim_path="/World/BERNARD_CAMERA",
    name="BERNARD_CAMERA",
    usd_path="C:\\Users\\pawel\\bernard\\source\\bernard\\bernard\\data\\BERNARD_CAMERA\\BERNARD_CAMERA.usd",
)

# Export current configuration to new folder
robot.save_configuration("C:\\Users\\pawel\\bernard\\source\\bernard\\bernard\\data\\BERNARD_CAMERA\\configuration")