
# from Sim_numpy_jit import Sim
from Sim_C import Sim

camera_index = 0

fullscreen = False
mirror_screen = 0 # 'Normal', 'Horizontal', 'Verticle', 'Both'
render_mask = True

bg_mode = 1 # 'white', 'black', 'hue', 'bg subtract'
mask_level = 0.19
mask_width = 0.1

sim_res_multiplier = 0.7
flow_speed = 0.2
flow_direction = 0 # 'right', 'down', 'left', 'up'
num_smoke_streams = 18
smoke_percentage = 0.7
