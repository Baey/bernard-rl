import pyvista as pv
import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm
from trimesh.transformations import rotation_matrix

# ---- 1. Mapowanie nazw linków na pliki mesh ----
# Dopasuj ścieżki do faktycznych plików w Twoim projekcie
mesh_dir = Path("BERNARD/meshes/obj")
mesh_map = {
    'body': mesh_dir / "Body.obj",
    'l_hip': mesh_dir / "Hip_Connector.obj",
    'r_hip': mesh_dir / "Hip_Connector.obj",
    'l_arm': mesh_dir / "L_Arm.obj",
    'r_arm': mesh_dir / "R_Arm.obj",
    'l_forearm': mesh_dir / "L_Forearm.obj",
    'r_forearm': mesh_dir / "R_Forearm.obj",
    'l_foot': mesh_dir / "Foot.obj",
    'r_foot': mesh_dir / "Foot.obj",
}

# ---- 2. Wczytanie trajektorii ----
motion_file = "scripts/plots/data/trajectories/amp_trajectory.npz"
data = np.load(motion_file)
first_frame = 15
num_frames = 500
env_idx = 2
names = data['body_names'].tolist()
positions = data["pos"][first_frame:num_frames+first_frame, env_idx, :, :]      # (frames, bodies, 3)
center_x = (positions[:, 0, 0].max() + positions[:, 0, 0].min()) / 2
center_y = (positions[:, 0, 1].max() + positions[:, 0, 1].min()) / 2
positions = positions - [center_x, center_y, 0]
orientations = data["quat"][first_frame:num_frames+first_frame, env_idx, :, :]  # (frames, bodies, 4)

traj_links = ['body', 'l_foot', 'r_foot']
links_translation = {
    'body': 'Tułów',
    'l_foot': 'Lewa stopa',
    'r_foot': 'Prawa stopa'
}

print(f'Loaded {positions.shape[0]} frames from {motion_file}')

# ---- 3. Funkcja: mesh robota w danej klatce ----
def get_robot_mesh_at_frame(frame_idx):
    meshes = []
    for link_idx, link_name in enumerate(names):
        mesh_file = mesh_map.get(link_name)
        if mesh_file is None or not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file for {link_name} not found: {mesh_file}")

        pos = positions[frame_idx, link_idx]
        quat = orientations[frame_idx, link_idx]

        tm = trimesh.load_mesh(mesh_file)
        if link_name == 'r_hip':
            R = rotation_matrix(np.pi, [1,0,0])
            tm.apply_transform(R)
        T = trimesh.transformations.quaternion_matrix(quat)
        T[:3, 3] = pos
        tm.apply_transform(T)
        meshes.append(pv.wrap(tm))
    return meshes

# ---- 4. PyVista scena ----
plotter = pv.Plotter(window_size=[3300, 1800])

# Trajektorie jako spline’y
legend_entries = []
for link_name, color in zip(traj_links, ['red', 'green', 'blue']):
    idx = names.index(link_name)
    pts = positions[:, idx, :]
    spline = pv.Spline(pts, 1000)
    scalars = np.linspace(0.4, 1, spline.n_points) ** 2
    scalars = scalars.tolist()
    plotter.add_mesh(spline, color=color, line_width=3, opacity=scalars)
    legend_entries.append([links_translation[link_name], color])

# Duchy robota w kilku klatkach
alpha = np.linspace(0.2, 0.7, num_frames) ** 2
frames_idx = np.linspace(0, num_frames - 1, 5, dtype=int)
# frames_idx = [7, 29, 50, 97, 120]
for frame in tqdm(frames_idx[:-1]):
    for m in get_robot_mesh_at_frame(frame):
        plotter.add_mesh(m, color="gray", opacity=alpha[frame], backface_culling=True)
for m in get_robot_mesh_at_frame(num_frames - 1):
        plotter.add_mesh(m, color="gray", opacity=1.0)

# plotter.show_grid(
#     color="black",
#     grid='front',       # tylko płaszczyzna XY
#     location='outer'
# )
# floor = pv.Plane(i_size=5, j_size=5, i_resolution=10, j_resolution=10)
# plotter.add_mesh(floor, color="lightgray", style="wireframe", line_width=1, opacity=0.5
# plotter.add_axes()
# plotter.add_legend(legend_entries, bcolor="white", face="circle", border=True, size=(0.11, 0.11))
plotter.show_grid(font_size=22)
plotter.enable_anti_aliasing()
plotter.set_background("white")

# Forward
plotter.set_position([0, 0, 1])
plotter.set_focus([0, 0, 0])
plotter.set_viewup([2, 0, 0]) 

plotter.camera.zoom(1.8)
plotter.show(auto_close=False)  
# plotter.screenshot("forward_trajectory.png")
