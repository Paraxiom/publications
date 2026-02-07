"""
Torus Injection Rejection Animation
Visualizes: "Prompt Injection is a Control Problem"

Shows:
- Green particle: Valid latent trajectory (stays on manifold)
- Red particle: Injection attempt (tries to escape, gets pulled back)
- Torus: Compact constraint manifold T²
"""

import bpy
import math
import mathutils

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Animation settings
FRAMES = 240
FPS = 24
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = FRAMES
bpy.context.scene.render.fps = FPS

# Torus parameters
R = 2.0  # Major radius
r = 0.7  # Minor radius

def torus_point(u, v, R=2.0, r=0.7):
    """Get point on torus surface given angles u, v"""
    x = (R + r * math.cos(v)) * math.cos(u)
    y = (R + r * math.cos(v)) * math.sin(u)
    z = r * math.sin(v)
    return (x, y, z)

def torus_normal(u, v, R=2.0, r=0.7):
    """Get outward normal at point on torus"""
    nx = math.cos(v) * math.cos(u)
    ny = math.cos(v) * math.sin(u)
    nz = math.sin(v)
    return mathutils.Vector((nx, ny, nz)).normalized()

# ============================================
# CREATE TORUS
# ============================================
bpy.ops.mesh.primitive_torus_add(
    major_radius=R,
    minor_radius=r,
    major_segments=64,
    minor_segments=32,
    location=(0, 0, 0)
)
torus = bpy.context.active_object
torus.name = "Manifold_T2"

# Torus material (dark with wireframe look)
mat_torus = bpy.data.materials.new(name="TorusMaterial")
mat_torus.use_nodes = True
nodes = mat_torus.node_tree.nodes
nodes.clear()

# Principled BSDF for torus
node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
node_bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.1, 1.0)
node_bsdf.inputs['Metallic'].default_value = 0.3
node_bsdf.inputs['Roughness'].default_value = 0.4
node_bsdf.inputs['Alpha'].default_value = 0.7

node_output = nodes.new('ShaderNodeOutputMaterial')
mat_torus.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

mat_torus.blend_method = 'BLEND'
torus.data.materials.append(mat_torus)

# ============================================
# CREATE GREEN PARTICLE (Valid trajectory)
# ============================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, 0))
green_particle = bpy.context.active_object
green_particle.name = "Valid_Trajectory"

mat_green = bpy.data.materials.new(name="GreenEmission")
mat_green.use_nodes = True
nodes_g = mat_green.node_tree.nodes
nodes_g.clear()

node_emission_g = nodes_g.new('ShaderNodeEmission')
node_emission_g.inputs['Color'].default_value = (0.1, 1.0, 0.3, 1.0)
node_emission_g.inputs['Strength'].default_value = 3.0

node_output_g = nodes_g.new('ShaderNodeOutputMaterial')
mat_green.node_tree.links.new(node_emission_g.outputs['Emission'], node_output_g.inputs['Surface'])
green_particle.data.materials.append(mat_green)

# ============================================
# CREATE RED PARTICLE (Injection attempt)
# ============================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, 0))
red_particle = bpy.context.active_object
red_particle.name = "Injection_Attempt"

mat_red = bpy.data.materials.new(name="RedEmission")
mat_red.use_nodes = True
nodes_r = mat_red.node_tree.nodes
nodes_r.clear()

node_emission_r = nodes_r.new('ShaderNodeEmission')
node_emission_r.inputs['Color'].default_value = (1.0, 0.2, 0.1, 1.0)
node_emission_r.inputs['Strength'].default_value = 3.0

node_output_r = nodes_r.new('ShaderNodeOutputMaterial')
mat_red.node_tree.links.new(node_emission_r.outputs['Emission'], node_output_r.inputs['Surface'])
red_particle.data.materials.append(mat_red)

# ============================================
# CREATE TRAILS (curve objects)
# ============================================
def create_trail_curve(name, color):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.03
    curve_data.bevel_resolution = 4

    spline = curve_data.splines.new('POLY')
    spline.points.add(FRAMES - 1)

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emission = mat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 2.0
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    curve_obj.data.materials.append(mat)

    return curve_obj, spline

green_trail, green_spline = create_trail_curve("GreenTrail", (0.1, 1.0, 0.3, 1.0))
red_trail, red_spline = create_trail_curve("RedTrail", (1.0, 0.2, 0.1, 1.0))

# ============================================
# ANIMATE PARTICLES
# ============================================

# Green: smooth geodesic on torus
green_positions = []
for frame in range(FRAMES):
    t = frame / FRAMES
    u = t * 4 * math.pi  # Two full loops around major
    v = t * 6 * math.pi  # Three loops around minor
    pos = torus_point(u, v, R, r)
    green_positions.append(pos)

# Red: tries to escape, gets pulled back
red_positions = []
for frame in range(FRAMES):
    t = frame / FRAMES
    u = t * 3 * math.pi
    v = t * 2 * math.pi

    # Base position on torus
    base_pos = torus_point(u, v, R, r)
    normal = torus_normal(u, v, R, r)

    # Injection attempts: periodic "escape" that gets damped
    # High frequency oscillation that decays
    escape_amplitude = 0.4 * math.sin(t * 20 * math.pi) * math.exp(-t * 3)

    # Add some chaotic high-frequency component
    chaos = 0.2 * math.sin(t * 50 * math.pi) * math.exp(-t * 5)

    total_escape = escape_amplitude + chaos

    # Position with escape attempt
    pos = (
        base_pos[0] + normal.x * total_escape,
        base_pos[1] + normal.y * total_escape,
        base_pos[2] + normal.z * total_escape
    )
    red_positions.append(pos)

# Set keyframes for particles
for frame in range(FRAMES):
    # Green particle
    green_particle.location = green_positions[frame]
    green_particle.keyframe_insert(data_path="location", frame=frame + 1)

    # Red particle
    red_particle.location = red_positions[frame]
    red_particle.keyframe_insert(data_path="location", frame=frame + 1)

    # Update trails
    green_spline.points[frame].co = (*green_positions[frame], 1.0)
    red_spline.points[frame].co = (*red_positions[frame], 1.0)

# ============================================
# LIGHTING
# ============================================
# Key light
bpy.ops.object.light_add(type='AREA', location=(5, -5, 5))
key_light = bpy.context.active_object
key_light.data.energy = 200
key_light.data.size = 3

# Fill light
bpy.ops.object.light_add(type='AREA', location=(-4, 4, 3))
fill_light = bpy.context.active_object
fill_light.data.energy = 100
fill_light.data.size = 2

# Rim light
bpy.ops.object.light_add(type='AREA', location=(0, 5, -2))
rim_light = bpy.context.active_object
rim_light.data.energy = 80

# ============================================
# CAMERA
# ============================================
bpy.ops.object.camera_add(location=(6, -6, 4))
camera = bpy.context.active_object
camera.name = "MainCamera"

# Point camera at torus center
direction = mathutils.Vector((0, 0, 0)) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

bpy.context.scene.camera = camera

# Animate camera rotation around torus
camera_empty = bpy.data.objects.new("CameraTarget", None)
bpy.context.collection.objects.link(camera_empty)
camera_empty.location = (0, 0, 0)

camera.parent = camera_empty
camera.location = (8, 0, 3)

# Slow rotation
camera_empty.rotation_euler = (0, 0, 0)
camera_empty.keyframe_insert(data_path="rotation_euler", frame=1)
camera_empty.rotation_euler = (0, 0, math.radians(90))
camera_empty.keyframe_insert(data_path="rotation_euler", frame=FRAMES)

# ============================================
# RENDER SETTINGS
# ============================================
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.resolution_x = 1080
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True

# World background
world = bpy.data.worlds.new("DarkWorld")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.01, 0.02, 1)

# Output path
bpy.context.scene.render.filepath = "/Users/sylvaincormier/paraxiom/publications/papers/torus_animation/frame_"
bpy.context.scene.render.image_settings.file_format = 'PNG'

print("=" * 50)
print("Animation setup complete!")
print(f"Frames: {FRAMES}")
print(f"Duration: {FRAMES/FPS:.1f} seconds")
print("")
print("To render animation:")
print("  bpy.ops.render.render(animation=True)")
print("")
print("Or render single frame:")
print("  bpy.ops.render.render(write_still=True)")
print("=" * 50)

# Save the .blend file
bpy.ops.wm.save_as_mainfile(filepath="/Users/sylvaincormier/paraxiom/publications/papers/torus_injection.blend")
print("Saved: torus_injection.blend")
