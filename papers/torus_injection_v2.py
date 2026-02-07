"""
Torus Injection Rejection Animation v2
"Prompt Injection is a Control Problem"

Enhanced with:
- Text labels
- Attention flow visualization (particles)
- Better visual storytelling
"""

import bpy
import math
import mathutils
from mathutils import Vector

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
    return Vector((nx, ny, nz)).normalized()

# ============================================
# WORLD SETUP
# ============================================
world = bpy.data.worlds.new("DarkWorld")
bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (0.015, 0.015, 0.04, 1)
bg_node.inputs[1].default_value = 0.3

bpy.context.scene.render.film_transparent = False

# ============================================
# CREATE TORUS (Constraint Manifold)
# ============================================
bpy.ops.mesh.primitive_torus_add(
    major_radius=R,
    minor_radius=r,
    major_segments=96,
    minor_segments=48,
    location=(0, 0, 0)
)
torus = bpy.context.active_object
torus.name = "Manifold_T2"

# Torus material - subtle grid pattern
mat_torus = bpy.data.materials.new(name="TorusMaterial")
mat_torus.use_nodes = True
nodes = mat_torus.node_tree.nodes
links = mat_torus.node_tree.links
nodes.clear()

# Create wireframe-like material
node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
node_bsdf.location = (0, 0)
node_bsdf.inputs['Base Color'].default_value = (0.08, 0.08, 0.15, 1.0)
node_bsdf.inputs['Metallic'].default_value = 0.2
node_bsdf.inputs['Roughness'].default_value = 0.5
node_bsdf.inputs['Alpha'].default_value = 0.75

# Add emission for edge glow
node_emission = nodes.new('ShaderNodeEmission')
node_emission.location = (0, -200)
node_emission.inputs['Color'].default_value = (0.1, 0.15, 0.3, 1.0)
node_emission.inputs['Strength'].default_value = 0.3

node_mix = nodes.new('ShaderNodeMixShader')
node_mix.location = (300, 0)
node_mix.inputs[0].default_value = 0.15

node_output = nodes.new('ShaderNodeOutputMaterial')
node_output.location = (500, 0)

links.new(node_bsdf.outputs['BSDF'], node_mix.inputs[1])
links.new(node_emission.outputs['Emission'], node_mix.inputs[2])
links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])

mat_torus.blend_method = 'BLEND'
torus.data.materials.append(mat_torus)

# ============================================
# CREATE PARTICLES (Valid trajectory)
# ============================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 0))
green_particle = bpy.context.active_object
green_particle.name = "Valid_Trajectory"

mat_green = bpy.data.materials.new(name="GreenEmission")
mat_green.use_nodes = True
nodes_g = mat_green.node_tree.nodes
nodes_g.clear()
node_emission_g = nodes_g.new('ShaderNodeEmission')
node_emission_g.inputs['Color'].default_value = (0.2, 1.0, 0.4, 1.0)
node_emission_g.inputs['Strength'].default_value = 5.0
node_output_g = nodes_g.new('ShaderNodeOutputMaterial')
mat_green.node_tree.links.new(node_emission_g.outputs['Emission'], node_output_g.inputs['Surface'])
green_particle.data.materials.append(mat_green)

# ============================================
# CREATE PARTICLES (Injection attempt)
# ============================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 0))
red_particle = bpy.context.active_object
red_particle.name = "Injection_Attempt"

mat_red = bpy.data.materials.new(name="RedEmission")
mat_red.use_nodes = True
nodes_r = mat_red.node_tree.nodes
nodes_r.clear()
node_emission_r = nodes_r.new('ShaderNodeEmission')
node_emission_r.inputs['Color'].default_value = (1.0, 0.3, 0.15, 1.0)
node_emission_r.inputs['Strength'].default_value = 5.0
node_output_r = nodes_r.new('ShaderNodeOutputMaterial')
mat_red.node_tree.links.new(node_emission_r.outputs['Emission'], node_output_r.inputs['Surface'])
red_particle.data.materials.append(mat_red)

# ============================================
# CREATE ATTENTION FLOW PARTICLES
# ============================================
def create_attention_particle(name, color, size=0.05):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(0, 0, 0))
    particle = bpy.context.active_object
    particle.name = name

    mat = bpy.data.materials.new(name=f"{name}_mat")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emission = mat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 3.0
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    particle.data.materials.append(mat)
    return particle

# Create attention flow particles (following the main particles)
attention_green = [create_attention_particle(f"Attn_Green_{i}", (0.1, 0.8, 0.3, 1.0), 0.04) for i in range(8)]
attention_red = [create_attention_particle(f"Attn_Red_{i}", (1.0, 0.4, 0.2, 1.0), 0.04) for i in range(8)]

# ============================================
# CREATE TRAILS
# ============================================
def create_trail_curve(name, color, bevel=0.04):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = bevel
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
    emission.inputs['Strength'].default_value = 2.5
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    curve_obj.data.materials.append(mat)

    return curve_obj, spline

green_trail, green_spline = create_trail_curve("GreenTrail", (0.15, 0.9, 0.35, 1.0))
red_trail, red_spline = create_trail_curve("RedTrail", (1.0, 0.35, 0.2, 1.0))

# ============================================
# TEXT LABELS
# ============================================
def create_text(text, location, size=0.3, color=(1, 1, 1, 1)):
    bpy.ops.object.text_add(location=location)
    txt = bpy.context.active_object
    txt.data.body = text
    txt.data.size = size
    txt.data.align_x = 'CENTER'
    txt.data.align_y = 'CENTER'

    # Extrude for 3D effect
    txt.data.extrude = 0.02

    mat = bpy.data.materials.new(name=f"Text_{text[:10]}")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    emission = mat.node_tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 2.0
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    txt.data.materials.append(mat)

    return txt

# Title
title = create_text("PROMPT INJECTION AS CONTROL PROBLEM", (0, 0, 4), 0.35, (0.9, 0.7, 0.2, 1))
title.rotation_euler = (math.radians(90), 0, 0)

# Manifold label
manifold_label = create_text("T² Manifold", (-3.5, 0, 0), 0.25, (0.4, 0.5, 0.8, 1))
manifold_label.rotation_euler = (math.radians(90), 0, math.radians(-30))

# Valid trajectory label
valid_label = create_text("Valid Trajectory", (3.5, -1, 1.5), 0.2, (0.2, 1.0, 0.4, 1))
valid_label.rotation_euler = (math.radians(90), 0, math.radians(20))

# Injection label
inject_label = create_text("Injection Attempt", (3.5, 1, -1), 0.2, (1.0, 0.3, 0.2, 1))
inject_label.rotation_euler = (math.radians(90), 0, math.radians(20))

# Spectral gap label (appears during rejection)
spectral_label = create_text("Spectral Gap Rejection", (0, 3, -1.5), 0.22, (1.0, 0.6, 0.1, 1))
spectral_label.rotation_euler = (math.radians(90), 0, 0)

# ============================================
# ANIMATE PARTICLES
# ============================================

# Green: smooth geodesic on torus (valid attention flow)
green_positions = []
for frame in range(FRAMES):
    t = frame / FRAMES
    u = t * 4 * math.pi
    v = t * 6 * math.pi
    pos = torus_point(u, v, R, r)
    green_positions.append(pos)

# Red: injection attempt with high-frequency perturbation
red_positions = []
for frame in range(FRAMES):
    t = frame / FRAMES
    u = t * 3 * math.pi
    v = t * 2 * math.pi

    base_pos = torus_point(u, v, R, r)
    normal = torus_normal(u, v, R, r)

    # Multiple escape attempts that get damped
    escape1 = 0.5 * math.sin(t * 25 * math.pi) * math.exp(-t * 2.5)
    escape2 = 0.3 * math.sin(t * 40 * math.pi) * math.exp(-t * 4)
    escape3 = 0.2 * math.sin(t * 60 * math.pi) * math.exp(-t * 6)

    total_escape = escape1 + escape2 + escape3

    pos = (
        base_pos[0] + normal.x * total_escape,
        base_pos[1] + normal.y * total_escape,
        base_pos[2] + normal.z * total_escape
    )
    red_positions.append(pos)

# Keyframe main particles
for frame in range(FRAMES):
    green_particle.location = green_positions[frame]
    green_particle.keyframe_insert(data_path="location", frame=frame + 1)

    red_particle.location = red_positions[frame]
    red_particle.keyframe_insert(data_path="location", frame=frame + 1)

    green_spline.points[frame].co = (*green_positions[frame], 1.0)
    red_spline.points[frame].co = (*red_positions[frame], 1.0)

# Keyframe attention particles (trailing behind main particles)
for i, attn in enumerate(attention_green):
    delay = (i + 1) * 3  # Frames behind
    for frame in range(FRAMES):
        src_frame = max(0, frame - delay)
        attn.location = green_positions[src_frame]
        # Add slight offset for spread
        offset = 0.1 * math.sin(frame * 0.5 + i)
        attn.location = (
            attn.location[0] + offset * 0.3,
            attn.location[1] + offset * 0.3,
            attn.location[2] + offset * 0.2
        )
        attn.keyframe_insert(data_path="location", frame=frame + 1)

for i, attn in enumerate(attention_red):
    delay = (i + 1) * 3
    for frame in range(FRAMES):
        src_frame = max(0, frame - delay)
        attn.location = red_positions[src_frame]
        # More chaotic offset for injection
        chaos = 0.15 * math.sin(frame * 2 + i * 1.5) * math.exp(-frame/FRAMES * 3)
        attn.location = (
            attn.location[0] + chaos,
            attn.location[1] + chaos * 0.8,
            attn.location[2] + chaos * 0.6
        )
        attn.keyframe_insert(data_path="location", frame=frame + 1)

# Animate spectral label visibility (fade in during rejection phase)
spectral_label.hide_render = True
spectral_label.hide_viewport = True
spectral_label.keyframe_insert(data_path="hide_render", frame=1)
spectral_label.keyframe_insert(data_path="hide_viewport", frame=1)

spectral_label.hide_render = False
spectral_label.hide_viewport = False
spectral_label.keyframe_insert(data_path="hide_render", frame=30)
spectral_label.keyframe_insert(data_path="hide_viewport", frame=30)

spectral_label.hide_render = True
spectral_label.hide_viewport = True
spectral_label.keyframe_insert(data_path="hide_render", frame=120)
spectral_label.keyframe_insert(data_path="hide_viewport", frame=120)

# ============================================
# LIGHTING
# ============================================
# Key light (golden)
bpy.ops.object.light_add(type='AREA', location=(6, -6, 6))
key_light = bpy.context.active_object
key_light.data.energy = 400
key_light.data.size = 4
key_light.data.color = (1.0, 0.95, 0.85)

# Fill light (blue)
bpy.ops.object.light_add(type='AREA', location=(-5, 5, 4))
fill_light = bpy.context.active_object
fill_light.data.energy = 200
fill_light.data.size = 3
fill_light.data.color = (0.8, 0.85, 1.0)

# Rim light
bpy.ops.object.light_add(type='AREA', location=(0, 6, -3))
rim_light = bpy.context.active_object
rim_light.data.energy = 150
rim_light.data.size = 2

# ============================================
# CAMERA (Slow orbit)
# ============================================
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
camera_target = bpy.context.active_object
camera_target.name = "CameraTarget"

bpy.ops.object.camera_add(location=(9, 0, 4))
camera = bpy.context.active_object
camera.name = "MainCamera"
camera.parent = camera_target

# Point at center
constraint = camera.constraints.new(type='TRACK_TO')
constraint.target = camera_target
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

bpy.context.scene.camera = camera

# Animate camera orbit
camera_target.rotation_euler = (0, 0, 0)
camera_target.keyframe_insert(data_path="rotation_euler", frame=1)
camera_target.rotation_euler = (0, 0, math.radians(60))
camera_target.keyframe_insert(data_path="rotation_euler", frame=FRAMES)

# ============================================
# RENDER SETTINGS
# ============================================
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.resolution_x = 1080
bpy.context.scene.render.resolution_y = 1080

# Output
bpy.context.scene.render.filepath = "/Users/sylvaincormier/paraxiom/publications/papers/torus_v2/frame_"
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Save
bpy.ops.wm.save_as_mainfile(filepath="/Users/sylvaincormier/paraxiom/publications/papers/torus_injection_v2.blend")

print("=" * 50)
print("Animation v2 setup complete!")
print("Features:")
print("  - Title: 'Prompt Injection as Control Problem'")
print("  - Labels: T² Manifold, Valid Trajectory, Injection Attempt")
print("  - Spectral Gap Rejection label (frames 30-120)")
print("  - Attention flow particles (8 per trajectory)")
print("  - Slow camera orbit")
print(f"Frames: {FRAMES} @ {FPS}fps = {FRAMES/FPS:.1f}s")
print("=" * 50)
