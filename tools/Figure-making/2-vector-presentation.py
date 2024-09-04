# Ensure this is at the top of your notebook to enable inline plotting
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Vector operations
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def project_vector_2d(v, basis1, basis2):
    """Project a 3D vector onto a 2D plane defined by two basis vectors."""
    return np.array([np.dot(v, basis1), np.dot(v, basis2)])

def angle_between(v1, v2):
    """Calculate the angle between two vectors in degrees."""
    return np.degrees(np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0)))

def rotate_2d(v, angle):
    """Rotate a 2D vector by a given angle (in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

# Define the two 3D vectors
v1 = np.array([7.48, 6.08, 5.67])
v2 = np.array([5.31, 6.11, 3.49])

# Calculate the normal vector to the plane
normal = np.cross(v1, v2)

# Create an orthonormal basis for the plane
basis1 = normalize(v1)
basis2 = normalize(np.cross(normal, v1))

# Project the vectors onto the plane
v1_2d = project_vector_2d(v1, basis1, basis2)
v2_2d = project_vector_2d(v2, basis1, basis2)

# Rotate vectors to align v1 with Y-axis
rotation_angle = -np.arctan2(v1_2d[0], v1_2d[1])
v1_2d_rotated = rotate_2d(v1_2d, rotation_angle)
v2_2d_rotated = rotate_2d(v2_2d, rotation_angle)

# Ensure v1 is pointing in the positive y direction
if v1_2d_rotated[1] < 0:
    v1_2d_rotated = -v1_2d_rotated
    v2_2d_rotated = -v2_2d_rotated

# Scale vectors to the length of the smaller vector
min_length = min(np.linalg.norm(v1_2d_rotated), np.linalg.norm(v2_2d_rotated))
v1_2d_scaled = v1_2d_rotated / np.linalg.norm(v1_2d_rotated) * min_length
v2_2d_scaled = v2_2d_rotated / np.linalg.norm(v2_2d_rotated) * min_length

# Calculate the angle between the vectors
angle = angle_between(v1, v2)

# Plotting functions
def setup_plot():
    """Set up the plot with transparent background and no axes."""
    fig = plt.figure(figsize=(30, 30), dpi=150)
    fig.patch.set_alpha(0)  # Make the figure background transparent
    ax = plt.gca()
    ax.set_facecolor('none')  # Make the axes background transparent
    ax.axis('off')
    return ax

def plot_vectors(ax, v1, v2, max_val):
    """Plot the two vectors."""
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#404040', width=0.006, headwidth=3, headlength=5, zorder=3)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#BFBFBF', width=0.006, headwidth=3, headlength=5, zorder=3)
    ax.set_xlim(-max_val*1.3, max_val*1.3)
    ax.set_ylim(-max_val*1.3, max_val*1.3)
    ax.set_aspect('equal')

def add_angle_arc(ax, angle, max_val):
    """Add an arc to represent the angle between vectors."""
    arc = plt.Circle((0, 0), radius=max_val*0.4, fill=False, color='#808080', linestyle='--', linewidth=5, zorder=2)
    ax.add_artist(arc)
    label_x_offset = -max_val * 0.5
    label_y_offset = max_val * 0.15
    ax.text(label_x_offset, label_y_offset, f'{angle:.1f}°', fontsize=64, color='black', ha='right', va='bottom', 
            fontweight='bold', zorder=4)

def add_label_along_vector(ax, vector, label, color, opposite_side=False, flip_text=False, max_val=1):
    """Add a label along a vector."""
    vector_angle = np.arctan2(vector[1], vector[0])
    label_pos = vector * 0.6
    rotation = np.degrees(vector_angle)
    offset = rotate_2d(normalize(vector), np.pi/2) * max_val * 0.04
    
    if vector[0] < 0:
        rotation += 180
    if opposite_side:
        offset = -offset
    if flip_text:
        rotation += 180
    
    ax.text(label_pos[0] + offset[0], label_pos[1] + offset[1], label, fontsize=48, color=color, ha='center', va='center', 
            rotation=rotation, rotation_mode='anchor', fontweight='bold', zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3))

def add_coordinate_indicator(ax, max_val):
    """Add a subtle coordinate system indicator."""
    coord_size = max_val * 0.1
    ax.plot([-coord_size, coord_size], [0, 0], color='#DDDDDD', linewidth=3, zorder=1)
    ax.plot([0, 0], [-coord_size, coord_size], color='#DDDDDD', linewidth=3, zorder=1)

# Main plotting function
def create_vector_plot(v1, v2, angle, min_length):
    ax = setup_plot()
    plot_vectors(ax, v1, v2, min_length)
    add_angle_arc(ax, angle, min_length)
    add_label_along_vector(ax, v1, 'Dipole unbound', '#404040', opposite_side=True, flip_text=True, max_val=min_length)
    add_label_along_vector(ax, v2, 'Dipole bound', '#BFBFBF', max_val=min_length)
    add_coordinate_indicator(ax, min_length)
    plt.tight_layout()
    return ax

# Create and save the plot
ax = create_vector_plot(v1_2d_scaled, v2_2d_scaled, angle, min_length)
plt.savefig('vector_projection_transparent.png', format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Print results
print(f"Original v1: {v1}")
print(f"Original v2: {v2}")
print(f"Scaled v1: {v1_2d_scaled}")
print(f"Scaled v2: {v2_2d_scaled}")
print(f"Angle between vectors: {angle:.2f}°")
print("Figure with transparent background saved as 'vector_projection_transparent.png'")
