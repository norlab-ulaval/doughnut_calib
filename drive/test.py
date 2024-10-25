import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint,conca
from shapely import concave_hull

def compute_concave_hull(x, y, alpha):
    """
    Compute the concave hull of a set of points defined by vectors x and y,
    and plot the resulting hull with a specified color for the interior.

    Parameters:
    - x: List or array-like, x-coordinates of the points.
    - y: List or array-like, y-coordinates of the points.
    - alpha: Parameter that controls the shape of the concave hull.
    - fill_color: Color for the interior of the hull.
    """

    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Create a MultiPoint object
    points = MultiPoint(list(zip(x, y)))

    # Compute the concave hull
    concave_hull_ = concave_hull(points, alpha)

    # Get the vertices of the concave hull
    vertices = list(concave_hull_.exterior.coords)

    # Plotting
    plt.figure(figsize=(8, 8))
    
    # Fill the concave hull
    x_hull, y_hull = zip(*vertices)
    plt.fill(x_hull, y_hull, color='lightblue', alpha=0.5, label='Concave Hull')

    # Plot original points
    plt.scatter(x, y, color='red', label='Original Points')
    
    # Add labels and legend
    plt.title('Concave Hull with Interior Color')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    return vertices

# Example usage:
x = [1, 2, 3, 5, 6, 7]
y = [1, 3, 2, 6, 5, 7]
alpha = 1.5  # Adjust this value for different concavity
hull_vertices = compute_concave_hull(x, y, alpha)
print(hull_vertices)
