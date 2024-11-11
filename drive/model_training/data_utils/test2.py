import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import shapely

def fill_between_polygons(poly1, poly2, ax):
    """
    Fills the space between two polygons with white.
    
    Parameters:
    poly1 (Polygon): The first Shapely polygon.
    poly2 (Polygon): The second Shapely polygon.
    ax (matplotlib.axes.Axes): The Matplotlib axis to draw on.
    """
    # Calculate the union of the two polygons
    difference = poly1.symmetric_difference(poly2)
    
    # Calculate the intersection of the two polygons
    # Extract polygons from the GeometryCollection
    polygons = [geom for geom in difference.geoms if isinstance(geom, Polygon)]


    # Fill the area between the polygons with white
    
    x, y = polygons[0].exterior.xy
    ax.fill(x, y, color='black', zorder=1)

    # Optionally, plot the original polygons for reference
    #x1, y1 = poly1.exterior.xy
    #ax.fill(x1, y1, alpha=0.5, color='blue', label='Polygon 1', zorder=2)
    
    #x2, y2 = poly2.exterior.xy
    #ax.fill(x2, y2, alpha=0.5, color='red', label='Polygon 2', zorder=2)

# Example usage
if __name__ == "__main__":
    # Create two polygons
    poly1 = Polygon([(0, 0), (0, 6), (6, 6), (0, 6)])
    poly2 = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Fill the space between the polygons
    fill_between_polygons(poly1, poly2, ax)

    # Set plot limits and show the legend
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.legend()
    plt.title('Filling Space Between Two Polygons')
    plt.grid()
    plt.show()

test = []
test.extend([10]*10)
print(test)