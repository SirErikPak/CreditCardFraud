# Calculate the geodesic distance between two latitude/longitude coordinate pairs.
from geopy.distance import geodesic

def calculate_distance(coord1, coord2, units="mi"):
    """
    Calculates the geodesic distance between two latitude/longitude coordinate pairs.

    Parameters:
    - coord1: tuple of (latitude, longitude)
    - coord2: tuple of (latitude, longitude)
    - units: "km" for kilometers (default), "mi" for miles

    Returns:
    - distance: float, the distance in the specified units
    """
    if units == "km":
        return geodesic(coord1, coord2).km
    elif units == "mi":
        return geodesic(coord1, coord2).miles
    else:
        raise ValueError("Units must be 'km' or 'mi'")