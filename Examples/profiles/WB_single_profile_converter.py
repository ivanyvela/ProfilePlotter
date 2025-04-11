

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
import math

def calculate_angle(p1, p2):
    """Calculate the angle between two points."""
    angle = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
    return angle % 360

def calculate_distance(p1, p2):
    """Calculate the distance between two points."""
    return p1.distance(p2)

def process_shapefile(shapefile_path, output_csv):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Prepare the list to store extracted information
    data = []
    
    for idx, feature in gdf.iterrows():
        geometry = feature.geometry
        
        if isinstance(geometry, LineString):
            points = list(geometry.coords)
            for i in range(len(points)):
                point = Point(points[i])
                x, y = point.x, point.y
                vertex_index = i
                vertex_part = 0  # Assuming a single part for simplicity
                vertex_part_index = i
                distance = calculate_distance(Point(points[i - 1]), point) if i > 0 else 0
                angle = calculate_angle(Point(points[i - 1]), point) if i > 0 else 0
                data.append([idx + 1, vertex_index, vertex_part, vertex_part_index, distance, angle, x, y])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['id', 'vertex_index', 'vertex_part', 'vertex_part_index', 'distance', 'angle', 'X', 'Y'])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data has been successfully exported to {output_csv}")

# Specify the path to your shapefile and the output CSV
shapefile_path = r"C:\Users\au487220\OneDrive - Aarhus universitet\Documents\Random_scripts\profiler_old\Kakuma\profiles\prof_test2.shp"
output_csv = "prof_test2.csv"

# Process the shapefile and generate the CSV
process_shapefile(shapefile_path, output_csv)