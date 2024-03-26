import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UCP_OOP_WR import UCPGenerator
import sys

def main(city_name):
    shapefile_path = 'MapPluto/MapPluto_data_polygons_projected.shp'
    # shapefile_path_1 = 'MapPluto_small_boundary_WGS_Mercator_points'
    # shapefile_path = f"MapPluto_small_boundary_WGS_Mercator_points_{city_name.upper()}_Sha.shp"
    # reference_raster_path = 'A_Building_Count.tif'
    # shapefile_path = 'MapPluto_small_boundary_WGS_Mercator_polygons.shp'
    processor = UCPGenerator(shapefile_path, city_name)
    
    # Build point data fom polygon data
    processor.convert_polygon_to_point()

    # new_raster_path = 'A_Building_Count_Same_Extent.tif'
    new_raster_path_A = 'A_Building_Count_'+ city_name + '.tif'
    #new_raster_path = 'A_Building_Count_Full_NYC.tif'
    processor.step_A(new_raster_path_A)

    new_raster_path_B = 'B_Building_Heights_Sum_' + city_name + '.tif'
    processor.step_B(new_raster_path_B, shapefile_path = None, reference_raster_path=new_raster_path_A)

    new_raster_path_C = 'C_Building_Heights_Mean_' + city_name + '.tif'
    processor.step_C(new_raster_path_C, shapefile_path=None, reference_raster_path=new_raster_path_A)

    new_raster_path_D = 'D_Total_Shape_Area_' + city_name + '.tif'
    processor.step_D(new_raster_path_D, shapefile_path=None, reference_raster_path=new_raster_path_A)

    new_raster_path_E = 'E_Building_Volume_Sum_' + city_name + '.tif'
    processor.step_E(new_raster_path_E, shapefile_path=None, reference_raster_path=new_raster_path_A)

    new_raster_path_F = 'F_Building_Area_Sum_' + city_name + '.tif'
    processor.step_F(new_raster_path_F, shapefile_path=None, reference_raster_path=new_raster_path_A)
    
    processor.execute_workflow(raster_path=new_raster_path_C)
    processor.generate_rasters_for_all_areas()
    processor.process_ucp2_rasters()
    processor.process_ucp3_rasters()
    
    modified_raster_path = f"Building_Heights_Dev_{city_name}.tif"
    modified_shapefile_path = "x_Mean_building_height_points_" + city_name + ".shp"
    processor.raster_average(new_raster_path=modified_raster_path, field_name='Bu_Dev_Sq', shapefile_path=modified_shapefile_path, reference_raster_path=new_raster_path_A)
    processor.process_ucp4_rasters(modified_raster_path, new_raster_path_A)
    
    processor.ucp5()
    processor.ucp6a()
    processor.ucp6b()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        city_name = sys.argv[1]
        main(city_name)
    else:
        print("Please provide city name")
