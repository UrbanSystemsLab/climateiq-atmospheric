import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class UCPGenerator:

    def __init__(self, polygonfile_path, city_name):
        self.polygonfile_path = polygonfile_path
        self.city_name = city_name.upper()
        self.points = None 
        self.shapefile_path = None
        self.reference_raster_path = None
        
        
    def convert_polygon_to_point(self):
        polygons = gpd.read_file(self.polygonfile_path)

        # Calculate area and perimeter
        polygons["Shape_Area"] = polygons["geometry"].area
        polygons["Shape_Len"] = polygons["geometry"].length

        # Calculate Build_Hgt2, Wall_Area2, Build_are2, Build_Vol2
        polygons["Build_Hgt"] = polygons["NumFloors"] * 3.35
        polygons["Wall_Area"] = polygons["Shape_Len"] * polygons["Build_Hgt"]
        polygons["Build_Area"] = polygons["Wall_Area"] + polygons["Shape_Area"]
        polygons["Build_Vol"] = polygons["Shape_Area"] * polygons["Build_Hgt"]
        
        # Remove polygons with 'Build_Hgt' of 0 before converting to points
        polygons = polygons[polygons["Build_Hgt"] > 0]

        # Convert polygons to points (using centroids)
        points = polygons.copy()
        points["geometry"] = points["geometry"].centroid
        
        # Save the new point shapefile
        self.shapefile_path = f"WGS_Mercator_points_{self.city_name}.shp"
        points.to_file(self.shapefile_path)
        self.points = points


    def step_A(self, new_raster_path):
        try:
            # Step 1: Read the Shapefile
            # points = gpd.read_file(self.shapefile_path)

            # Step 2: Create an Empty Raster
            # Determine bounds and dimensions
            minx, miny, maxx, maxy = self.points.total_bounds
            cell_size = 500
            width = int((maxx - minx) / cell_size)
            height = int((maxy - miny) / cell_size)

            # Create a transform
            transform = from_origin(minx, maxy, cell_size, cell_size)

            # Create the raster
            with rasterio.open(
                new_raster_path, 'w', 
                driver='GTiff', 
                height=height, 
                width=width, 
                count=1, 
                dtype=rasterio.uint32, 
                crs=self.points.crs, 
                transform=transform
            ) as dst:

                # Step 3: Count Points in Each Cell
                # Initialize an empty array
                counts = np.zeros((height, width), dtype=np.uint32)

                # Iterate through points and count
                for point, building_height in zip(self.points.geometry, self.points['Build_Hgt']):
                    if building_height > 0:
                        row, col = dst.index(point.x, point.y)
                        if 0 <= row < height and 0 <= col < width:
                            counts[row, col] += 1

                # Step 4: Update Raster Values
                # Set 0 counts to NA (represented by a specific value, e.g., 9999)
                counts[counts == 0] = 9999
                
                self.reference_raster_path = new_raster_path
                
                # Write the counts to the raster
                dst.write(counts, 1)
        
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            
        except Exception as e:
            print(f"An error occurred: {e}")


    def raster_as_numpy_array(self, tif_file_path):
        with rasterio.open(tif_file_path) as src:
            raster_data = src.read()
            numpy_array = np.array(raster_data)
        print(numpy_array)

    def visualize_raster(self, tif_file_path, missing_value=None):
        """
        Visualize a single-band raster file.

        This method opens a raster file and visualizes its data. If a missing value is specified,
        cells with that value are masked in the visualization. This function is useful for 
        quickly inspecting the contents of a raster file, especially to verify data processing results.

        :param tif_file_path: Path to the raster (.tif) file to be visualized.
        :param missing_value: Optional; a value in the raster to be treated as 'missing' or 'no data'.
                              Cells with this value will be masked (not displayed) in the visualization.
        """

        with rasterio.open(tif_file_path) as src:
            raster_data = src.read()
            numpy_array = np.array(raster_data)

        single_band_data = numpy_array[0] if numpy_array.ndim == 3 else numpy_array

        # Mask the missing values if specified
        if missing_value is not None:
            if missing_value > 0:
                masked_data = np.ma.masked_where(single_band_data == missing_value, single_band_data)
            else:
                masked_data = np.ma.masked_where(single_band_data <= missing_value, single_band_data)
        else:
            masked_data = single_band_data

        # Display the raster data
        plt.imshow(masked_data, cmap='gray')
        plt.colorbar()
        plt.title('Single-band raster data')
        plt.show()


    def raster_sum(self, new_raster_path, field_name, shapefile_path=None, reference_raster_path=None):
        """
        Sum the values of a specified field from a shapefile and create a raster file.

        This method reads point data from a shapefile and calculates the sum of a specified field
        for each grid cell in a raster. The sums are then written to a new raster file. Grid cells
        without data are assigned a default 'nodata' value.

        :param new_raster_path: Path where the new raster file will be saved.
        :param field_name: The field in the shapefile whose values are to be summed.
        :param shapefile_path: Optional; alternative path for the shapefile. If not provided, 
                               uses the path set during class initialization.
        :param reference_raster_path: Optional; alternative path for the reference raster. 
                                      If not provided, uses the path set during class initialization.
        """

        # Use provided shapefile path or default to self.shapefile_path
        shapefile_path = shapefile_path or self.shapefile_path

        # Use provided reference raster path or default to self.reference_raster_path
        # reference_raster_path = reference_raster_path or self.reference_raster_path

        with rasterio.open(reference_raster_path) as existing_raster:
            transform = existing_raster.transform
            bounds = existing_raster.bounds
            width = existing_raster.width
            height = existing_raster.height
            crs = existing_raster.crs

        points = gpd.read_file(shapefile_path)
        # points = self.points
        
        
        with rasterio.open(new_raster_path, 'w', driver='GTiff', height=height, width=width,
                           count=1, dtype=rasterio.float32, crs=crs, transform=transform,
                           nodata=9999.0) as dst:
            sums = np.full((height, width), 9999.0, dtype=np.float32)
            for point, value in zip(points.geometry, points[field_name]):
                row, col = dst.index(point.x, point.y)
                if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
                    if 0 <= row < height and 0 <= col < width:
                        if sums[row, col] == 9999.0:
                            sums[row, col] = 0.0
                        sums[row, col] += value

            dst.write(sums, 1)

    def step_B(self, new_raster_path, shapefile_path = None, reference_raster_path=None):
        self.raster_sum(new_raster_path, 'Build_Hgt', shapefile_path, reference_raster_path)
        
    
    def raster_average(self, new_raster_path, field_name, shapefile_path=None, reference_raster_path=None):
        """
        Calculate the average of a specified field from a shapefile and create a raster file.

        This method reads point data from a shapefile and calculates the average of a specified field
        for each grid cell in a raster. The averages are then written to a new raster file. Grid cells
        without data are assigned a default 'nodata' value.

        :param new_raster_path: Path where the new raster file will be saved.
        :param field_name: The field in the shapefile whose average needs to be calculated.
        :param shapefile_path: Optional; alternative path for the shapefile. If not provided, 
                               uses the path set during class initialization.
        :param reference_raster_path: Optional; alternative path for the reference raster. 
                                      If not provided, uses the path set during class initialization.
        """

        # Use provided shapefile path or default to self.shapefile_path
        shapefile_path = shapefile_path or self.shapefile_path

        # Use provided reference raster path or default to self.reference_raster_path
        # reference_raster_path = reference_raster_path or self.reference_raster_path

        with rasterio.open(reference_raster_path) as existing_raster:
            transform = existing_raster.transform
            bounds = existing_raster.bounds
            width = existing_raster.width
            height = existing_raster.height
            crs = existing_raster.crs

        points = gpd.read_file(shapefile_path)
    
        with rasterio.open(new_raster_path, 'w', driver='GTiff', height=height, width=width,
                           count=1, dtype=rasterio.float32, crs=crs, transform=transform,
                           nodata=9999.0) as dst:
            sums = np.full((height, width), 9999.0, dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.uint32)

            for point, value in zip(points.geometry, points[field_name]):
                row, col = dst.index(point.x, point.y)
                if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
                    if 0 <= row < height and 0 <= col < width:
                        if sums[row, col] == 9999.0:
                            sums[row, col] = 0.0
                        sums[row, col] += value
                        counts[row, col] += 1

            averages = np.where(counts > 0, sums / counts, 9999.0)
            dst.write(averages, 1)
            
    def step_C(self, new_raster_path, shapefile_path=None, reference_raster_path=None):
        ## Step C -- Build_Hgts_mean for each 500 m grid
        self.raster_average(new_raster_path, 'Build_Hgt', shapefile_path, reference_raster_path)
        
    def sum_total_area(self, new_raster_path, field_name, shapefile_path=None, reference_raster_path=None):
        ## Step D - Total_shape_area
        
        # Use provided shapefile path or default to self.shapefile_path
        shapefile_path = shapefile_path or self.shapefile_path

        # Use provided reference raster path or default to self.reference_raster_path
        # reference_raster_path = reference_raster_path or self.reference_raster_path

        # Step 0: Read the existing raster for bounds and transforms
        with rasterio.open(reference_raster_path) as existing_raster:
            transform = existing_raster.transform
            bounds = existing_raster.bounds
            cell_size = existing_raster.res[0]
            width = existing_raster.width
            height = existing_raster.height
            crs = existing_raster.crs

        # Step 1: Read the shapefile
        points = gpd.read_file(shapefile_path)

        # Step 2: Create the raster with the same extent as the existing raster
        with rasterio.open(new_raster_path, 'w', driver='GTiff', height=height, width=width,
                          count=1, dtype=rasterio.float32, crs=crs, transform=transform,
                          nodata=9999.0) as dst:
            sums = np.full((height, width), 9999.0, dtype=np.float32) # Initialize with NoData value
            for point, value1, value2 in zip(points.geometry, points[field_name], points['Build_Hgt']):
                row, col = dst.index(point.x, point.y)
                if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
                    if 0 <= row < height and 0 <= col < width:
                        if sums[row, col] == 9999.0: # If currently NoData, make it 0 before summing
                            sums[row, col] = 0.0
                        sums[row, col] += value1

            # Step 4: Write this sum to a raster file
            dst.write(sums, 1)
            
    
    def step_D(self, new_raster_path, shapefile_path=None, reference_raster_path=None):
        self.sum_total_area(new_raster_path, 'Shape_Area', shapefile_path, reference_raster_path)
        
    
    def step_E(self, new_raster_path, shapefile_path=None, reference_raster_path=None):
        ## Step E -- Build_vol_sum
        self.sum_total_area(new_raster_path, 'Build_Vol', shapefile_path, reference_raster_path)
        
    def step_F(self, new_raster_path, shapefile_path=None, reference_raster_path=None):
        ## Step F - Build_area_sum (i.e., total area of roof and walls)
        self.sum_total_area(new_raster_path, 'Build_Area', shapefile_path, reference_raster_path)
        
    
    ###%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%%%%#####@@@@@@@@@@@@@@@@#################
    
    ### Create the file x_MapPLUTO_small_boundary_intermediary.shp 
    
    ###%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&%%%%%%%%%%%%%%#####@@@@@@@@@@@@@@@@#################
    
    def add_mean_height_to_shapefile(self, raster_path, new_field_name):
        """
        Add a field for mean height to the shapefile.
        """
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)  # Read the first band
            transform = src.transform

        gdf = gpd.read_file(self.shapefile_path)

        mean_heights = []
        for point in gdf.geometry:
            row, col = src.index(point.x, point.y)
            # Validate the indices
            if (0 <= row < raster_data.shape[0]) and (0 <= col < raster_data.shape[1]):
                mean_height = raster_data[row, col]
                #mean_height = np.nan if mean_height == 9999.0 else mean_height
            else:
                #mean_height = np.nan  # or some other nodata value
                mean_height = 9999.0
            mean_heights.append(mean_height)


        gdf[new_field_name] = mean_heights
        return gdf

    def create_bins(self, gdf):
        """
        Create bins for height and area.
        """
        def value_in_range(row, lower_bound, upper_bound, building_height, column_name):
            return row[column_name] if lower_bound <= row[building_height] < upper_bound else 0

        for lower_bound in range(0, 46, 5):
            upper_bound = lower_bound + 5
            gdf[f'Hgt_{lower_bound}_{upper_bound}'] = gdf.apply(
                value_in_range, axis=1, lower_bound=lower_bound, 
                upper_bound=upper_bound, building_height='Build_Hgt', column_name='Build_Hgt')
            gdf[f'Ar_{lower_bound}_{upper_bound}'] = gdf.apply(
                value_in_range, axis=1, lower_bound=lower_bound, 
                upper_bound=upper_bound, building_height='Build_Hgt', column_name='Shape_Area')

        gdf['Hgt_50_abo'] = gdf.apply(lambda row: row['Build_Hgt'] if row['Build_Hgt'] >= 50 else 0, axis=1)
        gdf['Ar_50_abo'] = gdf.apply(lambda row: row['Shape_Area'] if row['Build_Hgt'] >= 50 else 0, axis=1)

        return gdf

    def execute_workflow(self, raster_path = 'C_Building_Heights_Mean_Same_Extent.tif'):
        """
        Execute the complete workflow.
        """
        
        output_shp_file="x_Mean_building_height_points_" + self.city_name + ".shp"
        new_shp_mean_hgt = self.add_mean_height_to_shapefile(raster_path, 'Mean_hgt')
        new_shp_mean_hgt['Build_Dev'] = new_shp_mean_hgt['Build_Hgt'] - new_shp_mean_hgt['Mean_hgt']
        new_shp_mean_hgt['Bu_Dev_Sq'] = new_shp_mean_hgt['Build_Dev'] ** 2
        new_shp_mean_hgt = self.create_bins(new_shp_mean_hgt)

        new_shp_mean_hgt.to_file(output_shp_file)
    
    def generate_rasters_for_all_areas(self):
        
        base_raster_path='./Dist_building_hgts/Sum_area_bins_build_hgt' 
        shapefile_path='x_Mean_building_height_points_' + self.city_name + '.shp' 
        reference_raster_path='A_Building_Count_' + self.city_name + '.tif'
        
        # Read the existing raster for bounds and transforms
        with rasterio.open(reference_raster_path) as existing_raster:
            transform = existing_raster.transform
            bounds = existing_raster.bounds
            width = existing_raster.width
            height = existing_raster.height
            crs = existing_raster.crs

        # Read the shapefile
        points = gpd.read_file(shapefile_path)

        # Loop over each area range
        for lower_bound in range(0, 51, 5):
            if lower_bound == 50:
                field_name = f'Ar_50_abo'
                new_raster_path = f'{base_raster_path}/Sum_area_50_above_build_hgt_{self.city_name}.tif'
            else:
                upper_bound = lower_bound + 5
                field_name = f'Ar_{lower_bound}_{upper_bound}'

                # Check if field exists
                if field_name not in points.columns:
                    print(f"Field {field_name} not found in the shapefile.")
                    continue

                new_raster_path = f'{base_raster_path}/Sum_area_{lower_bound}_{upper_bound}_build_hgt_{self.city_name}.tif'

            # Create the raster for the current field
            with rasterio.open(new_raster_path, 'w', driver='GTiff', height=height, width=width,
                              count=1, dtype=rasterio.float32, crs=crs, transform=transform,
                              nodata=9999.0) as dst:
                sums = np.full((height, width), 9999.0, dtype=np.float32)  # Initialize with NoData value
                for point, value in zip(points.geometry, points[field_name]):
                    row, col = dst.index(point.x, point.y)
                    if bounds.left <= point.x <= bounds.right and bounds.bottom <= point.y <= bounds.top:
                        if 0 <= row < height and 0 <= col < width:
                            if sums[row, col] == 9999.0:  # If currently NoData, make it 0 before summing
                                sums[row, col] = 0.0
                            sums[row, col] += value

                # Write the sum to the raster file
                dst.write(sums, 1)


    def process_ucp2_rasters(self):
        """
        Process and generate UCP2 rasters based on specific conditions.
        """
    
        base_path = './Dist_building_hgts/Sum_area_bins_build_hgt/' 
        output_base_path = './Dist_building_hgts/UCP2_Dist_build_Hgt_UCPs/'
        tiff2_path = 'D_Total_Shape_Area_' + self.city_name + '.tif'
    
        # Open the second tiff file
        with rasterio.open(tiff2_path) as src2:
            tiff2_data = src2.read(1)
            nodata = 9999.0

        # Loop through the range of files
        for lower_bound in range(0, 51, 5):
            if lower_bound == 50:
                tiff1_path = f'{base_path}Sum_area_50_above_build_hgt_{self.city_name}.tif'
                output_tiff_path = f'{output_base_path}UCP2_Dist_Build_Hgts_50_above_{self.city_name}.tif'
            else:
                upper_bound = lower_bound + 5
                tiff1_path = f'{base_path}Sum_area_{lower_bound}_{upper_bound}_build_hgt_{self.city_name}.tif'
                output_tiff_path = f'{output_base_path}UCP2_Dist_Build_Hgts_{lower_bound}_{upper_bound}m_{self.city_name}.tif'

            # Open and read the tiff file
            with rasterio.open(tiff1_path) as src1:
                tiff1_data = src1.read(1)
                meta = src1.meta

            # Perform division except the NoData values
            result = np.where((tiff1_data != nodata) & (tiff2_data != nodata) & (tiff2_data != 0),
                              tiff1_data / tiff2_data, nodata)

            # Update metadata for output raster and write the result as tiff file
            meta.update(dtype=rasterio.float32)
            with rasterio.open(output_tiff_path, 'w', **meta) as dst:
                dst.write(result, 1)
                
    
    def raster_division(self, tiff1_path, tiff2_path, output_tiff_path):
        
        with rasterio.open(tiff1_path) as src1:
            tiff1_data = src1.read(1)
            meta = src1.meta

        # Open the second TIFf file'
        with rasterio.open(tiff2_path) as src2:
            tiff2_data = src2.read(1)

        # Set nodata value
        nodata = 9999.0

        # Perform division by handling with nodata values
        result = np.where((tiff1_data != nodata) & (tiff2_data != nodata) & (tiff2_data != 0),
                         tiff1_data / tiff2_data, nodata)

        # Update metadata for output raster
        meta.update(dtype = rasterio.float32)

        # Write the result to a new tiff file
        with rasterio.open(output_tiff_path, 'w', **meta) as dst:
            dst.write(result, 1)
            
            
    def square_root_raster_division(self, tiff1_path, tiff2_path, output_tiff_path):
        
        with rasterio.open(tiff1_path) as src1:
            tiff1_data = src1.read(1)
            meta = src1.meta

        # Open the second TIFf file'
        with rasterio.open(tiff2_path) as src2:
            tiff2_data = src2.read(1)

        # Set nodata value
        nodata = 9999.0
        
        # Perform division by handling with nodata values
        result = np.where((tiff1_data != nodata) & (tiff2_data != nodata) & (tiff2_data != 0),
                 np.sqrt(tiff1_data / tiff2_data), nodata)

        # Update metadata for output raster
        meta.update(dtype = rasterio.float32)

        # Write the result to a new tiff file
        with rasterio.open(output_tiff_path, 'w', **meta) as dst:
            dst.write(result, 1)
        
    
                
    def process_ucp3_rasters(self):
        tiff1_path = f'E_Building_Volume_Sum_{self.city_name}.tif' 
        tiff2_path = f'D_Total_Shape_Area_{self.city_name}.tif' 
        output_tiff_path = f'UCP3_Area_Weighted_Mean_Building_Height_{self.city_name}.tif'
        self.raster_division(tiff1_path, tiff2_path, output_tiff_path)
            
    
    def process_ucp4_rasters(self, tiff1_path = 'Building_Heights_Dev_Same_Extent.tif', tiff2_path = 'A_Building_Count_Same_Extent.tif'):
        output_tiff_path = f"UCP4_StdDev_Building_Height_{self.city_name}.tif"
        
        
        self.square_root_raster_division(tiff1_path, tiff2_path, output_tiff_path)
    
    
    def divide_tiff_by_value(self, input_tiff, output_tiff, divisor):
        """
        Reads a TIFF file, divides each cell value by the given divisor, 
        and saves the result to a new TIFF file.

        Parameters:
        input_tiff (str): Path to the input TIFF file.
        output_tiff (str): Path to the output TIFF file.
        divisor (float): The value by which to divide each cell. Here, it should be 25000
        """

        # Open the input TIFF file
        with rasterio.open(input_tiff) as src:
            # Read the data
            data = src.read()

            # Set nodata value
            nodata = 9999.0

            # Perform division by handling with nodata values
            result = np.where(data != nodata, data / divisor, nodata)

            # Copy the metadata
            meta = src.meta.copy()

            # Update metadata for output raster
            meta.update(dtype = rasterio.float32)

            # Write the new data to a new TIFF file
            with rasterio.open(output_tiff, 'w', **meta) as dst:
                dst.write(result)
                
    
    def ucp5(self):
        input_tiff = f'D_Total_Shape_Area_{self.city_name}.tif' 
        output_tiff = f'UCP5_{self.city_name}.tif'
        divisor = 250000.0
        self.divide_tiff_by_value(input_tiff, output_tiff, divisor)
        
    def ucp6a(self):
        tiff1_path = f'F_Building_Area_Sum_{self.city_name}.tif' 
        tiff2_path = f'D_Total_Shape_Area_{self.city_name}.tif' 
        output_tiff_path = f'UCP6_a_{self.city_name}.tif'
        self.raster_division(tiff1_path, tiff2_path, output_tiff_path)
        
    def ucp6b(self):
        input_tiff = f'F_Building_Area_Sum_{self.city_name}.tif' 
        output_tiff = f'UCP6_b_{self.city_name}.tif' 
        divisor=250000.0
        self.divide_tiff_by_value(input_tiff, output_tiff, divisor)
