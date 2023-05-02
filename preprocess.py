import os
import geopandas as gpd

from glob import glob
import numpy as np
import pandas as pd
import rasterio
import shapely.speedups
from shapely.geometry import Point

shapely.speedups.enable()

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_truncated_ignitions(full_ignitions_df, burn_file_names):
    #TODO :FILTER OUT THE INTERSECTING RASTERS
    # raster_names = list(map(lambda x: os.path(x).stem, burn_file_names))
    raster_names = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], burn_file_names))
    
    raster_names = list(map(lambda x: remove_prefix(x, 'burned_area-'), raster_names))

    full_ignition_idx = full_ignitions_df['filename'].isin(raster_names)
    truncated_ignitions_df = full_ignitions_df[full_ignition_idx]
    truncated_ignitions_df.reset_index(drop = True, inplace = True)

    return truncated_ignitions_df

## Call to get truncated ignitions list
#truncated_ignitions_df = get_truncated_ignitions(full_ignitions_df, burn_file_names)

def get_burn_area_values(truncated_ignitions_df, burned_area_dir, bldg_dmg_file_names, habitat_dmg_file_names):
    values_df = pd.DataFrame(columns = ['filename', 'x_ignition', 'y_ignition', 
                                    'burn_area', 'bldg_dmg', 'habitat_dmg'])
    values_df[['filename', 'x_ignition', 'y_ignition']] =  truncated_ignitions_df[['filename', 'x_ignition', 'y_ignition']]

    burn_area_values = []
    bld_dmg_values = []
    habitat_dmg_values = []

    for name in values_df.filename:
        file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        raster = rasterio.open(file_name)
        img = raster.read(1)
        burn_val = np.sum(img)
        burn_area_values.append(burn_val)
    burn_area_values = np.array(burn_area_values)

    values_df['burn_area'] = burn_area_values.tolist()

    bld_dmg_values = np.zeros(shape = burn_area_values.shape)

    for file_name in bldg_dmg_file_names:
    # file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        # fire_number = remove_prefix(os.path(file_name).stem, 'building_damage-intensity-')
        fire_number = remove_prefix(os.path.splitext(os.path.basename(file_name))[0], 'building_damage-intensity-')
        fire_index = np.where(values_df.filename == fire_number)[0][0]

        raster = rasterio.open(file_name)
        img = raster.read(1)
        bld_dmg_val = np.sum(img)
        bld_dmg_values[fire_index] = bld_dmg_val
    
    values_df['bldg_dmg'] = bld_dmg_values.tolist() 

    habitat_dmg_values = np.zeros(shape = burn_area_values.shape)
    ct = 1
    for file_name in habitat_dmg_file_names:
    # file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        # fire_number = remove_prefix(os.path(file_name).stem, 'habitat_damage-')
        fire_number = remove_prefix(os.path.splitext(os.path.basename(file_name))[0], 'habitat_damage-')
        # fire_index = np.where(values_df.filename == fire_number)[0][0]
        fire_index = values_df.index[values_df['filename'] == fire_number].tolist()[0]
        try:
            raster = rasterio.open(file_name)
            img = raster.read(1)
            habitat_dmg_val = np.sum(img)
        except:
            print(fire_number)
            habitat_dmg_val = 0.0
        habitat_dmg_values[fire_index] = habitat_dmg_val

    values_df['habitat_dmg'] = habitat_dmg_values.tolist()
    
    return values_df

def write_csv_to_file(file_path, data):
    data.to_csv(file_path, index=False)



def point_in_poly(point, polygon):
    return polygon.contains(Point(point))

def generate_prevention_df(rx_burn_units_path, values_df):
    prevention_df = pd.DataFrame(columns=['geometry','f1', 'f2', 'f3', 'covered_raster_ids'])

    func = lambda x, y : point_in_poly(x, y)
    vector_func = np.vectorize(func)

    f1s = []
    f2s = []
    f3s = []
    covered_ids = []

    ignition_points = [(x, y) for x, y in zip(values_df.x_ignition, values_df.y_ignition)]
    ignition_points = np.array(ignition_points)
    
    rx_burn_units = gpd.read_file(rx_burn_units_path)
    rx_burn_units = rx_burn_units.to_crs('epsg:32610')

    burn_candidates = rx_burn_units.geometry
    contained_idx = None
    for poly in burn_candidates:
        contained_idx = list(map(lambda x: point_in_poly(x, poly), ignition_points))
        covered = np.where(contained_idx)[0]

        f1 = np.sum(values_df[contained_idx].burn_area)
        f2 = np.sum(values_df[contained_idx].bldg_dmg)
        f3 = np.sum(values_df[contained_idx].habitat_dmg)
        
        covered_ids.append(list(covered))
        f1s.append(f1)
        f2s.append(f2)
        f3s.append(f3)

    prevention_df['geometry'] = burn_candidates
    prevention_df['f1'] = f1s
    prevention_df['f2'] = f2s
    prevention_df['f3'] = f3s
    prevention_df['covered_raster_ids'] = covered_ids

    return prevention_df

# Preprocess
def preprocess(
        rx_burn_units_dir, 
        full_ignitions_file_path,
        burned_area_dir,
        bldg_dmg_dir,
        habitat_dmg_dir,
        values_file_path, 
        prevention_file_path):
    
    # Preprocessing 
    full_ignitions_df = pd.read_csv(full_ignitions_file_path)
    
    # full_ignitions_df = []
    burn_file_names = glob(os.path.join(burned_area_dir, '*.tif'))
    bldg_dmg_file_names = glob(os.path.join(bldg_dmg_dir, '*.tif'))
    habitat_dmg_file_names = glob(os.path.join(habitat_dmg_dir, '*.tif'))

    rx_burn_units_path = glob(os.path.join(rx_burn_units_dir, '*.shp'))[0]

    burned_area_tifs = glob(os.path.join(burned_area_dir, '*.tif'))
    print('total simulated burns =', len(burned_area_tifs))

    print("Run main function")

    ## Call when values_df needs to be created
    truncated_ignitions_df = get_truncated_ignitions(full_ignitions_df, burn_file_names)
    values_df = get_burn_area_values(truncated_ignitions_df, burned_area_dir, bldg_dmg_file_names, habitat_dmg_file_names)
    write_csv_to_file(values_file_path, values_df)

    ## Call to generate prevention_df
    prevention_df = generate_prevention_df(rx_burn_units_path, values_df)
    write_csv_to_file(prevention_file_path, prevention_df)

    print("Generated prevention df")