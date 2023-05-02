import sys
import tempfile
import pandas as pd
import numpy as np
import shutil
from glob import glob
import os
import rasterio
import geopandas as gpd

import shapely.speedups
shapely.speedups.enable()

import math
from rasterio.transform import Affine
from time import time_ns
from scipy.sparse import csr_matrix

def main(rx_burn_units_path, values_file_path, prevention_file_path):
    rx_burn_units = gpd.read_file(rx_burn_units_path)
    rx_burn_units = rx_burn_units.to_crs('epsg:32610')

    values_df = pd.read_csv(values_file_path)

    ignition_points = [(x, y) for x, y in zip(values_df.x_ignition, values_df.y_ignition)]
    ignition_points = np.array(ignition_points)
    def converter(instr):
        return np.fromstring(instr.strip('[]'),sep=' ')
     #df1=pd.read_csv('tmp.csv',converters={'Vec':converter})
    prevention_df = pd.read_csv(prevention_file_path, converters = {'covered_raster_ids': converter})
    return rx_burn_units, values_df, ignition_points, prevention_df

def converter(instr):
    return np.fromstring(instr[1:-1],sep=',')


def heatmap_files_generate(
        index,
        heat_array,
        heat_array_bldg,
        heat_array_habi,
        raster,
        transform,
        heatmap_res_dir
):
      
    np.savez_compressed(os.path.join(heatmap_res_dir,"heatmaps/np_heatmaps/benefits_heatmap_{}".format(index + 1)), burned_area=heat_array, building_damage=heat_array_bldg, habitat_damage=heat_array_habi)
    
    with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/summed_raster_heatmaps/summed_raster_heatmap_area_{}.tif'.format(index+1)),
                            'w',
                            #driver='GTiff',
                            height = heat_array.shape[0],
                            width = heat_array.shape[1],
                            count=1,
                            dtype=heat_array.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array, 1)
                dst.close()

    with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/summed_raster_heatmaps/summed_raster_heatmap_habi_{}.tif'.format(index+1)),
                            'w',
                            driver='GTiff',
                            height = heat_array.shape[0],
                            width = heat_array.shape[1],
                            count=1,
                            dtype=heat_array.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_habi, 1)
                dst.close()

    with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/summed_raster_heatmaps/summed_raster_heatmap_bldg_{}.tif'.format(index+1)),
                            'w',
                            driver='GTiff',
                            height = heat_array.shape[0],
                            width = heat_array.shape[1],
                            count=1,
                            dtype=heat_array.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_bldg, 1)
                dst.close()

        
def heatmap(
          values_file_path, 
          prevention_file_path, 
          base_formulation_results_file_path,
          base_formulation_result_subsets_file_path,
          heatmaps_zip_file_path,
          burned_area_dir,
          bldg_dmg_dir,
          habitat_dmg_dir):

    burned_area_tifs = glob(os.path.join(burned_area_dir, '*.tif'))
    print('total simulated burns =', len(burned_area_tifs))

    values_df = pd.read_csv(values_file_path)
    
    up_prevention_df = pd.read_csv(prevention_file_path, converters = {'covered_raster_ids': converter})
    prevention_df = up_prevention_df

    res1 = np.loadtxt(base_formulation_results_file_path)
    result_subsets = np.loadtxt(base_formulation_result_subsets_file_path)

    complete_lb, complete_rb, complete_tb, complete_bb = [753003.071258364, 839057.6399108863, 4133476.546731642, 4230634.714689108]
    print(complete_lb, complete_rb, complete_tb, complete_bb)

    heatmap_res_dir = tempfile.mkdtemp(prefix='heatmap_res_dir_')

    if not os.path.exists(heatmap_res_dir):
        os.makedirs(heatmap_res_dir)
    
    if not os.path.exists(os.path.join(heatmap_res_dir, 'heatmaps')):
        os.makedirs(os.path.join(heatmap_res_dir, 'heatmaps'))
    
    if not os.path.exists(os.path.join(heatmap_res_dir, 'heatmaps', 'np_heatmaps')):
        os.makedirs(os.path.join(heatmap_res_dir, 'heatmaps', 'np_heatmaps'))
    
    if not os.path.exists(os.path.join(heatmap_res_dir, 'heatmaps', 'summed_raster_heatmaps')):
        os.makedirs(os.path.join(heatmap_res_dir, 'heatmaps', 'summed_raster_heatmaps'))
    
    transform = Affine(10, 0.0, complete_lb, 
                    0.0, -10, complete_bb)

    skipped = 0

    plan_heat_array = []
    plan_heat_array_bldg = []
    plan_heat_array_habi = []

    used_mask = np.zeros([len(prevention_df)], dtype=bool)

    for i,plan in enumerate(result_subsets):
        for plan_idx in plan.astype(int):
            used_mask[plan_idx]=True

    for poly_idx in range(len(prevention_df)):
        if used_mask[poly_idx] == False:
            plan_heat_array.append([])
            plan_heat_array_bldg.append([])
            plan_heat_array_habi.append([])
            continue
        
        xdim = int(math.ceil(complete_bb - complete_tb))//10
        ydim = int(math.ceil(complete_rb - complete_lb))//10
        
        heat_array = np.zeros([xdim, ydim])
        heat_array_bldg = np.zeros([xdim, ydim])
        heat_array_habi = np.zeros([xdim, ydim])

        # print('heat array init done')
        # poly = prevention_df.iloc[poly_idx].geometry
        # raster1 = rasterio.open(burned_area_tifs[prevention_df.iloc[poly_idx]['covered_raster_ids'][0].astype(int)])

        for raster_num in prevention_df.iloc[poly_idx]['covered_raster_ids'].astype(int):
            # rasters_to_merge = []
            # file_name = values_df.iloc[raster_num]['filename']
            raster_row = values_df.iloc[raster_num]
            file_name = raster_row['filename']

            raster = rasterio.open(burned_area_dir + 'burned_area-' + file_name + '.tif')
            xmin, ymin, xmax, ymax = raster.bounds
            img = raster.read(1)
            try:
                heat_array[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
            except:
                skipped += 1
                continue
            raster.close()

            if raster_row['bldg_dmg'].astype(int) > 0:
                raster = rasterio.open(bldg_dmg_dir + 'building_damage-intensity-' + file_name + '.tif')
                xmin, ymin, xmax, ymax = raster.bounds
                # print(xmin, ymin, xmax, ymax)
                img = raster.read(1)
                # img = np.flip(img, 0)
                # print(img.shape, raster.bounds)
                try:
                    heat_array_bldg[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
                except:
                    skipped += 1
                    continue
                raster.close()

            if raster_row['habitat_dmg'].astype(int) > 0:
                raster = rasterio.open(habitat_dmg_dir + 'habitat_damage-' + file_name + '.tif')
                xmin, ymin, xmax, ymax = raster.bounds
                # print(xmin, ymin, xmax, ymax)
                img = raster.read(1)
                # img = np.flip(img, 0)
                # print(img.shape, raster.bounds)
                try:
                    heat_array_habi[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
                except:
                    skipped += 1
                    continue
                raster.close()
        
        plan_heat_array.append(csr_matrix(heat_array))
        plan_heat_array_bldg.append(csr_matrix(heat_array_bldg))
        plan_heat_array_habi.append(csr_matrix(heat_array_habi))
        
        print(f"heat array done: {poly_idx}")


    raster = rasterio.open(burned_area_tifs[1])

    for index, plan in enumerate(result_subsets[0:]):
        print("Plan ", index, " - Skipped: ", skipped)
        start = time_ns()

        heat_array = np.sum([plan_heat_array[idx] for idx in plan.astype(int)], axis=0).toarray()
        heat_array_bldg = np.sum([plan_heat_array_bldg[idx] for idx in plan.astype(int)], axis=0).toarray()
        heat_array_habi = np.sum([plan_heat_array_habi[idx] for idx in plan.astype(int)], axis=0).toarray()

        heatmap_files_generate(
            index,
            heat_array, 
            heat_array_bldg, 
            heat_array_habi, 
            raster, transform,
            heatmap_res_dir)
        
        del heat_array
        del heat_array_bldg
        del heat_array_habi

        end = time_ns() - start
        print("Time taken: " + str(end/(10**9)/60) + " mins")
    
    raster.close()

    shutil.make_archive(heatmaps_zip_file_path, 'zip', heatmap_res_dir)
    os.rename(heatmaps_zip_file_path + '.zip', heatmaps_zip_file_path)
    
    shutil.rmtree(heatmap_res_dir)
