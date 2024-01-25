from glob import glob
import os
import geopandas as gpd
#from pymoo.indicators import get_performance_indicator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.speedups
#from pymoo.factory import get_visualization
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import utils

shapely.speedups.enable()

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

def base_optimization(values_df, budget, rx_burn_units, prevention_df):

    algorithm = NSGA2(
        pop_size = 200,
        sampling = utils.MySampling(),
        crossover = utils.BinaryCrossover(),
        mutation = utils.MyMutation(),
        eliminate_duplicates = True)
    
    problem = utils.BaseProblem(values_df, budget, rx_burn_units, prevention_df)

    res1 = minimize(problem, algorithm, ('n_gen', 1000), seed=1, verbose=True, callback = utils.GenCallback(), save_history=True)
    val200 = res1.algorithm.callback.data['gen200']
    val500 = res1.algorithm.callback.data['gen500']
    val1000 = res1.algorithm.callback.data['gen1000']

    res1_200 = val200[0]
    res1_500 = val500[0]
    res1_1000 = val1000[0]

    return res1, res1_200, res1_500, res1_1000

def get_results_subset(res):

    result_subsets = []
    if res.X.any():
        for subset in res.X:
            result_subsets.append(np.where(subset)[0])
    return np.array(result_subsets)

def calculate_hypervolumes(res1_200, res1_500, res1_1000):
    ref = [0,0,0]
    hv_base = HV(ref_point = ref)
    print("hv for base_formulation at 200 gens", hv_base.do(res1_200))
    print("hv for base_formulation at 500 gens", hv_base.do(res1_500))
    print("hv for base_formulation at 1000 gens", hv_base.do(res1_1000))
    return hv_base


def plot_results(callback_val, res, plot_file):
    print("Function values at gen " + callback_val + ": %s" % res)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2])
    ax.view_init(-30, 30)
    ax.set_xlabel('Burn_Area')
    ax.set_ylabel('Bldg_Dmg')
    ax.set_zlabel('Habitat_Dmg')
    plt.savefig(plot_file)
    plt.show()

def save_results(res1, result_subsets, 
        base_formulation_gen_res,
        base_formulation_gen_res_sub):
    np.savetxt(base_formulation_gen_res, X = res1.F)
    np.savetxt(base_formulation_gen_res_sub, X = result_subsets)


def main(
        rx_burn_units_dir, 
        values_file_path, 
        prevention_file_path, 
        budget,
        solutions_file_path,
        solutions_values_file_path,
        base_formulation_gen_res,
        base_formulation_gen_res_sub):
   
   print("Run main function")

   values_df = pd.read_csv(values_file_path)
   print("Read Values_table from file")

   up_prevention_df = pd.read_csv(prevention_file_path, converters = {'covered_raster_ids': converter})
   prevention_df = up_prevention_df
   
   rx_burn_units_path = glob(os.path.join(rx_burn_units_dir, '*.shp'))[0]
   
   rx_burn_units = gpd.read_file(rx_burn_units_path)
   rx_burn_units = rx_burn_units.to_crs('epsg:32610')

   res1, res1_200, res1_500, res1_1000 = base_optimization(values_df, budget, rx_burn_units, prevention_df)
   result_subsets = get_results_subset(res1)

   save_results(res1, result_subsets, 
                base_formulation_gen_res,
                base_formulation_gen_res_sub)

   hv_base = calculate_hypervolumes(res1_200, res1_500, res1_1000)

   print('Hypervolume for base formulation =', hv_base)
   np.savetxt(solutions_file_path, result_subsets, fmt='%i', delimiter=",")
   np.savetxt(solutions_values_file_path, -res1_1000, delimiter=",", header="burned_area,building_damage,habitat_damage")