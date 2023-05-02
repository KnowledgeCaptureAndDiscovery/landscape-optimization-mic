from pymoo.core.callback import Callback
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
import numpy as np

class GenCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["gen200"] = []
        self.data["gen500"] = []
        self.data["gen1000"] = []
        # self.data["best"] = []

    def notify(self, algorithm):
      if algorithm.n_gen == 200:
        self.data["gen200"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 500:
        self.data["gen500"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 1000:
        self.data["gen1000"].append(algorithm.opt.get("F"))

class ModGenCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["gen200"] = []
        self.data["gen500"] = []
        self.data["gen1000"] = []
        # self.data["best"] = []

    def notify(self, algorithm):
      if algorithm.n_gen == 1:
        self.data["gen200"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 2:
        self.data["gen500"].append(algorithm.opt.get("F"))
      if algorithm.n_gen == 3:
        self.data["gen1000"].append(algorithm.opt.get("F"))

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X

class BinaryCrossover(Crossover):

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X

class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X

class BaseProblem(ElementwiseProblem):
    def __init__(self,
                 values_df,
                 n_max,
                 rx_burn_units,
                #  ignition_points,
                 prevention_df
                 ):
        super().__init__(n_var=rx_burn_units.shape[0], n_obj=3, n_constr=1)
        self.values_df = values_df
        self.n_max = n_max
        # self.rx_burn_units = rx_burn_units
        # self.ignition_points = ignition_points
        self.prevention_df = prevention_df

    def _evaluate(self, x, out, *args, **kwargs):
        
        # # TODO: MAKE THE FOLLOWING SECTION FASTER, VECTORIZE WHEREVER POSSIBLE
        # plan_burns = self.rx_burn_units.iloc[x]
        
        # # TODO: PREPROCESS THE CONTAINED IGNITIONS FOR EACH RX_BURN_UNIT
        # plan_burns_dissolved = plan_burns.dissolve()
        # plan_polys = plan_burns_dissolved.geometry[0]
        # contained_idx = np.apply_along_axis(lambda x : point_in_poly(x, plan_polys), 1, self.ignition_points)

        f1 = -np.sum(self.prevention_df[x].f1)
        f2 = -np.sum(self.prevention_df[x].f2)
        f3 = -np.sum(self.prevention_df[x].f3)

        out["F"] = [f1, f2, f3]
        out["G"] = (self.n_max - np.sum(x)) ** 2

class OneDimProblem(ElementwiseProblem):
    def __init__(self,
                 n_max,
                 rx_burn_units,
                 function_vals
                 ):
        super().__init__(n_var=rx_burn_units.shape[0], n_obj=1, n_constr=1)
        self.n_max = n_max
        self.rx_burn_units = rx_burn_units
        self.function_vals = function_vals

    def _evaluate(self, x, out, *args, **kwargs):
        
        out["F"] = -np.sum(self.function_vals[x])
        out["G"] = (self.n_max - np.sum(x)) ** 2
