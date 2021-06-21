import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import pandas as pd
import bisect
import random
import time
import warnings

#from ..neutrality_analysis import BrayCurtis, BrayCurtis_neutrality, KullbackLeibler, KullbackLeibler_neutrality, JensenShannon
from scipy.stats import pearsonr
#from ..variation import variation_coefficient, JS
#from .models import MODEL


class IBM(Enum):
    SOLE = 1 # model from Solé et al. 2002 Phil Trans Roy Soc B (https://doi.org/10.1098/rstb.2001.0992)
    HEYVAERT = 2 # model from Heyvaert 2017 master thesis (https://lib.ugent.be/nl/catalog/rug01:002349815) this is adaptation from Solé above

class WeightedRandomGenerator(object):
    use_lattice = True

    def __init__(self, weights, r=0):
        if self.use_lattice:
            self.lattice = []
            for species in range(len(weights)):
                for i in range(weights[species]):
                    self.lattice.append(species)
        else:
            # code from https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
            # faster than using np random choice
            self.totals = []
            running_total = 0

            if r == 0:
                r = int(time.time())
            random.seed(r)

            for w in weights:
                running_total += w
                self.totals.append(running_total)


    def __call__(self):
        if self.use_lattice:
            return random.choice(self.lattice)
        else:
            return bisect.bisect_right(self.totals, random.random() * self.totals[-1])

class Timeseries():
    def __init__(self, params, model=IBM.HEYVAERT, T=100, f=None, seed=None):
        self.params = params
        self.model = model
        self.T = T
        self.f = f
        
        self.set_seed(seed)

        self.check_input_parameters()
        self.set_parameters(params)
        self.set_initial_condition()

        self.add_step = self.add_step_function()

        if self.f != None:
            self.write_header()

        self.integrate()

    def set_seed(self, seed):
        np.random.seed(seed if seed != None else int(time.time()))

    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        # number of species
        self.Nspecies = len(self.interaction_matrix)  
        
        # matrix combining growth and interaction matrix, growth rate in first column (growth is interaction with empty sites)
        self.growth_interaction_matrix = np.zeros([self.Nspecies + 1, self.Nspecies + 1])
        self.growth_interaction_matrix[1:, 1:] = self.interaction_matrix
        self.growth_interaction_matrix[1:, 0] = self.growth_rate.flatten()

        # SISvector = Strongly interacting species (multiplication factor to strength of interactions with other species)
        self.SISvector = np.hstack(([1], self.SISvector))
        
        # zero immigration / extinction for empty sites
        self.immigration_rate = np.hstack(([0], self.immigration_rate.flatten()))
        self.extinction_rate = np.hstack(([0], self.extinction_rate.flatten()))

    def set_initial_condition(self):
        # add number of empty sites as a pseudospecies at index 0
        empty_sites = [self.sites - sum(self.initial_condition)]
        self.x = np.append(empty_sites, self.initial_condition)

    def check_input_parameters(self):
        # Function to check if all necessary parameters where provided, raises error if parameters are missing

        if self.model in [IBM.HEYVAERT]:
            parlist = ['interaction_matrix', 'immigration_rate', 'extinction_rate',
                       'initial_condition', 'SISvector', 'growth_rate', 'sites']

        for par in parlist:
            if not par in self.params:
                raise KeyError('Parameter %s needs to be specified for the %s model.' % (
                    par, self.model.name))
        for label in self.params:
            if label not in parlist:
                warnings.warn('Parameter %s was not used.' % label)

        if self.params['sites'] < np.sum(self.params['initial_condition']):
            raise KeyError('Total number of species of initial condition may not be higher than number of sites.')

    def write_header(self):
        # Write down header in file f
        with open(self.f, "a") as file:
            file.write("time")
            for k in range(1, self.Nspecies + 1):
                file.write(",species_%d" % k)

            file.write("\n")

            file.write("%.3E" % 0)
            for k in self.initial_condition:
                file.write(",%.3E" % k)
            file.write("\n")

    def integrate(self):
        x_ts = np.zeros([int(self.T), self.Nspecies])

        # set initial condition
        x_ts[0] = self.x[1:] # do not save empty sites

        # Integrate
        for i in range(1, int(self.T)):
            self.add_step(self)

            # Save abundances
            if self.f != None:
                self.write_abundances_to_file(i)

            x_ts[i] = self.x[1:] # do not save empty sites

            if np.all(np.isnan(self.x)):
                break

        # dataframe to save sglv_timeseries
        self.x_ts = pd.DataFrame(x_ts, columns=['species_%d' % i for i in range(1, self.Nspecies + 1)])
        self.x_ts['time'] = np.arange(0, int(self.T))

        return

    def add_step_function(self):
        if self.model == IBM.SOLE:
            def func(self):
                # TODO : implement Solé model (exact implementation unclear from paper)
                return 0

        elif self.model == IBM.HEYVAERT:
            def func(self):
                x = self.x

                # initialize the so-called "interaction vector"
                y_virtual = x * self.SISvector

                # the number of interactions
                N_interactions = int(sum(y_virtual))

                # random number generator
                randomx = WeightedRandomGenerator(x)
                randomy = WeightedRandomGenerator(y_virtual)

                for _ in range(N_interactions):
                    # choose a random species B from virtual lattice
                    A = randomx()
                    B = randomy()

                    ext = True

                    # interaction between A and B or growth of A if B is empty site
                    if A != 0 and random.random() < abs(self.growth_interaction_matrix[A][B]):
                        if self.growth_interaction_matrix[A][B] < 0 and x[A] > 0:
                            x[0] += 1
                            x[A] -= 1
                            ext = False
                        elif self.growth_interaction_matrix[A][B] > 0 and x[0] > 0:
                            x[0] -= 1
                            x[A] += 1

                    if A == 0 and x[0] > 0:  # possible immigration event
                        C = random.randint(1, self.Nspecies)
                        if random.random() < self.immigration_rate[C]:
                            x[0] -= 1
                            x[C] += 1
                    else:  # possible emigration event
                        if x[A] > 0 and random.random() < self.extinction_rate[A] and ext:
                            x[0] += 1
                            x[A] -= 1

                self.x = x

        return func

    def write_abundances_to_file(self, i):
        with open(self.f, "a") as file:
            file.write("%.5E" % i)
            for k in self.x:
                file.write(",%.5E" % k)
            if self.model == MODEL.QSMI:
                for k in self.y:
                    file.write(",%.5E" % k)
            file.write("\n")

    @property
    def timeseries(self):
        #make sure indices in right order

        columns = ['time'] + ['species_%d' % i for i in range(1, self.Nspecies + 1)]
        self.x_ts = self.x_ts[columns]

        return self.x_ts

    @property
    def endpoint(self):
        df = pd.DataFrame(self.x[1:], columns=['endpoint'], index=['species_%d' % i for i in range(1, self.Nspecies+1)])
        return df

def test_timeseries():
    print('test Timeseries')

    N = 50
    sites = 5000

    params = {}

    # no interaction
    omega = np.zeros([N, N]);
    np.fill_diagonal(omega, -1)

    params['interaction_matrix'] = omega

    # no immigration
    params['immigration_rate'] = np.zeros([N, 1])

    # different growth rates determined by the steady state
    params['growth_rate'] = np.full([N, 1], 0.3)
    params['extinction_rate'] = np.full([N,1], 0.1)

    params['SISvector'] = np.ones(N, dtype=int)

    params['initial_condition'] = np.random.randint(0, int(sites/N), N)
    
    params['sites'] = sites

    ts = Timeseries(params, T=100, seed=int(time.time()))

    print("sglv_timeseries")
    print(ts.timeseries.head())

    print("endpoint")
    print(ts.endpoint)

    return ts

def main():
    test_timeseries()

if __name__ == "__main__":
    main()