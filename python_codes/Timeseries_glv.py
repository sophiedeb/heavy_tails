import numpy as np
import pandas as pd
import time
import warnings
from scipy.stats import pearsonr
from python_codes.brownian_glv import brownian
#import brownian_glv #.brownian


from python_codes.neutrality_analysis import BrayCurtis, BrayCurtis_neutrality, KullbackLeibler, KullbackLeibler_neutrality, JensenShannon
from python_codes.variation import variation_coefficient, JS
from python_codes.noise_parameters import NOISE
from python_codes.models import MODEL

def make_params(steadystate,
                interaction=0, selfint=1,
                immigration=0, init_dev=0.1,
                noise=1e-1, connectivity=1):
    """
    Return set of parameters: interaction matrix, growth rate, immigration rate, noise and initial condition

    :param steadystate: steady state, list of floats
    :param interaction: interaction strength, float
    :param selfint: self interaction, float or list of floats
    :param immigration: immigration rate, float or list of floats
    :param init_dev: deviation from steady state for initial condition, float
    :param noise: amount of noise, float
    :param connectivity: connectivity, float between 0 and 1
    :return: dictionary of parameters
    """
    params = {}

    n = len(steadystate)

    if interaction == 0:
        omega = np.zeros([n, n]);
    else:
        omega = np.random.normal(0, interaction, [n,n])
        omega *= np.random.choice([0, 1], omega.shape, p=[1-connectivity, connectivity])
    np.fill_diagonal(omega, -selfint)

    params['interaction_matrix'] = omega

    params['immigration_rate'] = np.full(steadystate.shape, immigration)

    # different growth rates determined by the steady state
    params['growth_rate'] = - (omega).dot(steadystate)

    params['initial_condition'] = np.copy(steadystate) * np.random.normal(1, init_dev, steadystate.shape)

    params['noise_linear'] = noise
    params['noise_sqrt'] = noise
    params['noise_constant'] = noise

    return params

def Jacobian(intmat, ss, K=None):
    J = np.diag(ss.flatten()).dot(intmat)

    return J

# TODO check shape of steadystate
def is_stable(steadystate, interaction_matrix):
    """
    Checks whether steady state is stable solution of generalized Lotka Volterra system with given interaction matrix

    :param steadystate: np.array with steady state
    :param interaction_matrix: np.array with interaction matrix
    :return: bool: stability of steady state
    """

    # Jacobian
    Jac = Jacobian(interaction_matrix, steadystate)

    if np.any(np.real(np.linalg.eigvals(Jac)) > 0):
        return False
    else:
        return True

def test_validity_Jacobian():
    """
    Test to check whether the Jacobian used in the is_stable function is correctly defined
    Print statements
    :return: bool if definition is correct
    """

    def numeric_Jacobian(intmat, ss, K):
        epsilon = 1e-5

        def f(x):
            return x * (K + np.dot(intmat, x))

        J = np.zeros(intmat.shape)

        for i in range(len(J)):
            dx = np.zeros(ss.shape);
            dx[i] = epsilon
            J[:, i] = ((f(ss + dx) - f(ss - dx)) / (2 * epsilon)).flatten()

        return J

    n = 6

    ss = np.random.uniform(0, 5, [n, 1])  # np.ones([N,1])
    intmat = np.random.normal(0, 3, [n, n])
    K = - np.dot(intmat, ss)

    Jac = Jacobian(intmat, ss, K)
    numJac = numeric_Jacobian(intmat, ss, K)

    print("Jacobian", Jac)
    print("numeric Jacobian", numJac)
    print("maximal relative difference", np.max(abs((numJac - Jac)/Jac)))
    return np.max(abs((numJac - Jac)/Jac)) < 1e-6

class Timeseries():
    def __init__(self, params, model=MODEL.GLV, noise_implementation=NOISE.LANGEVIN_LINEAR, dt=0.01, T=100, tskip=0,
                 f=0, seed=None):
        self.model = model
        self.noise_implementation = noise_implementation
        self.dt = dt
        self.T = T
        self.tskip = tskip
        self.f = f
        
        self.set_seed(seed)

        self.check_input_parameters(params)
        self.set_parameters(params)

        self.init_Nspecies_Nmetabolites()

        self.deterministic_step = self.deterministic_step_function()
        self.stochastic_step = self.stochastic_step_function()
        self.add_step = self.add_step_function()

        self.set_initial_condition()

        if self.f != 0:
            self.write_header()

        self.integrate()

    def set_seed(self, seed):
        np.random.seed(seed if seed != None else int(time.time()))

    def set_parameters(self, params):
        for key, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                setattr(self, key, value.copy())
            else:
                setattr(self, key, value)

        if 'extinction_rate' not in params:
            self.extinction_rate = np.zeros_like(self.growth_rate)

        if self.model == MODEL.GLV:
            # no distinction between growth and extinction terms
            self.growth_rate -= self.extinction_rate

        if 'MAX' in self.model.name:
            # separate positive and negative interaction terms
            self.interaction_matrix_pos = self.interaction_matrix.copy();
            self.interaction_matrix_pos[self.interaction_matrix < 0] = 0;

            self.interaction_matrix_neg = self.interaction_matrix.copy();
            self.interaction_matrix_neg[self.interaction_matrix > 0] = 0;

            self.growth_rate_pos = self.growth_rate.copy();
            self.growth_rate_pos[self.growth_rate < 0] = 0;

            self.growth_rate_neg = self.growth_rate.copy();
            self.growth_rate_neg[self.growth_rate > 0] = 0;
            self.growth_rate_neg -= self.extinction_rate;

    def set_initial_condition(self):
        if self.model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
            self.x = np.copy(self.initial_condition)
        elif self.model == MODEL.QSMI:
            self.x = np.copy(self.initial_condition)[:len(self.d)]  # initial state species
            self.y = np.copy(self.initial_condition)[len(self.d):]  # initial state metabolites

    def check_input_parameters(self, params):
        # Function to check if all necessary parameters where provided, raises error if parameters are missing

        if self.model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
            parlist = ['interaction_matrix', 'immigration_rate', 'growth_rate', 'initial_condition']

            if 'LINEAR' in self.noise_implementation.name:
                parlist += ['noise_linear']
            elif 'SQRT' in self.noise_implementation.name:
                parlist += ['noise_sqrt']
            elif 'CONSTANT' in self.noise_implementation.name:
                parlist += ['noise_constant']
            elif 'INTERACTION' in self.noise_implementation.name:
                parlist += ['noise_interaction']

            if 'MAX' in self.model.name:
                parlist += ['maximum_capacity']

        elif self.model == MODEL.QSMI:
            parlist = ['psi', 'd', 'g', 'dm', 'kappa', 'phi', 'initcond', 'noise']

        if 'MAX' in self.noise_implementation.name:
            parlist += ['maximum_capacity']

        for par in parlist:
            if not par in params:
                raise KeyError('Parameter %s needs to be specified for the %s model and %s noise implementation.' % (
                    par, self.model.name, self.noise_implementation.name))
        for label in params:
            if label != 'extinction_rate' and label not in parlist:
                warnings.warn('Parameter %s was not used.' % label)

        # check whether matrix shapes are correct
        if self.model == MODEL.GLV:
            if not np.all(len(row) == len(params['interaction_matrix']) for row in
                          params['interaction_matrix']):
                raise ValueError('Interaction matrix is not square.')

            for parname in ['immigration_rate', 'growth_rate', 'initial_condition']:
                if np.any(params[parname].shape != (params['interaction_matrix'].shape[0], 1)):
                    raise ValueError('%s has the incorrect shape: %s instead of (%d,1)' % (
                        parname, str(params[parname].shape), params['interaction_matrix'].shape[0]))

    def write_header(self):
        # Write down header in file f
        with open(self.f, "a") as file:
            file.write("time")
            for k in range(1, self.Nspecies + 1):
                file.write(",species_%d" % k)
            for k in range(1, self.Nmetabolites + 1):
                file.write(",metabolite_%d" % k)

            file.write("\n")

            file.write("%.3E" % 0)
            for k in self.initial_condition:
                file.write(",%.3E" % k)
            file.write("\n")

    def init_Nspecies_Nmetabolites(self):
        if self.model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
            self.Nspecies = len(self.interaction_matrix)  # number of species
            self.Nmetabolites = 0  # number of metabolites, 0 in the GLV models
        elif self.model == MODEL.QSMI:
            self.Nspecies = len(self.d)  # number of species
            self.Nmetabolites = len(self.dm)  # number of metabolites

    def integrate(self):
        # If noise is Ito, first generate brownian motion.
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt = np.zeros_like(self.initial_condition)
            self.bm = brownian(np.zeros(len(self.initial_condition)), int(self.T / self.dt), self.dt, 1,
                               out=None)

        # initialize array for sglv_timeseries
        x_ts = np.zeros([int(self.T / (self.dt * (self.tskip + 1))), self.Nspecies])

        # set initial condition
        x_ts[0] = self.x.flatten()

        # Integrate ODEs according to model and noise
        for i in range(1, int(self.T / (self.dt * (self.tskip + 1)))):
            for j in range(self.tskip + 1):
                self.add_step(self, i * (self.tskip + 1) + j)

            # Save abundances
            if self.f != 0:
                self.write_abundances_to_file(i * (self.tskip + 1) + j)

            x_ts[i] = self.x.flatten()

            if np.all(np.isnan(self.x)):
                break

        # transform array of sglv_timeseries to dataframe
        self.x_ts = pd.DataFrame(x_ts, columns=['species_%d' % i for i in range(1, self.Nspecies + 1)])
        self.x_ts['time'] = (self.dt * (self.tskip + 1) * np.arange(0, int(self.T / (self.dt * (self.tskip + 1)))))

        return

    def add_step_function(self):
        if self.model in [MODEL.GLV]:
            if ('LANGEVIN' in self.noise_implementation.name or 'MILSTEIN' in self.noise_implementation.name):
                if 'LANGEVIN_LINEAR' in self.noise_implementation.name and self.noise_linear == 0:
                    def func(self, i):
                        self.x += self.deterministic_step(self)

                        # abundance cannot be negative
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.x[self.x < 0] = 0
                else:
                    def func(self, i):
                        dx_det = self.deterministic_step(self)
                        dx_stoch = self.stochastic_step(self)

                        self.x += dx_det + dx_stoch

                        # abundance cannot be negative
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.x[self.x < 0] = 0

            elif 'RICKER' in self.noise_implementation.name:
                def func(self, i):
                    self.ricker_step()

            elif 'ARATO' in self.noise_implementation.name:
                def func(self, i):
                    self.arato_step(i)

        elif 'MAX' in self.model.name:
            if ('LANGEVIN' in self.noise_implementation.name or 'MILSTEIN' in self.noise_implementation.name):
                if 'LANGEVIN_LINEAR' in self.noise_implementation.name and self.noise_linear == 0:
                    # zero noise
                    def func(self, i):
                        self.probability_to_grow = 1 - max(0, min(1, np.sum(self.x)/self.maximum_capacity))

                        self.x += self.deterministic_step(self)

                        # abundance cannot be negative
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.x[self.x < 0] = 0
                else:
                    # with noise
                    def func(self, i):
                        self.probability_to_grow = 1 - max(0, min(1, np.sum(self.x)/self.maximum_capacity))

                        dx_det = self.deterministic_step(self)
                        dx_stoch = self.stochastic_step(self)

                        self.x += dx_det + dx_stoch

                        # abundance cannot be negative
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.x[self.x < 0] = 0

        elif self.model == MODEL.QSMI:
            def func(self, i):
                dx_det, dy_det = self.deterministic_step()

                # TODO implement the stochastic version of QMSI

                self.x += dx_det
                self.y += dy_det

                self.x = self.x.clip(min=0)
                self.y = self.y.clip(min=0)

        return func

    def deterministic_step_function(self):
        if self.model == MODEL.GLV:
            def func(self):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return ((self.interaction_matrix.dot(self.x) +  self.growth_rate) * self.x + self.immigration_rate) * self.dt
        
        if self.model in [MODEL.MAX]:
            def func(self):
                return (self.immigration_rate * self.probability_to_grow + (
                    (self.interaction_matrix_pos * self.probability_to_grow + self.interaction_matrix_neg).dot(
                        self.x) + self.growth_rate_pos * self.probability_to_grow + self.growth_rate_neg) * self.x) * self.dt

        if self.model in [MODEL.MAX_IMMI]:
            def func(self):
                return (self.immigration_rate + (
                    (self.interaction_matrix_pos * self.probability_to_grow + self.interaction_matrix_neg).dot(
                        self.x) + self.growth_rate_pos * self.probability_to_grow + self.growth_rate_neg) * self.x) * self.dt

        elif self.model == MODEL.QSMI:
            if self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                def func(self):
                    dx = self.x * (self.psi.dot(self.y) - self.d) * self.dt
                    dy = (self.g - self.dm * self.y - self.y * self.kappa.dot(self.x) + (
                    (self.phi.dot(self.x)).reshape([self.Nmetabolites, self.Nmetabolites])).dot(
                        self.y)) * self.dt
                    return dx, dy

        return func

    def stochastic_step_function(self):
        if self.model in [MODEL.GLV, MODEL.MAX, MODEL.MAX_IMMI]:
            if self.noise_implementation == NOISE.LANGEVIN_LINEAR:
                def func(self):
                    return self.noise_linear * self.x * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape)
            elif self.noise_implementation == NOISE.LANGEVIN_LINEAR_MAX:
                def func(self):
                    g = np.random.standard_normal(self.x.shape)
                    g[g > 0] *= self.probability_to_grow
                    return self.noise_linear * self.x * np.sqrt(self.dt) * g
            elif self.noise_implementation == NOISE.GROWTH_AND_INTERACTION_LINEAR:
                def func(self):
                    return (
                    self.noise_linear * self.x * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape) + (
                    self.noise_interaction * np.random.standard_normal(self.interaction_matrix.shape)).dot(
                        self.x) * self.x * np.sqrt(self.dt))
            elif self.noise_implementation == NOISE.LANGEVIN_SQRT:
                def func(self):
                    return self.noise_sqrt * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape)
            elif self.noise_implementation == NOISE.LANGEVIN_LINEAR_SQRT:
                def func(self):
                    return self.noise_linear * self.x * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape) + \
                           self.noise_sqrt * np.sqrt(self.x) * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape)
            elif self.noise_implementation == NOISE.SQRT_MILSTEIN:
                def func(self):
                    dW = np.sqrt(self.dt) * np.random.standard_normal(self.x.shape)
                    return np.sqrt(self.noise_sqrt * self.x) * dW + self.noise_sqrt ** 2 / 4 * (dW ** 2 - self.dt ** 2)

            elif self.noise_implementation == NOISE.LANGEVIN_CONSTANT:
                def func(self):
                    return self.noise_constant * np.sqrt(self.dt) * np.random.standard_normal(self.x.shape)
            return func

    def ricker_step(self):
        if self.noise_implementation == NOISE.RICKER_LINEAR:
            if self.noise_linear == 0:
                b = np.ones(self.x.shape)
            else:
                b = np.exp(self.noise_linear * np.sqrt(self.dt) * np.random.normal(0, 1, self.x.shape))
            self.x = b * self.x * np.exp(self.interaction_matrix.dot(
                self.x + np.linalg.inv(self.interaction_matrix).dot(self.growth_rate)) * self.dt)
        else:
            raise ValueError('No implementation for "%s"' % self.noise_implementation.name)

    def arato_step(self, i):
        if self.noise_implementation == NOISE.ARATO_LINEAR:
            self.xt += self.x * self.dt

            t = i * self.dt

            Y = self.growth_rate * t - self.noise_linear ** 2 / 2 * t + self.interaction_matrix.dot(self.xt) + \
                self.noise_linear * self.bm[:, i].reshape(self.x.shape)  # noise * np.random.normal(0, 1, initcond.shape)
            self.x = self.initial_condition * np.exp(Y)

    def write_abundances_to_file(self, i):
        with open(self.f, "a") as file:
            file.write("%.5E" % (i * self.dt))
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
        df = pd.DataFrame(self.x, columns=['endpoint'], index=['species_%d' % i for i in range(1, self.Nspecies+1)])
        return df

def test_timeseries():
    print('test Timeseries')

    N = 50

    params = {}

    steadystate = np.ones([N,1]) #np.logspace(-3, 2, N).reshape([N, 1])

    # no interaction
    omega = np.zeros([N, N]);
    np.fill_diagonal(omega, -1)

    params['interaction_matrix'] = omega

    # no immigration
    params['immigration_rate'] = np.zeros([N, 1])

    # different growth rates determined by the steady state
    params['growth_rate'] = - (omega).dot(steadystate)

    params['initial_condition'] = np.copy(steadystate) * np.random.normal(1, 0.1, steadystate.shape)

    params['noise_linear'] = 1e-1

    params['maximum_capacity'] = 5000

    ts = Timeseries(params, noise_implementation=NOISE.LANGEVIN_LINEAR, dt=0.01, tskip=4, T=100.0,
                    seed=int(time.time()), model=MODEL.MAX)

    print("sglv_timeseries")
    print(ts.timeseries.head())

    print("endpoint")
    print(ts.endpoint)

    return ts

def main():
    #test_validity_Jacobian()
    return test_timeseries()

if __name__ == "__main__":
    main()