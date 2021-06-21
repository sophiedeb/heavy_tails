import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlotTimeseries():
    def __init__(self, ts, ax=None, species=None, species_rank=None, raw=False):
        if ax == None:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax = ax

        self.ts = ts

        if species == None:
            self.selection = self.select_species(species_rank)
        else:
            self.selection = species

        self.plot_timeseries(raw, species_rank)
        self.ax.set_yscale('log')

        #self.ax.legend()

    def select_species(self, ranks):
        self.mean = self.ts.mean()
        self.mean.drop('time', inplace=True)

        # don't select species that are 0 everywhere
        self.mean = self.mean[ self.mean > 0]

        self.Nspecies = len(self.ts.columns) - 1

        sorted_species = self.mean.sort_values().index.to_numpy()[::-1]

        sorted_species = sorted_species[ self.mean.loc[sorted_species] > 0 ]

        if ranks == None:
            return sorted_species[::max(1, int(len(sorted_species) / 4))]
        else:
            return sorted_species[[i-1 for i in ranks]] # python starts counting at 0, rank at 1

    def plot_timeseries(self, raw, species_rank):
        skip = max(1, int(len(self.ts) / 500))

        if not isinstance(species_rank, np.ndarray) and species_rank == None:
            for s in self.selection:
                self.ax.plot(self.ts['time'][::skip], self.ts[s][::skip], label=s)
        else:
            for s, rank in zip(self.selection, species_rank):
                self.ax.plot(self.ts['time'][::skip], self.ts[s][::skip], label='Rank %d' % rank)

        if not raw:
            self.ax.set_ylabel('Abundance')

class PlotRankAbundance():
    COLORS = ['b', 'r', 'g', 'k', 'orange', 'purple']

    def __init__(self, endpoints, ax=None, raw=False, labels=None, relative=True, **kwargs):
        if ax == None:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax = ax

        if isinstance(endpoints, list) and (labels == None or len(labels) != len(endpoints)):
            labels = [None] * len(endpoints)
            add_legend = False
        else:
            add_legend = True

        self.endpoints = endpoints

        if isinstance(endpoints, list):
            for x, label in zip(endpoints, labels):
                self.plot_rank_abundance(x, label, relative, **kwargs)
        else:
            self.plot_rank_abundance(endpoints, labels, relative, **kwargs)

        self.set_layout(raw, add_legend)

    def set_layout(self, raw, legend):
        if raw == False:
            self.ax.set_xlabel('Rank')
            self.ax.set_ylabel('Relative abundance')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        if legend:
            self.ax.legend()

    def plot_rank_abundance(self, data, label, relative=True, **kwargs):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            x = data.values.flatten()
        elif isinstance(data, np.ndarray):
            x = data.flatten()

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'grey' if label is None else None
        if 'alpha' not in kwargs.keys():
            kwargs['alpha'] =  0.5

        y = np.sort(x/np.sum(x))[::-1] if relative else np.sort(x)[::-1]

        self.ax.plot(np.arange(1, len(x)+1), y, label=label, **kwargs)

    def identify_rank_species(self, species):
        for i, ep in enumerate(self.endpoints):
            rank = ep.rank(ascending=False)

            for c, s in zip(self.COLORS, species):
                x = rank.loc[s]
                y = ep.loc[s] / np.sum(ep.values)
                plt.scatter(x, y, marker='*', color=c, label=s if i == 0 else None)
        self.ax.legend()