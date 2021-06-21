from matplotlib.colors import Normalize, SymLogNorm
import numpy as np

class PiecewiseNormalize(Normalize):
    def __init__(self, xvalues, cvalues):
        self.xvalues = xvalues
        self.cvalues = cvalues

        Normalize.__init__(self)

        self.vmin = np.min(xvalues)
        self.vmax = np.max(xvalues)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            x, y = self.xvalues, self.cvalues
            ma = np.interp(value, x, y)
            ma = np.ma.masked_where(np.isnan(ma), ma)
            return ma
        else:
            vmin = self.vmin
            vmax = self.vmax
            r = Normalize.__call__(self, value, clip)
            if vmin != self.vmin or vmax != self.vmax:
                print("vmin / vmax has changed!")
            return r

class PiecewiseLogNorm(SymLogNorm):
    def __init__(self, xvalues, cvalues, linthresh=1e-2):
        self.xvalues = np.array(xvalues, dtype=float)
        self.cvalues = np.array(cvalues, dtype=float)

        self._lower = np.nanmin(xvalues)
        self._upper = np.nanmax(xvalues)

        self.symlog = False
        if np.sign(self._lower) != np.sign(self._upper):
            self.symlog = True

        if self.symlog:
            self.xvalues_pos = self.xvalues[self.xvalues>=0]
            self.xvalues_pos[self.xvalues_pos == 0] = linthresh
            self.xvalues_neg = self.xvalues[self.xvalues<=0]
            self.xvalues_neg[self.xvalues_neg == 0] = -linthresh

            self.cvalues_pos = self.cvalues[self.xvalues >= 0]
            self.cvalues_neg = self.cvalues[self.xvalues <= 0]
        SymLogNorm.__init__(self, linthresh, linscale=0.01, vmin=self._lower, vmax=self._upper)

        #self.vmin = np.nanmin(xvalues)
        #self.vmax = np.nanmax(xvalues)

    def interpolate_1(self, value):
        if not self.symlog:
            return np.interp(np.sign(value) * np.log10(np.abs(value)), np.sign(self.xvalues) * np.log10(np.abs(self.xvalues)), self.cvalues)
        else:
            if value >= 0:
                return np.interp(np.log10(value), np.log10(self.xvalues_pos), self.cvalues_pos)
            else:
                return np.interp(np.log10(-value), np.log10(-1 * self.xvalues_neg[::-1]), self.cvalues_neg[::-1])

    def interpolate(self, value):
        return np.vectorize(self.interpolate_1, otypes=[float])(value)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            ma = self.interpolate(value)
            ma = np.ma.masked_where(np.isnan(ma), ma)
            return ma
        else:
            vmin = self.vmin
            vmax = self.vmax
            r = SymLogNorm.__call__(self, value, clip)
            if vmin != self.vmin or vmax != self.vmax:
                print("vmin / vmax has changed!")
            return r

class PiecewiseSymLogNorm(SymLogNorm):
    def __init__(self, xvalues, cvalues):
        self.xvalues = xvalues
        self.cvalues = cvalues

        self.linthresh = 1e-10

        SymLogNorm.__init__(self, self.linthresh)

        self.vmin = np.min(xvalues)
        self.vmax = np.max(xvalues)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            x, y = self.xvalues, self.cvalues
            if np.sign(np.nanmin(x)) != np.sign(np.nanmax(x)):
                ma = np.interp(np.sign(value)*np.log10(np.abs(value)),
                                                np.sign(x)*np.log10(np.abs(x)), y)
                ma = np.ma.masked_where(np.isnan(ma), ma)
            return ma
        else:
            return SymLogNorm.__call__(self, value, clip)