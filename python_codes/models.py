from enum import Enum


class MODEL(Enum):
    GLV = 1
    QSMI = 2 # quadratic species metabolite interaction
    MAX = 3 # glv with maximum total number of species
    MAX_IMMI = 4
    IBM = 5