import pandas as pd
import numpy as np
from metrica_IO import *
from traitement import *

DATADIR = "/home/UR/kontaous/Bureau/Projet Stage/Data/HAC"
gameid = 1
Home, Away, Events = read_match_data(DATADIR, gameid)
data = merge_tracking_data(Home, Away)

print(Events)


