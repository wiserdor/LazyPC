import os
import time
from GUI import GUI
from LazyNN import LazyNN

lnn = LazyNN()

g = GUI(lnn).start()
while not os.path.isfile('./Models/lazy_mod.h5py'):
    time.sleep(2)

from ClientPC import ClientPC

s = ClientPC()
s.connect()

from LazyModel import LazyModel

m = LazyModel(server=s, gui=g, lann=lnn)
m.start()
