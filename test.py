import math
import os
import random
import time
import sys

import jsbsim

sys.path.append(str(jsbsim.get_default_root_dir()) + '/FCM/')

from myModule.representation_learning.getRl import get_rl
from myModule.representation_learning import getStatus

fdm = jsbsim.FGFDMExec(None)
fdm.load_model('f16')



fdm['ic/vc-kts'] = 1000
fdm["ic/h-sl-ft"] = 30005.5

fdm.run_ic()

i = 0

while fdm.run():
    x = time.time()
    i = i + 1
    if i > 1000:
        break
    fdm.run_ic()
    fdm['ic/vc-kts'] = 1000
    print(1 / (time.time() - x))
