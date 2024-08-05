import numpy as np
import os

import iridia

for file in os.listdir("../pythonIR/resources/cifs"):
    ir = iridia(f"../pythonIR/resources/cifs/{file}", numAtoms = 1000)
    ir.get_dynamical()
    np.save("../pythonIR/resources/arrays/{file[:-4]}-A.npy", ir.dyn)

ir.plot()
