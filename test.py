import iridia

ir = iridia.ir.iridia("../pythonIR/resources/cifs/wollastonite.cif")
print(ir.atoms)
print(len(ir.atoms))
ir.plot()
