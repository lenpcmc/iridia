import iridia
from iridia.vibrations.vparts import *
from iridia.visualize.vspect import *
from iridia.pqeq.qeq import qeq

from rruffIR import *

rpref = "../resources/rruff/processed/data/"
rreff = [
        #"Andalusite__R050449-1__Infrared__Infrared_Data_Processed__391.txt",
        #"Danburite__R050602-1__Infrared__Infrared_Data_Processed__640.txt",
        #"Diopside__R040097-1__Infrared__Infrared_Data_Processed__253.txt",
        #"StilbiteCa__R050012-1__Infrared__Infrared_Data_Processed__369.txt",
        #"Wollastonite__R040008-1__Infrared__Infrared_Data_Processed__1001.txt",
        "../../../ref/stratosC-2.csv",
    ]
ppref = "../resources/relaxed/"
preff = [
        #"andalusite.cif",
        #"danburite.cif",
        #"diopside.cif",
        #"stilbiteCA.cif",
        #"wollastonite.cif",
        "Li2B4O7.cif",
        #"Li2B4O7-2.cif",
        "LiB.cif",
    ]

def main():
    for r,p in zip(rreff, preff):
        spectrum, rs = rruffIR(f"{rpref}/{r}")
        spectrum = spectrum[:850]
        w = spectrum * 0.03
        rs = rs[:850]
        rs = rs.reshape( rs.shape + (1,) )

        ir = iridia(f"{ppref}/{p}", relax = False)

        ir.dyn = np.load(f"../resources/arrays/{p[:-4]}-A.npy")
        #ir.ddm = np.load(f"../resources/arrays/{p[:-4]}-DipoleA.npy")
        #print(ir.get_charges())

        #charge = iridia.qeq(ir.atoms)

        choose = lambda atoms: [ f"{sym}{coord}" for sym,coord in zip(atoms.get_chemical_symbols(), coordNumber(atoms)) ]
        cats = np.unique(choose(ir.atoms))

        ps = ir.absorbance(w, 0.65, choose)
        ps = spectNorm(ps)
        #ps = ps / np.max(np.sum(ps, axis = -1))

        rs = rs - np.min(rs)
        rs = spectNorm(rs)
        #rs = rs / np.max(rs)
        #rs = rs + 0.5 * np.max(np.abs(ps - rs))
        
        fig,ax = spectrumPlot()
        #ax.plot(spectrum, ps, label = "Prediction")
        ax.stackplot(spectrum, ps.T, labels = cats)
        ax.plot(spectrum, rs, label = "Literature", color = "#b026bd")
        ax.legend()
        
        #fig.savefig(f"{p[0].upper()}{p[1:-4]}.png", dpi = 600)
        fig.savefig(f"{p[0].upper()}{p[1:-4]}-StackA.png", dpi = 600)

        plt.show()


main()
