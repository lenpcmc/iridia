# iridia
A Universal Model for Predicting IR Absorbance Spectra

## Quickstart

```
    import iridia

    ir = iridia("resources/cifs/{myfile}")
    ir.dyn = np.load("resources/arrays/{mystructure}.npy")   # Faster than computing hessian.

    ir.plot()
```


## Participation

```
    choose = lambda atoms: [ symbol for symbol in atoms.get_chemical_symbols() ]
    ir.plot(choose()

    from iridia.vibrations.coordNumber 
    choose = lambda atoms: [ symbol + str(coord) if symbol == 'B' else symbol for (symbol,coord) in zip(atoms.get_chemical_symbols(), coordNumber(atoms)) ]
    ir.plot(choose)
```
