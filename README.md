# ThreeLayeredSphereModel

This tool computes analytically the electrical potential in a three-layered
sphere (head) model due to two point electrodes at the boundary. It was used for
validating our tDCS simulation solver and is mentioned in the paper
**A flexible workflow for simulating transcranial electric stimulation in healthy and lesioned brains**
by *Benjamin Kalloch, Pierre-Louis Bazin, Arno Villringer, Bernhard Sehm, and Mario Hlawitschka*.

The theory behind the tool was introduced by Stanley Rush and Danial A. Driscoll
in their paper "Current distribution in the brain from surface electrodes." in
1968.

The tool was written in Python3 and offers a GUI using via PySide2. 
Within the GIU you may:
- change the conductivity values assigned to each of the three layers
- change the radius of each layer
- define the position of the electrodes (in spherical coordinates)
- visualize the result
