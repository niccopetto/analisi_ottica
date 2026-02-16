
import sys
import os
import numpy as np

# Adjust path so we can import from the current directory
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    import AuxiliaryTools
    print("AuxiliaryTools imported successfully.")
    # Test class instantiation
    ac = AuxiliaryTools.AuxiliaryClass()
    ic = AuxiliaryTools.ImageClass()
    print("Classes instantiated successfully.")
except Exception as e:
    print(f"Failed to import/use AuxiliaryTools: {e}")
    import traceback
    traceback.print_exc()

try:
    import gs_engine
    print("gs_engine imported successfully.")
    # Test class instantiation (requires args, so we just check import for now, or use dummy)
    # GS = gs_engine.GerSaxPhaseRetriever(AuxiliaryTools.ImageClass(), distances=[1], wavelength=633e-9, ordering=[0])
    # The above would fail because ImageClass is empty.
except Exception as e:
    print(f"Failed to import gs_engine: {e}")
    import traceback
    traceback.print_exc()

try:
    import analisi_ottica
    print("analisi_ottica imported successfully.")
except Exception as e:
    print(f"Failed to import analisi_ottica: {e}")

print("Verification complete.")
