import os
import sys

# rende importabile il pacchetto `murlen` (la cartella sopra)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
