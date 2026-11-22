"""Rigenera tests/dati/riferimento_ponte.json dal codice
attuale. Da eseguire SOLO sul codice di riferimento
(prima delle modifiche): python -m murlen.tests.riferimento
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_ponte import _storico_riferimento, DATI  # noqa: E402

if __name__ == '__main__':
    with open(os.path.join(DATI, 'riferimento_ponte.json'), 'w') as f:
        json.dump({'tempo0': _storico_riferimento(0)}, f, indent=1)
    print('scritto')
