"""Scala del lettore indirizzato: N=32 a 2000 passi.

sonda_scala_indirizzo: a N=8 e 2000 passi il canale
nasce (prima 0.41 vs 3.82/6.81). sonda_trasferimento:
a N=32 senza training su quei fatti gli embedding delle
query non esistono — il trasferimento non e' misurabile.
Domanda pulita: allenato direttamente a N=32, il canale
indirizzato si forma o satura? Se si forma -> motore
generale (la competenza vale su N fatti); se piatto ->
primo limite di capacita' misurato.

python murlen/tests/sonda_scala_n32.py [cuda|cpu]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
import torch  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import K, allena  # noqa: E402
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, finestre, lettore, GIUN, NVAL)

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = 2000
    print('=== N=32 d=384 passi=%d ===' % passi, flush=True)
    test = dati(64, 2, 32)
    t0 = dati(8, 9, 32)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K)
    train = dati(passi * 8, 1, 32)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    lettore(core, testa, test, vocab, dev)
