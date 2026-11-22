"""Scala: era taglia/tempo o e' limite strutturale?

sonda_pressione_indirizzo (d=384, 500 passi): sulle
prime occorrenze per finestra il recall indirizzato
resta AL CASO in ogni ablazione — il compito si
risolve solo per induzione in-window. Qui il
compito e' IDENTICO (N=8, flusso di sole coppie
query->risposta) ma:

- A: d=384, passi 2000  — asse del TEMPO
- B: d=768, passi 1000  — asse della TAGLIA (+tempo)

Se il canale indirizzato nasce (intatto << caso su
'prima', svuotato peggio) in una delle due -> era
scala: mente piu' grande o piu' vissuta. Se piatta
in entrambe -> limite strutturale del foglietto
dichiarato a parita' di hardware.

python murlen/tests/sonda_scala_indirizzo.py [cuda|cpu]
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

COND = [
    ('A d=384  passi=2000', dict(d=384), 2000),
    ('B d=768  passi=1000', dict(d=768), 1000),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    for nome, kw, passi in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, 8)
        t0 = dati(8, 9, 8)
        tutti = [list(range(0, GIUN + NVAL + 2))]
        vocab, n = vocab_da_sequenze(t0 + test + tutti)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K, **kw)
        train = dati(passi * 8, 1, 8)
        hist, testa = allena(
            core, finestre(train, vocab, dev), vocab, dev,
            passi=passi, batch=8)
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        lettore(core, testa, test, vocab, dev)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
