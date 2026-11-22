"""La saturazione strutturale vera, sul compito giusto.

sonda_saturazione (richiami) e sonda_sviluppo (d96 vs
d384) non trovarono plateau: i compiti erano troppo
facili — i corpi si eguagliavano col tempo. Ora il
compito lo abbiamo: recall indirizzato a N=8 ad alta
pressione, dove d=384 legge a contenuto dopo ~2000
passi (sonda_scala_indirizzo: prima 0.41 vs 3.82/6.81).

Domanda: un corpo piccolo SATURA dove quello grande
legge? Sweep d in {96, 192}, stesso compito e passi.
Se prima resta al caso mentre d384 leggeva -> plateau
strutturale vero -> sblocca la vigilanza-stallo su
un carico dove il segnale esiste.

Riferimento noto: d=384 -> prima intatto 0.41.

python murlen/tests/sonda_saturazione_vera.py [cuda|cpu]
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
    ('d=96',  dict(d=96,  nhead=4, enc_layers=4, m=16)),
    ('d=192', dict(d=192, nhead=4, enc_layers=4, m=16)),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = 2000
    for nome, kw in COND:
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
