"""A quante letture per fatto nasce il lettore?

sonda_lettore_fattori ha isolato la variabile decisiva: uso
fitto -> lettura di contenuto sempre (anche senza
curriculum, anche con binding arbitrario); uso singolo ->
mai. Qui la soglia: quante volte un fatto deve essere letto
in un episodio perche' il canale query->traccia si formi.

Condizioni come i libri veri: SENZA curriculum (pos_modi=0,
il mondo non gradua il carico) e tabella arbitraria
(nome+valore casuali a episodio — i fatti veri non sono
copie). La copia resta come riferimento meccanico.

usi = giunzioni in finestra 2 (~jun_ogni coppie fra letture):
ogni giunzione e' una lettura allenante del canale.

python murlen/tests/sonda_lettore_densita.py [cuda|cpu] [passi]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import K, allena  # noqa: E402
import sonda_lettore_fattori as SF  # noqa: E402

PASSI_FASE = 100


def dati(n, seed, tabella, usi, L=2 * SF.WIN):
    """un_fatto senza curriculum: fatto in testa (pos 0),
    giunzioni distribuite lungo la sequenza ogni
    jun_ogni coppie -> ~usi letture in finestra 2.
    usi=1 -> giunzione singola a JUN_POS_UNA."""
    if usi == 1:
        return SF.dati(n, seed, 0, tabella, usa_una=True, L=L)
    jun_ogni = max(2, 128 // usi)
    out = []
    import random
    r = random.Random(seed)
    for _ in range(n):
        a = r.randrange(SF.NMOD)
        v = r.randrange(SF.NVAL)
        fatto = [SF.MA + a, SF.ANS + v] if tabella \
            else [SF.MA + a]
        s = list(fatto)
        ca, cb = r.randrange(8), r.randrange(8)
        p = 0
        while len(s) < L:
            if p % jun_ogni == SF.JUN_FASE:
                s.append(SF.GIUN + (v if tabella else a))
            else:
                s.append(SF.CNTA + ca % 8)
            ca += 1
            s.append(SF.CNTB + cb % 8)
            cb += 1
            p += 1
        out.append(s[:L])
    return out


COND = [
    # nome, tabella, usi (letture in finestra 2)
    ('tabella usi=1', True, 1),
    ('tabella usi=2', True, 2),
    ('tabella usi=4', True, 4),
    ('tabella usi=8', True, 8),
    ('tabella usi=16', True, 16),
    ('copia   usi=2', False, 2),
    ('copia   usi=4', False, 4),
    ('copia   usi=8', False, 8),
]

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 \
        else 5 * PASSI_FASE
    for nome, tabella, usi in COND:
        print('=== %s ===' % nome, flush=True)
        test = dati(64, 2, tabella, usi)
        t0 = dati(8, 9, tabella, usi)
        vocab, n = vocab_da_sequenze(t0 + test)
        torch.manual_seed(0)
        core = MM.MultiUniverse(n, K=K)
        train = dati(passi * 8, 1, tabella, usi)
        hist, testa = allena(
            core, SF.finestre(train, vocab, dev), vocab,
            dev, passi=passi, batch=8)
        print('train: %.3f -> %.3f' % (
            hist[0], sum(hist[-10:]) / 10), flush=True)
        SF.lettore(core, testa, test, vocab, dev, tabella)
        del core, testa
        if dev == 'cuda':
            torch.cuda.empty_cache()
