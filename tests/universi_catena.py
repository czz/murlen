"""Prova della catena: l'inferenza di una giunzione
alimenta la successiva — reasoning a piu' passi.
Come il dialogo, ma dopo ogni giunzione lo stato dei
modi cambia: (a,b) -> (b, (a+b) mod 4). Il codice della
giunzione i serve per calcolare la giunzione i+1:
sbagliarne una spezza le seguenti. Nella finestra 2 la
prima giunzione richiede lo stato CALCOLATO in
finestra 1 — non un fatto letto, un fatto prodotto.
Misura: valuta del dialogo (1a giunzione pulita, tutte
le ablazioni); il curriculum avvicina i modi al confine.
python murlen/tests/universi_catena.py [cuda|cpu] [passi]"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
import universi_dialogo as ud  # noqa: E402
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena)


def dati_catena(n, seed, L=2 * ud.WIN, pos_modi=0):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = r.randrange(ud.NMOD), r.randrange(ud.NMOD)
        s = [30] * pos_modi + [ud.MA + a, ud.MB + b]
        ca, cb = r.randrange(8), r.randrange(8)
        p = 0
        while len(s) < L:
            if p % ud.JUN_OGNI == ud.JUN_FASE:
                s.append(ud.GIUN + a * ud.NMOD + b)
                a, b = b, (a + b) % ud.NMOD   # la risposta
                    # diventa il nuovo stato: la giunzione
                    # dopo dipende da quella appena emessa
            else:
                s.append(ud.CNTA + ca % 8)
            ca += 1
            s.append(ud.CNTB + cb % 8)
            cb += 1
            p += 1
        out.append(s[:L])
    return out


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    test = dati_catena(64, 2)
    train0 = dati_catena(8, 9, pos_modi=200)
    vocab, n = vocab_da_sequenze(train0 + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    testa, hist = None, []
    for pos in (200, 150, 100, 50, 0):
        train = dati_catena(passi * 8, 1, pos_modi=pos)
        h2, testa = allena(core, finestre(train, vocab, dev),
                           vocab, dev, passi=passi, batch=8,
                           testa=testa)
        hist += h2
        print('fase pos_modi=%d finita' % pos, flush=True)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', 'universi_catena.pt'))
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    ud.valuta(core, testa, finestre(test, vocab, dev),
              dev, vocab)
