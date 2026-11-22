"""Errore di bordo su testo vero.

Come sonda_campo ma su libri Wikijunior a caratteri: i bordi di
campo tagliano parole, nomi, frasi — la struttura a lungo raggio
e' quella reale del linguaggio, non un compito costruito. Se
sotto testo vero il gap residuo resta piccolo, la risposta
biologica e' "saccade + memoria bastano" e ctx resta una
dichiarazione di nascita; se il gap e' grande, esiste il
segnale per allargare il campo.

python murlen/tests/sonda_campo_libro.py [cuda|cpu]"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import allena  # noqa: E402
from universi_libro import libri  # noqa: E402
from sonda_campo import misura  # noqa: E402
from murlen.ponte import tensorizza  # noqa: E402

CAMPO, PASSI, K = 128, 150, 3
L = 4 * CAMPO            # pezzi corti: i libri sono piccoli,
                         # 512 caratteri = 3 bordi


def finestre(seqs, vocab, dev):
    out = []
    for a in range(0, L, CAMPO):
        ids, pad = tensorizza([s[a:a + CAMPO] for s in seqs],
                              vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


def pezzi_liberi(testi, r, ntrain, ntest, lun=L):
    """ntrain + ntest pezzi di lun caratteri: train dai primi
    libri mischiati, test dai libri dopo — il testo di prova
    non e' mai stato visto."""
    lib = [t for lista in testi.values() for t in lista]
    r.shuffle(lib)
    out = []
    for t in lib:
        for a in range(0, len(t) - lun, lun):
            out.append([ord(c) for c in t[a:a + lun]])
            if len(out) >= ntrain + ntest:
                return out[:ntrain], out[ntrain:]
    return out[:ntrain], out[ntrain:]


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    r = random.Random(1)
    testi = libri(per_lingua=200)
    train, test = pezzi_liberi(testi, r, 1200, 64)
    print('%d pezzi train, %d test' % (len(train), len(test)))
    vocab, n = vocab_da_sequenze(train + test)
    print('vocabolario: %d simboli' % n)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                         enc_layers=4, m=16, nhead=4)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=PASSI)
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    misura(core, testa, finestre(test, vocab, dev), dev)
