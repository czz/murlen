"""La fase di sviluppo: esiste il segnale?

Due misure.

E1 — crescita giovanile sull'asse K (l'unico asse dove il
meccanismo esiste): una mente nata con K=1 con la vigilanza
accesa (soglia derivata dalla mediana della sua stessa loss di
warmup, non scelta) contro una nata K=3. La domanda: crescere
durante lo sviluppo raggiunge la nascita grande? Quando scatta
cresci()? Con che curva?

E2 — saturazione strutturale (asse d, dove il meccanismo non
esiste): d=96 vs d=384 sullo stesso compito, a tratti di
convergenza — il gap residuo al plateau e' quanto varrebbe una
crescita strutturale; se il gap chiude, la nascita piccola e'
solo piu' lenta e la crescita non serve.

python murlen/tests/sonda_sviluppo.py [cuda|cpu]"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          impara_predizione)
from confronto_init import dati  # noqa: E402

PASSI, WARM, BLOCCO = 300, 40, 20


def config(K, d=192):
    return dict(K=K, d=d, ctx=256, enc_layers=4, m=16,
                nhead=4, tempo=64)


def a_tratti(core, testa, seqs, vocab, dev, passi, **kw):
    """Allena a blocchi di BLOCCO passi e registra la curva:
    [(passi_fatti, loss_recente, K)]."""
    curva, hist = [], []
    fatti = 0
    while fatti < passi:
        h, testa = impara_predizione(
            core, seqs, vocab, passi=BLOCCO, device=dev,
            batch=1, testa=testa, **kw)
        hist += h
        fatti += BLOCCO
        curva.append((fatti, sum(h[-BLOCCO:]) / BLOCCO,
                      core.K))
    return curva, hist, testa


def e1(seqs, vocab, n, dev):
    print('=== E1: crescita giovanile su K ===')
    torch.manual_seed(0)
    piccolo = MultiUniverse(n, **config(K=1))
    h0, testa = impara_predizione(
        piccolo, seqs, vocab, passi=WARM, device=dev,
        batch=1)
    vig = statistics.median(h0[-WARM // 2:])
    print('soglia derivata dal warmup: %.3f (mediana '
          'ultimi %d passi)' % (vig, WARM // 2))
    curva_p, hist_p, testa = a_tratti(
        piccolo, testa, seqs, vocab, dev, PASSI,
        vigilanza=vig, pazienza=20)
    torch.manual_seed(0)
    grande = MultiUniverse(n, **config(K=3))
    curva_g, hist_g, _ = a_tratti(
        grande, None, seqs, vocab, dev, PASSI)
    print('%-8s %-22s %s' % ('passi', 'nato K=1 (K)', 'nato K=3'))
    for (p, l1, k1), (_, l3, _) in zip(curva_p, curva_g):
        print('%-8d %-22s %s' % (
            p + WARM, '%.3f (%d)' % (l1, k1),
            '%.3f' % l3))
    print('esiti: K piccolo %d -> %d; loss finale %.3f vs '
          '%.3f' % (1, piccolo.K, hist_p[-1], hist_g[-1]))


def e2(seqs, vocab, n, dev):
    print('=== E2: saturazione strutturale su d ===')
    ris = {}
    for d in (96, 384):
        torch.manual_seed(0)
        core = MultiUniverse(n, **config(K=2, d=d))
        curva, hist, _ = a_tratti(
            core, None, seqs, vocab, dev, PASSI)
        ris[d] = curva
        print('d=%-4d' % d + '  '.join(
            '%.2f' % l for _, l, _ in curva))
    g96 = ris[96][-1][1]
    g384 = ris[384][-1][1]
    print('gap al plateau (d96 - d384): %.3f — '
          'se resta aperto e\' il segnale per la crescita '
          'strutturale; se ~0 la nascita piccola basta'
          % (g96 - g384))


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    seqs = dati(600, 1)
    vocab, n = vocab_da_sequenze(seqs)
    e1(seqs, vocab, n, dev)
    e2(seqs, vocab, n, dev)
