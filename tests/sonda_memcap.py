"""MEM_CAP: il tetto del foglietto misurato.

Il foglietto evince le tracce piu' vecchie (FIFO) quando
supera MEM_CAP: per misurarlo serve un episodio con piu'
scritture del tetto. Qui N_FATTI marcatori con risposte
casuali riempiono una finestra di scrittura (il filler dopo e'
prevedibile, cosi' il contrasto e' alto e il fence traccia le
risposte); poi una finestra di richiamo ripresenta i marcatori
mescolati — la loss sulle risposte per eta' del fatto misura
cosa costa l'evizione: vecchi (scritti primi, evasi per primi)
contro recenti. MEM_CAP e' patchato a tempo di valutazione.

python murlen/tests/sonda_memcap.py [cuda|cpu]"""
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza,  # noqa: E402
                          PAD_ID)
from universi_oltre_finestra import (  # noqa: E402
    allena, sorpresa_simbolo)

CAMPO, PASSI, K = 512, 300, 2
MARK, N_ANS, N_FATTI = 300, 20, 60


def dati(n, seed, nfatti=N_FATTI):
    """n episodi: (scrittura, richiamo, etas).
    scrittura: nfatti marca+risposta casuale + filler '1'
    fino a CAMPO (le risposte imprevedibili vengono tracciate,
    il filler non compete per il fence).
    richiamo: i marcatori mescolati con le stesse risposte.
    etas[b][j] = indice del fatto alla j-esima marca del
    richiamo (0 = scritto per primo = evaso prima dal FIFO)."""
    r = random.Random(seed)
    scr, ric, etas = [], [], []
    for _ in range(n):
        ans = [r.randrange(2, 2 + N_ANS)
               for _ in range(nfatti)]
        w = []
        for i in range(nfatti):
            w += [MARK + i, ans[i], 1]
        w += [1] * (CAMPO - len(w))
        ordine = list(range(nfatti))
        r.shuffle(ordine)
        rc = []
        for i in ordine:
            rc += [MARK + i, ans[i], 1]
        scr.append(w)
        ric.append(rc)
        etas.append(ordine)
    return scr, ric, etas


def seqs_unite(scr, ric):
    return [a + b for a, b in zip(scr, ric)]


@torch.no_grad()
def richiamo(core, testa, ric, vocab, dev, z, mem0):
    """Loss per fatto sulle risposte del richiamo, dato lo
    stato (z, mem0) portato dalla finestra di scrittura."""
    ids2, pad2 = tensorizza(ric, vocab, CAMPO)
    X2, Y2 = ids2[:, :-1].to(dev), ids2[:, 1:].to(dev)
    z2, h2 = core(X2, pad2[:, :-1].to(dev), causal=True,
                  z0=z, mem0=mem0)
    ce = nn.functional.cross_entropy(
        testa(h2).flatten(0, 1), Y2.flatten(), reduction='none',
        ignore_index=PAD_ID).view(Y2.shape)
    return [float(ce[b, 3 * j].item())
            for j in range(N_FATTI)
            for b in range(ids2.size(0))]


@torch.no_grad()
def misura(core, testa, scr, ric, etas, vocab, dev, cap):
    """Sotto il cap dato: scorre la scrittura (annota a ogni
    finestra), poi il richiamo con le tracce rimaste — loss
    sulle risposte per terzo di eta'."""
    MM.MEM_CAP = cap
    core.eval()
    ids, pad = tensorizza(scr, vocab, CAMPO)
    X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
    z, h = core(X, pad[:, :-1].to(dev), causal=True)
    lo = testa(h)
    mem = core.annota(h, sorpresa_simbolo(lo, Y),
                      pad[:, :-1].to(dev))
    ntr = mem[0].size(0) if mem else 0
    ce_f = richiamo(core, testa, ric, vocab, dev, z, mem)
    errs = [[] for _ in range(3)]
    for j in range(N_FATTI):
        for b in range(len(ric)):
            terzo = min(2, 3 * etas[b][j] // N_FATTI)
            errs[terzo].append(ce_f[j * len(ric) + b])
    print('MEM_CAP=%-4d tracce=%-4d  vecchi %.3f  medi %.3f'
          '  recenti %.3f' % (
              cap, ntr,
              sum(errs[0]) / max(1, len(errs[0])),
              sum(errs[1]) / max(1, len(errs[1])),
              sum(errs[2]) / max(1, len(errs[2]))),
          flush=True)


@torch.no_grad()
def lettore(core, testa, scr, ric, vocab, dev, cap=256):
    """Il lettore esiste? Stessa scrittura, richiamo con
    foglietto intatto / svuotato / tracce di un altro
    episodio — se la loss non cambia, il modello non legge
    le tracce e il piatto-per-eta' non dice nulla sul cap."""
    MM.MEM_CAP = cap
    core.eval()
    ids, pad = tensorizza(scr, vocab, CAMPO)
    X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
    z, h = core(X, pad[:, :-1].to(dev), causal=True)
    mem = core.annota(h, sorpresa_simbolo(testa(h), Y),
                      pad[:, :-1].to(dev))
    # l'attention e' invariante all'ordine delle righe: la
    # corruzione giusta e' il contenuto — tracce dell'episodio
    # sbagliato (fatti diversi)
    B = len(mem)
    cond = {
        'intatto': mem,
        'svuotato': None,
        'episodio-1': [mem[(b + 1) % B] for b in range(B)],
    }
    for nome, m0 in cond.items():
        ce_f = richiamo(core, testa, ric, vocab, dev, z, m0)
        print('lettore %-10s loss risposte %.3f' % (
            nome, sum(ce_f) / len(ce_f)), flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    scr_t, ric_t, _ = dati(PASSI, 1)
    scr_v, ric_v, etas = dati(8, 2)
    vocab, n = vocab_da_sequenze(
        seqs_unite(scr_t, ric_t) + seqs_unite(scr_v, ric_v))
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                            enc_layers=4, m=16, nhead=4,
                            tempo=64)

    def W(seqs):
        out = []
        for a in range(0, CAMPO + 3 * N_FATTI, CAMPO):
            ids, pad = tensorizza(
                [s[a:a + CAMPO] for s in seqs], vocab, CAMPO)
            out.append((ids.to(dev), pad.to(dev)))
        return out

    hist, testa = allena(core, W(seqs_unite(scr_t, ric_t)),
                         vocab, dev, passi=PASSI)
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    for cap in (8, 32, 64, 256):
        misura(core, testa, scr_v, ric_v, etas, vocab, dev,
               cap)
    lettore(core, testa, scr_v, ric_v, vocab, dev)
