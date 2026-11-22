"""Il lettore trasferisce?

sonda_scala_indirizzo: a 2000 passi il canale indirizzato
nasce sul compito N=8 (prima 0.41 vs 3.82/6.81). Domanda
decisiva per MIMO: la competenza e' AGNOSTICA al fatto —
un motore generale — o legata agli 8 fatti del training?

Un solo training (N=8 denso, 2000 passi), poi due eval
sullo stesso modello:

1. N=8  — controllo: deve replicare il positivo
2. N=32 — stesso formato, 24 fatti mai visti: il canale
   generalizza senza riallenare?

python murlen/tests/sonda_trasferimento.py [cuda|cpu]
"""
import glob
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import PAD_ID  # noqa: E402
from murlen.ponte import vocab_da_sequenze, tensorizza  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, finestre, lettore, GIUN, NVAL, WIN)
import sonda_pressione_indirizzo as SP  # noqa: E402


@torch.no_grad()
def lettore_generico(core, testa, test, vocab, dev,
                     maschere_extra=None):
    """Ablazioni sulle posizioni GIUN e sulla maschera
    'prima'; se maschere_extra e' dato (lista (riu,det)
    per sequenza) misura anche il riuso lontano."""
    core.eval()
    giun = torch.tensor([vocab[GIUN + i]
                         for i in range(NVAL)], device=dev)
    qry = torch.tensor([vocab[SP.QRY + i]
                        for i in range(SP.NFATTI_MAX)],
                       device=dev)
    W = finestre(test, vocab, dev)
    z, mem, stati = None, None, []
    for ids, pad in W:
        X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
        stati.append((z, mem))
        z, h = core(X, pad[:, :-1].to(dev), causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, sorpresa_simbolo(testa(h), Y),
                          pad[:, :-1].to(dev))

    def maschere(X, Y):
        sel = torch.isin(Y, giun)
        prima = torch.zeros_like(sel)
        B, T = X.shape
        for b in range(B):
            visti = torch.zeros(len(qry), dtype=torch.bool,
                              device=dev)
            for i in range(T):
                hit = (X[b, i] == qry).nonzero()
                if len(hit):
                    q = hit[0].item()
                    if sel[b, i] and not visti[q]:
                        prima[b, i] = True
                    visti[q] = True
        return sel, prima

    for nome in ('intatto', 'svuotato', 'episodio-1',
                 'senza-z'):
        tot = {'giunzioni': 0., 'prima': 0., 'lontano': 0.}
        cnt = {k: 0 for k in tot}
        for wi, (ids, pad) in enumerate(W):
            if wi == 0:
                continue
            X, Y = ids[:, :-1].to(dev), ids[:, 1:].to(dev)
            z_in, mem_in = stati[wi]
            if nome == 'senza-z':
                z_in = None
            elif nome == 'svuotato':
                mem_in = None
            elif nome == 'episodio-1':
                mem_in = [mem_in[(b + 1) % len(mem_in)]
                          for b in range(len(mem_in))]
            _, h = core(X, pad[:, :-1].to(dev), causal=True,
                        z0=z_in, mem0=mem_in)
            lo = testa(h)
            ce = nn.functional.cross_entropy(
                lo.flatten(0, 1), Y.flatten(),
                reduction='none',
                ignore_index=PAD_ID).view(Y.shape)
            sel_all, sel_prima = maschere(X, Y)
            for et, sel in (('giunzioni', sel_all),
                            ('prima', sel_prima)):
                if sel.any():
                    tot[et] += (ce[sel].sum().item())
                    cnt[et] += int(sel.sum().item())
            if maschere_extra is not None:
                a = wi * WIN
                for b in range(ids.size(0)):
                    for j in range(Y.size(1)):
                        q = a + j + 1
                        if q < len(maschere_extra[b][0]) \
                                and maschere_extra[b][0][q] \
                                and maschere_extra[b][1][q]:
                            tot['lontano'] += ce[b, j].item()
                            cnt['lontano'] += 1
        caso = torch.log(torch.tensor(float(NVAL))).item()
        print('lettore %-10s giunzioni %.3f  prima %.3f'
              '%s  (n=%d/%d%s, a caso %.3f)' % (
                  nome,
                  tot['giunzioni'] / max(cnt['giunzioni'], 1),
                  tot['prima'] / max(cnt['prima'], 1),
                  ('  lontano %.3f' % (
                      tot['lontano'] / max(cnt['lontano'], 1))
                   if maschere_extra is not None else ''),
                  cnt['giunzioni'], cnt['prima'],
                  ('/%d' % cnt['lontano']
                   if maschere_extra is not None else ''),
                  caso), flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = 2000
    # training: stesso compito N=8 denso del positivo
    t0 = dati(8, 9, 8)
    test8 = dati(64, 2, 8)
    test32 = dati(64, 3, 32)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test8 + test32 + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K)
    train = dati(passi * 8, 1, 8)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)

    print('=== eval N=8 (controllo) ===', flush=True)
    lettore_generico(core, testa, test8, vocab, dev)

    print('=== eval N=32 (fatti mai visti) ===', flush=True)
    lettore_generico(core, testa, test32, vocab, dev)
