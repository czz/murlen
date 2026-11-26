"""Il lettore trasferisce?

check_decode: con coppia+simbolo il canale indirizzato
nasce a N=32 (prima 0.030) e regge a N=128 solo con
cap+passi in piu'. Domanda decisiva per MIMO: la
competenza e' AGNOSTICA al fatto — un motore generale —
o si riallena a ogni scala?

Un solo training (N=32, coppia+simbolo), poi eval a
ladder N=32/64/128 sullo stesso modello: le marche di
eval sono simboli mai visti in training (blocchi
disgiunti) — se il canale e' agnostico basta la
consistenza delle chiavi, non l'embedding appresa.

python murlen/tests/sonda_trasferimento.py \
    [cuda|cpu] [passi] [seed<N>] [sim] [cap<N>]
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
                     blocchi, nfatti, maschere_extra=None):
    """Ablazioni sulle posizioni GIUN e sulla maschera
    'prima'; se maschere_extra e' dato (lista (riu,det)
    per sequenza) misura anche il riuso lontano."""
    core.eval()
    gj, qy = blocchi[3], blocchi[1]
    giun = torch.tensor([vocab[gj + i]
                         for i in range(NVAL)], device=dev)
    qry = torch.tensor([vocab[qy + i]
                        for i in range(nfatti)],
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
    passi = next((int(a) for a in sys.argv[2:]
                  if a.isdigit()), 2000)
    seed = next((int(a[4:]) for a in sys.argv
                 if a.startswith('seed')), 0)
    sim = 'sim' in sys.argv
    cap = next((int(a[3:]) for a in sys.argv
                if a.startswith('cap')), None)
    NTR = 32
    NEV = [32, 64, 128]
    # ANS/GIUN fissi fra i blocchi: il vocabolario delle
    # risposte e' condiviso. Per N>32 marche e query
    # vanno su id 0-299: mai emesse dai dati di training,
    # righe emb non allenate ma consistenti — l'eval
    # misura l'agnosticismo del canale, non simboli noti.
    # Bonus: n resta ~472 — stesso stream di init del
    # run da prima=0.030, il controllo replica per
    # costruzione.
    blocchi = {}
    for n_ in NEV:
        blocchi[n_] = (SP.MARK, SP.QRY, SP.ANS, SP.GIUN) \
            if n_ <= SP.NFATTI_MAX else \
            (0, n_, SP.ANS, SP.GIUN)
    t0 = dati(8, 9, NTR, blocchi[NTR])
    tests = {n_: dati(64, 2 + n_, n_, blocchi[n_])
             for n_ in NEV}
    tutti = [list(range(0, SP.GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(
        t0 + sum(tests.values(), []) + tutti)
    torch.manual_seed(seed)
    core = MM.MultiUniverse(n, K=K, coppia=True,
                            simbolo=sim)
    train = dati(passi * 8, 1, NTR, blocchi[NTR])
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8, seed=seed)
    print('train: %.3f -> %.3f   lam=%.3f   cap=%d' % (
        hist[0], sum(hist[-10:]) / 10,
        core.foglio.lam.item(), core._mem_cap), flush=True)

    for n_ in NEV:
        # la capienza e' una risorsa di retention, non
        # un peso appreso: si dimensiona per N in eval
        # (8x i fatti copre coppie+transizioni)
        core._mem_cap = cap or max(256, 8 * n_)
        print('=== eval N=%d (cap=%d) ===' % (
            n_, core._mem_cap), flush=True)
        lettore_generico(core, testa, tests[n_], vocab, dev,
                         blocchi[n_], n_)
