"""Il campo visivo congenito: esiste un errore di bordo che la
memoria non compensa?

Domanda di Fase 0 (prima di qualsiasi meccanismo di crescita di
ctx): quando una sequenza supera il campo ctx e il contesto viene
troncato, le prime posizioni della finestra dopo perdono il
contesto locale. z e foglietto — che passano comunque — bastano a
compensare? Se si', allargare il campo e' inutile; se resta un
gap persistente, esiste il segnale da cui la crescita potrebbe
emergere.

Compito: l'argomento e' un solo simbolo all'inizio della
sequenza (finestra 0); le parole che seguono dipendono da lui.
Con ctx=CAMPO e L=4*CAMPO ci sono 3 bordi: in ogni finestra
dopo la prima il contesto necessario e' fuori campo. Si allena
portando z e foglietto fra le finestre (gradiente attraverso il
bordo, come nella vita), poi si misura la sorpresa per
posizione nelle finestre 1..3 con lo stato portato, dimezzato
(solo z / solo foglietto) o azzerato.

Stampa la curva media per posizione e il verdetto: il gap fra
le prime posizioni (bordo) e le ultime (interno) sotto stato
pieno. python murlen/tests/sonda_campo.py [cuda|cpu]"""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza,  # noqa: E402
                          TestaPredizione, PAD_ID)
from universi_oltre_finestra import (  # noqa: E402
    dati, allena, sorpresa_simbolo, _ce)

CAMPO, PASSI, K, BATCH = 256, 150, 3, 1
L = 4 * CAMPO                       # 3 bordi di campo
GRANA = 32                          # posizioni per tacca


def finestre(seqs, vocab, dev):
    """Ogni sequenza -> finestre di CAMPO simboli."""
    out = []
    for a in range(0, L, CAMPO):
        ids, pad = tensorizza([s[a:a + CAMPO] for s in seqs],
                              vocab, CAMPO)
        out.append((ids.to(dev), pad.to(dev)))
    return out


def ce_pos(lo, Y):
    """(B,N) loss per posizione."""
    return nn.functional.cross_entropy(
        lo.flatten(0, 1), Y.flatten(), reduction='none',
        ignore_index=PAD_ID).view(Y.shape)


@torch.no_grad()
def misura(core, testa, W, dev):
    """Curva della sorpresa per posizione, media sulle finestre
    oltre il primo bordo, per 4 condizioni di stato portato."""
    core.eval()
    X = [w[0][:, :-1] for w in W]
    Y = [w[0][:, 1:] for w in W]
    P = [w[1][:, :-1] for w in W]
    N = Y[0].size(1)              # larghezza dai dati: la
                                  # sonda vale per ogni campo
    conds = [('porta', True, True),
             ('no foglietto', True, False),
             ('no z', False, True),
             ('amnesia', False, False)]
    curve = {n: torch.zeros(N) for n, _, _ in conds}
    base = torch.zeros(N)
    z, mem, nbordo = None, None, 0
    for k, (x, y, p) in enumerate(zip(X, Y, P)):
        if k == 0:
            z, h = core(x, p, causal=True)
            base += ce_pos(testa(h), y).mean(0).cpu()
            mem = core.annota(
                h, sorpresa_simbolo(testa(h), y), p)
            continue
        nbordo += 1
        for nome, uz, um in conds:
            _, h = core(x, p, causal=True,
                        z0=z if uz else None,
                        mem0=mem if um else None)
            curve[nome] += ce_pos(testa(h), y).mean(0).cpu()
        z, h = core(x, p, causal=True, z0=z, mem0=mem)
        mem = core.annota(
            h, sorpresa_simbolo(testa(h), y), p)
    for n in curve:
        curve[n] /= nbordo
    base /= 1

    print('sorpresa media per posizione '
          '(finestre oltre il bordo):')
    print('%-10s' % 'pos' + ''.join('%-14s' % n
                                    for n, _, _ in conds))
    for a in range(0, N, GRANA):
        t = '%d-%d' % (a + 1, min(a + GRANA, N))
        r = ['%-14.3f' % curve[n][a:a + GRANA].mean()
             for n, _, _ in conds]
        print('%-10s' % t + ''.join(r))
    print('finestra 0 (nessun bordo): %.3f' % base.mean())

    bordo = GRANA
    interno = slice(max(N // 2, N - 4 * GRANA), N)
    print()
    print('%-14s %8s %8s %8s' % ('condizione', 'bordo',
                                 'interno', 'gap'))
    for nome, _, _ in conds:
        b = curve[nome][:bordo].mean().item()
        i = curve[nome][interno].mean().item()
        print('%-14s %8.3f %8.3f %8.3f' % (nome, b, i, b - i))
    p = curve['porta'][:bordo].mean().item()
    a = curve['amnesia'][:bordo].mean().item()
    print()
    print('compenso dello stato al bordo: %.3f '
          '(amnesia %.3f - porta %.3f)' % (a - p, a, p))
    print('gap residuo sotto stato pieno: %.3f — se ~0 la '
          'memoria basta, il campo non va allargato; se >0 '
          'e persistente esiste il segnale di crescita'
          % (p - curve['porta'][interno].mean().item()))


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    train = dati(PASSI * BATCH, 1, L=L)
    test = dati(16, 2, L=L)
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K, d=192, ctx=CAMPO,
                         enc_layers=4, m=16, nhead=4)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=PASSI)
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    misura(core, testa, finestre(test, vocab, dev), dev)
