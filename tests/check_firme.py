"""Firme per-fatto in h: allenate o assenti?

L'ipotesi dello stallo: per leggere la traccia del
fatto f, la h alla posizione QRY+f deve essere
VICINA alla firma di quel fatto (h scritta quando
il fatto e' apparso) piu' che alle altre 31. Se
questa geometria esiste, l'indirizzamento e'
apprendibile; se non si forma mai nemmeno dopo il
training, al substrato manca la proprieta'.

Misura: allena sul task N=32 standard (L=1024),
poi in eval calcola la sim-coseno fra h alle
posizioni QRY+f e le firme dei 32 mark (h alla
posizione ANS+f della prima finestra).
'indirizzabilita' = frazione di query il cui
argmax e' il proprio fatto. Confronto col modello
vergine (a caso ~1/32).

python murlen/tests/check_firme.py [cuda|cpu] [passi]
"""
import os
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from sonda_pressione_indirizzo import (  # noqa: E402
    GIUN, NVAL, MARK, QRY, ANS,
    NFATTI_MAX as NFATTI, WIN, dati, finestre)


@torch.no_grad()
def indirizzabilita(core, testa, test, vocab, dev,
                    blocco=16):
    """per ogni posizione QRY+f: argmax-cos fra le
    firme ANS+g -> conta se g == f. Firme = h della
    finestra 0 alle posizioni ANS (2f+1)."""
    core.eval()
    ok, tot = 0, 0
    sim_pr, sim_al = [], []
    for c in range(0, len(test), blocco):
        W = finestre(test[c:c + blocco], vocab, dev)
        z, mem, firme = None, None, None
        for wi, (ids, pad) in enumerate(W):
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            mem = core.annota(h, sorpresa_simbolo(
                testa(h), Y), pad[:, :-1])
            hn = Fn.normalize(h, dim=-1)
            if wi == 0:
                firme = hn[:, [2 * f + 1
                               for f in range(NFATTI)]]
                continue
            for b in range(ids.size(0)):
                for p in range(ids.size(1) - 1):
                    g = wi * WIN + p
                    if g < 2 * NFATTI:
                        continue
                    j = (g - 2 * NFATTI) // 2
                    if (g - 2 * NFATTI) % 2:
                        continue  # solo posizioni QRY
                    f = j % NFATTI
                    sim = (firme[b] * hn[b, p]).sum(-1)
                    a = sim.argmax().item()
                    tot += 1
                    ok += (a == f)
                    sim_pr.append(sim[f].item())
                    sim_al.append(
                        sim[[i for i in range(NFATTI)
                             if i != f]].mean().item())
    caso = 1. / NFATTI
    print('indirizzabilita: %.3f (a caso %.3f)   '
          'sim propria %.3f vs altrui %.3f  '
          '(n=%d)' % (ok / max(tot, 1), caso,
                      sum(sim_pr) / len(sim_pr),
                      sum(sim_al) / len(sim_al), tot),
          flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 \
        else 800
    test = dati(64, 2, NFATTI)
    t0 = dati(8, 9, NFATTI)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K)
    core.to(dev)
    testa0 = None
    print('--- modello vergine ---', flush=True)
    # testa finta per la sorpresa: serve annota
    import universi_oltre_finestra as U
    testa0 = U.TestaPredizione(
        max(vocab.values()) + 1,
        core.enc.tok.embedding_dim).to(dev)
    indirizzabilita(core, testa0, test, vocab, dev)
    print('--- alleno %d passi ---' % passi, flush=True)
    train = dati(passi * 8, 1, NFATTI)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    indirizzabilita(core, testa, test, vocab, dev)
