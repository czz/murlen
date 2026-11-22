"""Significativita' degli organi su testo vero:
delta PER SEQUENZA (paired), non media aggregata —
se il foglietto aiuta davvero sulla fetta 'sor',
il delta no_z - dimenticato deve essere negativo
in modo consistente sulle singole sequenze, non solo
in aggregato. Riporta media, errore standard e
quante sequenze concordano — il falsificatore chiede
di vedere il segnale, non la media di rumore.
python murlen/tests/universi_libro_stat.py [cuda|cpu]"""
import math
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (  # noqa: E402
    vocab_da_sequenze, TestaPredizione, PAD_ID)
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, sorpresa_simbolo)
import universi_libro_finestre as U  # noqa: E402


@torch.no_grad()
def per_seq(core, testa, W, dev):
    """Per ogni sequenza: CE per posizione in W2 per
    4 condizioni; maschere ric/sor/reale."""
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    B, T = Y2.shape
    reale = ~p2[:, 1:]

    _, h1 = core(X1, p1[:, :-1], causal=True)
    lo1 = testa(h1)
    V = lo1.size(-1)
    visti = torch.zeros(B, V, device=dev)
    visti.scatter_add_(1, ids1,
                       torch.ones_like(ids1, dtype=torch.float))
    s1 = sorpresa_simbolo(lo1, Y1)
    fin = torch.isfinite(s1)
    ss = torch.where(fin, s1, s1.new_zeros(()))
    q1 = ss.quantile(.25, 1, keepdim=True)
    q3 = ss.quantile(.75, 1, keepdim=True)
    sm = fin & (s1 > q3 + 1.5 * (q3 - q1)) & ~p1[:, :-1]
    sm[:, 0] = ~p1[:, 0]
    sorprese = torch.zeros(B, V, device=dev)
    sorprese.scatter_add_(1, X1, sm.float())
    sor = sorprese.gather(1, Y2) > 0
    ric = visti.gather(1, Y2) > 0

    mem1 = core.annota(h1, s1, p1[:, :-1])
    z1, _ = core(X1, p1[:, :-1], causal=True)

    def ce_con(z0, mem0, lam_comm=None, lam_mem=None):
        oc, om = core.comm.lam.item(), core.foglio.lam.item()
        if lam_comm is not None:
            core.comm.lam.fill_(lam_comm)
        if lam_mem is not None:
            core.foglio.lam.fill_(lam_mem)
        _, h2 = core(X2, p2[:, :-1], causal=True, z0=z0,
                     mem0=mem0)
        ce = nn.functional.cross_entropy(
            testa(h2).flatten(0, 1), Y2.flatten(),
            reduction='none', ignore_index=PAD_ID).view(B, T)
        core.comm.lam.fill_(oc)
        core.foglio.lam.fill_(om)
        return ce

    out = {
        'portati': ce_con(z1, mem1),
        'dimenticato': ce_con(None, None),
        'no_z': ce_con(None, mem1),
        'no_comm': ce_con(z1, mem1, lam_comm=0.),
        'no_mem': ce_con(z1, mem1, lam_mem=0.),
    }
    return out, {'ric': ric & reale, 'sor': sor & reale,
                 'tot': reale}


def stat(deltas):
    n = len(deltas)
    if n < 2:
        return float('nan'), float('nan'), 0
    m = sum(deltas) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in deltas) / (n - 1))
    neg = sum(1 for d in deltas if d < 0)
    return m, sd / math.sqrt(n), neg


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    r = random.Random(1)
    testi = U.libri()
    train_t, test_t = {}, {}
    for lg, lista in testi.items():
        nt = max(1, len(lista) // 10)
        test_t[lg], train_t[lg] = lista[:nt], lista[nt:]
    train = [s for _, s in U.coppie(train_t, r, per_testo=64)]
    test = U.coppie(test_t, r, per_testo=4)
    per_lingua = {lg: [s for l2, s in test if l2 == lg]
                  for lg in U.LIBRI}
    vocab, n = vocab_da_sequenze(train +
                                 [s for _, s in test])
    core = MultiUniverse(n, K=K)
    ck = torch.load(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'dati',
        'universi_libro_finestre.pt'),
        map_location=dev, weights_only=False)
    core.load_state_dict(ck['core'])
    core.to(dev).eval()
    testa = TestaPredizione(n, core.enc.tok.embedding_dim)
    testa.load_state_dict(ck['testa'])
    testa.to(dev).eval()
    print('lam foglio %.4f comm %.4f' %
          (core.foglio.lam.item(), core.comm.lam.item()))

    BLOCCO = 8
    for lg in U.LIBRI:
        seqs = per_lingua[lg]
        # delta per sequenza: condizione - riferimento
        D = {(c, f): [] for c in
             ('portati', 'no_z', 'no_mem', 'no_comm')
             for f in ('sor', 'ric', 'tot')}
        for a in range(0, len(seqs), BLOCCO):
            W = finestre(seqs[a:a + BLOCCO], vocab, dev)
            ces, masch = per_seq(core, testa, W, dev)
            base = ces['dimenticato']
            for b in range(len(seqs[a:a + BLOCCO])):
                for f, mk in masch.items():
                    mb = mk[b]
                    if not mb.any():
                        continue
                    for c in ('portati', 'no_z', 'no_mem',
                              'no_comm'):
                        D[(c, f)].append(
                            (ces[c][b][mb] - base[b][mb])
                            .mean().item())
        print('=== %s (%d sequenze) — delta vs '
              'dimenticato ===' % (lg, len(seqs)))
        for f in ('sor', 'ric', 'tot'):
            for c in ('portati', 'no_z', 'no_mem',
                      'no_comm'):
                m, se, neg = stat(D[(c, f)])
                nn_ = len(D[(c, f)])
                print('  %-8s su %-3s  %+7.4f +- %.4f  '
                      '(%d/%d migliori)' %
                      (c, f, m, se, neg, nn_))
