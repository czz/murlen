"""Sonda di specializzazione: gli universi hanno RUOLI?
Non basta che zittirne uno costi: serve che costi su
PARTI diverse del compito. Nel dialogo la struttura e'
a due partiti (modo A/modo B, contatore A/contatore B):
se un universo tiene un partito, zittirlo deve far male
solo sulle giunzioni/posizioni di quel partito — danno
uniforme = ridondanza, danno selettivo = ruolo.
Misura: per ogni universo k, loss in finestra 2
zittendo solo lui, scomposta per simbolo-target
(giunzioni per valore di a/b, contatori A, contatori B).
python murlen/tests/universi_ruoli.py [ckpt] [cuda|cpu]"""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import TestaPredizione  # noqa: E402
import universi_dialogo as ud  # noqa: E402
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    sorpresa_simbolo, K)

DATI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'dati')


@torch.no_grad()
def perdite_per_target(core, testa, W, dev, vocab):
    """Loss della finestra 2 scomposta per classe di
    target: giunzioni divise per il modo che richiedono,
    contatore A, contatore B, resto."""
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    # i due modi di ogni sequenza: sono i primi simboli
    inv = {v: s for s, v in vocab.items()}
    raw2 = ids2.clone()
    out = {}

    def misura(m):
        z1, h1 = m(X1, p1[:, :-1], causal=True)
        mem1 = m.annota(h1, sorpresa_simbolo(testa(h1), Y1),
                        p1[:, :-1])
        _, h2 = m(X2, p2[:, :-1], causal=True, z0=z1,
                  mem0=mem1)
        lo = testa(h2)
        ce = nn.functional.cross_entropy(
            lo.flatten(0, 1), Y2.flatten(),
            reduction='none').view_as(Y2)
        return ce

    # classi di target: ogni codice giunzione presente
    # (selettivita' per risposta), contatori A e B —
    # universo muto che fa male solo su certe classi = ruolo
    def classi():
        y = Y2.cpu()
        cls = {}
        presenti = sorted(
            c for c in range(ud.NMOD * ud.NMOD)
            if ud.GIUN + c in vocab)
        for c in presenti:
            cls['g%d' % c] = [y == vocab[ud.GIUN + c]]
        cls['giun'] = [torch.isin(y, torch.tensor(
            [vocab[ud.GIUN + c] for c in presenti]))]
        cls['cntA'] = [torch.isin(
            y, torch.tensor([vocab[ud.CNTA + t]
                             for t in range(8)
                             if ud.CNTA + t in vocab]))]
        cls['cntB'] = [torch.isin(
            y, torch.tensor([vocab[ud.CNTB + t]
                             for t in range(8)
                             if ud.CNTB + t in vocab]))]
        return {k: torch.cat(v, 0).to(dev) for k, v in
                cls.items()}

    ce0 = misura(core)
    cls = classi()
    d = core.zinj.out_features
    for k in [-1] + list(range(core.K)):
        if k >= 0:
            m = copy.deepcopy(core).to(dev)
            with torch.no_grad():
                m.zinj.weight[:, k * d:(k + 1) * d].zero_()
            ce = misura(m)
            del m
        else:
            ce = ce0
        nome = 'normale' if k < 0 else 'universo %d muto' % k
        riga = {c: float(ce[s].mean()) for c, s in
                cls.items()}
        out[nome] = riga
    return out


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(DATI, 'universi_dialogo_1f.pt')
    dev = sys.argv[2] if len(sys.argv) > 2 else 'cuda'
    ck = torch.load(ckpt, map_location=dev,
                    weights_only=False)
    vocab = ck['vocab']
    core = MultiUniverse(max(vocab.values()) + 1, K=K)
    core.load_state_dict(ck['core'])
    testa = TestaPredizione(max(vocab.values()) + 1,
                            core.enc.tok.embedding_dim)
    testa.load_state_dict(ck['testa'])
    un_fatto = ud.MB not in vocab    # ckpt a fatto singolo
    test = ud.dati_dialogo(64, 2, un_fatto=un_fatto)
    r = perdite_per_target(core, testa,
                           finestre(test, vocab, dev),
                           dev, vocab)
    for nome, riga in r.items():
        print('%-16s' % nome, '  '.join(
            '%s %.3f' % (c, v) for c, v in riga.items()))
