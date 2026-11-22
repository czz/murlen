"""Il modello impara a scegliere cosa ricordare?
Foglietto con gate di scrittura appreso (mem_gate=True)
contro la scrittura a sorpresa, sulla stessa sonda a
due fatti del dialogo (positivo noto: la scrittura a
sorpresa porta la 1a giunzione di W2 sotto il caso
2.773, ~2.6 a 750 passi di curriculum).
Falsificatore: se il gate non replica il positivo su
una sonda dove la memoria DEVE funzionare, lo
strumento e' calibrato male — si dichiara, non si
rattoppa. Se lo replica o lo batte, la scelta della
consolidazione e' apprendibile.
python murlen/tests/universi_gate.py [cuda|cpu] [passi] [seed]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_dialogo import dati_dialogo, valuta  # noqa: E402
from universi_oltre_finestra import K, allena  # noqa: E402
from universi_chiave import finestre  # noqa: E402

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    test = dati_dialogo(64, 2)
    train0 = dati_dialogo(8, 9 + seed, pos_modi=200)
    vocab, n = vocab_da_sequenze(train0 + test)
    torch.manual_seed(seed)
    core = MultiUniverse(n, K=K, mem_gate=True)
    testa, hist = None, []
    for pos in (200, 150, 100, 50, 0):
        train = dati_dialogo(passi * 8, 1 + seed, pos_modi=pos)
        h2, testa = allena(core, finestre(train, vocab, dev),
                           vocab, dev, passi=passi, batch=8,
                           testa=testa)
        hist += h2
        print('fase pos_modi=%d finita' % pos, flush=True)
    nome = ('universi_gate.pt' if seed == 0
            else 'universi_gate_s%d.pt' % seed)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', nome))
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    W = finestre(test, vocab, dev)
    ids1, p1 = W[0]
    core.eval()
    with torch.no_grad():
        _, h1 = core(ids1[:, :-1], p1[:, :-1], causal=True)
        g = torch.sigmoid(core.foglio.scrittura(h1))
    print('gate: media %.3f  lam foglio %.3f' %
          (g[~p1[:, :-1]].mean().item(),
           core.foglio.lam.item()))
    valuta(core, testa, finestre(test, vocab, dev), dev,
           vocab)
