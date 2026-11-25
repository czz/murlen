"""Chiavi a coppia di simboli: il lettore a N=32 si accende?

La diagnosi del muro N=32 (sonda_densita_n32 + check_copertura
+ check_firme):
1. scrittura: il fence affama sotto novita' uniforme — solo
   2-5 fatti su 32 lasciano traccia;
2. geometria: le firme h dei fatti sono troppo deboli per
   indirizzare (4.9% vs caso 3.1%);
3. lettura: il canale non si accende mai (svuotato == intatto).

Modalita' `coppia=True` del foglietto: la posizione p lascia
traccia se la sua coppia di simboli (s_{p-1}, s_p) non e' gia'
coperta — la chiave e' il binding discreto, stabile e unico,
non la firma h. Atteso: ~64 tracce/episodio (32 mark + 32
prima-query per fatto), dentro MEM_CAP, copertura completa —
e il lettore ha un bersaglio trovabile da allenare.

Task identico a sonda_pressione_indirizzo / sonda_scala_n32:
N=32, L=1024 (4 finestre), stesse ablazioni del lettore.
Il muro noto: 'prima' ~3.0 == a caso.

python murlen/tests/sonda_coppia.py [cuda|cpu] [passi]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze, TestaPredizione  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, finestre, lettore, GIUN, NVAL)

NFATTI = 32

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    ancora = len(sys.argv) > 3 and sys.argv[3] == 'ancora'
    test = dati(64, 2, NFATTI)
    t0 = dati(8, 9, NFATTI)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K, coppia=True,
                            ancora=ancora)
    core.to(dev)
    testa = TestaPredizione(
        max(vocab.values()) + 1,
        core.enc.tok.embedding_dim).to(dev)
    # sanity: copertura della scrittura a coppia su un
    # episodio intero — atteso ~65 tracce (32 mark +
    # 32 prima-query + confine)
    W1 = finestre(dati(1, 7, NFATTI), vocab, dev)
    z, mem = None, None
    with torch.no_grad():
        for ids, pad in W1:
            X, Y = ids[:, :-1], ids[:, 1:]
            z, h = core(X, pad[:, :-1], causal=True,
                        z0=z, mem0=mem)
            mem = core.annota(
                h, sorpresa_simbolo(testa(h), Y),
                pad[:, :-1])
    print('copertura: tracce dopo un episodio = %s'
          % [m.size(0) for m in mem], flush=True)
    train = dati(passi * 8, 1, NFATTI)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    lettore(core, testa, test, vocab, dev)
