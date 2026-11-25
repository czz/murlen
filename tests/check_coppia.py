"""Dove si rompe la catena coppia? (diagnostica post-sonda)

sonda_coppia: copertura completa (130 tracce/episodio),
chiavi univoche — eppure 'prima' resta a caso e
svuotato == intatto. La catena lettura ha tre anelli
ancora da verificare:

1. lam: il gate del canale si e' aperto in training?
2. indirizzamento: alle posizioni 'prima' (prima query
   del fatto nella finestra) qpair(h) trova la traccia
   giusta? argmax del punteggio sulle chiavi del buffer,
   decodificato a simbolo: deve essere MARK+q o QRY+q.
3. decodifica: se (2) funziona ma la loss e' piatta, il
   valore recuperato non e' usabile dalla testa.

python murlen/tests/check_coppia.py [cuda|cpu] [passi]
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import vocab_da_sequenze  # noqa: E402
from universi_oltre_finestra import K, allena  # noqa: E402
from sonda_pressione_indirizzo import (  # noqa: E402
    dati, finestre, lettore, MARK, QRY, GIUN, NVAL,
    NFATTI_MAX)

NFATTI = 32


@torch.no_grad()
def diagnosi(core, testa, test, vocab, dev):
    """Alle posizioni 'prima' delle finestre >=1: argmax
    qpair(h) @ chiavi -> la traccia vincente ha per
    prima meta' MARK+q o QRY+q? (hit-rate vs caso)."""
    core.eval()
    W = finestre(test, vocab, dev)
    emb = core.enc.tok.weight
    qry_ids = torch.tensor(
        [vocab[QRY + i] for i in range(NFATTI_MAX)],
        device=dev)
    mark_ids = torch.tensor(
        [vocab[MARK + i] for i in range(NFATTI_MAX)],
        device=dev)
    z, mem, stati = None, None, []
    for ids, pad in W:
        stati.append((z, mem))
        z, h = core(ids[:, :-1].to(dev),
                    pad[:, :-1].to(dev), causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, None, pad[:, :-1].to(dev))
    d = h.size(2)
    tot, hit, hitm, hitq = 0, 0, 0, 0
    for wi, (ids, pad) in enumerate(W):
        if wi == 0:
            continue
        X = ids[:, :-1].to(dev)
        z_in, mem_in = stati[wi]
        _, h = core(X, pad[:, :-1].to(dev), causal=True,
                    z0=z_in, mem0=mem_in)
        if getattr(core, '_ancora', False):
            q = torch.cat(
                [core.enc.tok(X), core.foglio.qpair(h)],
                -1)
        else:
            q = core.foglio.qpair(h)
        for b in range(X.size(0)):
            if mem_in is None or not mem_in[b].size(0):
                continue
            chiavi = mem_in[b][:, :2 * d]
            visti = set()
            for i in range(X.size(1)):
                trov = (X[b, i] == qry_ids).nonzero()
                if not len(trov):
                    continue
                fq = trov[0].item()
                if fq in visti:
                    continue
                visti.add(fq)
                win = (q[b, i] @ chiavi.T).argmax()
                sim = (emb - chiavi[win, :d]).norm(
                    dim=1).argmin().item()
                tot += 1
                if sim == mark_ids[fq].item():
                    hit += 1
                    hitm += 1
                elif sim == qry_ids[fq].item():
                    hit += 1
                    hitq += 1
    print('lam = %.4f' % core.foglio.lam.item())
    print('indirizzamento prima: %.3f  (n=%d, mark %d,'
          ' query %d — a caso ~%.3f)'
          % (hit / max(tot, 1), tot, hitm, hitq,
             2. / 130), flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    ancora = len(sys.argv) > 3 and sys.argv[3] == 'ancora'
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    test = dati(64, 2, NFATTI)
    t0 = dati(8, 9, NFATTI)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(t0 + test + tutti)
    torch.manual_seed(seed)
    core = MM.MultiUniverse(n, K=K, coppia=True,
                            ancora=ancora)
    train = dati(passi * 8, 1, NFATTI)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    diagnosi(core, testa, test, vocab, dev)
    lettore(core, testa, test, vocab, dev)
