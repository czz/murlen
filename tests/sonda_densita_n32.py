"""N=32 a densita' piena: ~60 letture per fatto.

sonda_scala_n32: a N=32 il lettore resta piatto con
~15 letture/fatto per episodio — l'ipotesi e' che il
vincolo sia la densita' per indirizzo (a N=8 servono
~63). Qui stesso compito, stessi fatti, ma le query
arrivano ~60 volte per fatto: L=3904, 16 finestre.

Se il lettore nasce -> confermato: il muro e' la
densita' per indirizzo, e il replay nel sonno deve
diventare re-interrogazione densa (query nuove sugli
stessi fatti), non ri-lettura verbatim.
Se resta piatto -> il muro non e' solo densita':
interferenza fra indirizzi a N grande.

Nota memoria: un episodio da 16 finestre tiene ~1GB
di attivazioni a finestra -> passo con gradient
checkpoint (matematica identica, attivazioni
ricalcolate nel backward). dati vettoriali numpy:
niente liste da GB.

Verificato (check_cap_foglietto): le tracce della
finestra 0 sopravvivono a tutte le 16 — MEM_CAP=256
non espelle i mark.

python murlen/tests/sonda_densita_n32.py [cuda|cpu] [letture]
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import murlen.model as MM  # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          tensorizza)
import universi_oltre_finestra as U  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from sonda_pressione_indirizzo import (  # noqa: E402
    GIUN, NVAL, MARK, QRY, ANS,
    NFATTI_MAX, WIN)

NFATTI, LETTURE = 32, 60


def dati(n, seed, nfatti, letture):
    """Come sonda_pressione_indirizzo.dati ma vettoriale
    e con `letture` query per fatto. -> ndarray (n,L)."""
    r = np.random.RandomState(seed)
    vals = r.randint(0, NVAL, (n, nfatti))
    nm = 2 * nfatti
    s = np.empty((n, nm + letture * nfatti * 2),
                 dtype=np.int64)
    s[:, :nm:2] = MARK + np.arange(nfatti)
    s[:, 1:nm:2] = ANS + vals
    qi = np.tile(np.arange(nfatti), letture)
    s[:, nm::2] = QRY + qi
    s[:, nm + 1::2] = GIUN + vals[:, qi]
    return s


def finestre(seqs, vocab, dev):
    """seqs ndarray (B,L) -> finestre da WIN simboli."""
    out = []
    for a in range(0, seqs.shape[1], WIN):
        ids, pad = tensorizza(
            seqs[:, a:a + WIN].tolist(), vocab, WIN)
        out.append((ids.to(dev), pad.to(dev)))
    return out


# --- passo con gradient checkpoint ---
# il grafo di una finestra pesa ~1GB; ricalcolarlo in
# backward e' identico in matematica (RNG preservato)

def _passo_ck(core, testa, opt, W, sel, dev):
    z, mem, loss = None, None, 0.
    for ids, pad in W:
        X, Y = ids[sel, :-1], ids[sel, 1:]
        z, h = checkpoint(
            lambda x, p, zi, mi: core(
                x, p, causal=True, z0=zi, mem0=mi),
            X, pad[sel, :-1], z, mem,
            use_reentrant=False)
        lo = testa(h)
        loss = loss + U._ce(lo, Y)
        mem = core.annota(h, sorpresa_simbolo(lo, Y),
                          pad[sel, :-1])
    opt.zero_grad()
    loss.backward()
    ng = nn.utils.clip_grad_norm_(
        list(core.parameters()) +
        list(testa.parameters()), 1.)
    if torch.isfinite(ng):
        opt.step()
    else:
        opt.zero_grad()
    return loss.item()


U.passo = _passo_ck


@torch.no_grad()
def lettore(core, testa, test, vocab, dev):
    """CE 'prima' occorrenza per ablazione — copia di
    sonda_pressione_indirizzo.lettore con finestre a
    lunghezza propria."""
    core.eval()
    giun = torch.tensor([vocab[GIUN + i]
                         for i in range(NVAL)], device=dev)
    qry = torch.tensor([vocab[QRY + i]
                        for i in range(NFATTI_MAX)],
                       device=dev)
    W = finestre(test, vocab, dev)
    z, mem, stati = None, None, []
    for ids, pad in W:
        X, Y = ids[:, :-1], ids[:, 1:]
        stati.append((z, mem))
        z, h = core(X, pad[:, :-1], causal=True,
                    z0=z, mem0=mem)
        mem = core.annota(h, sorpresa_simbolo(
            testa(h), Y), pad[:, :-1])

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
        tot = {'giunzioni': 0., 'prima': 0.}
        cnt = {'giunzioni': 0, 'prima': 0}
        for wi, (ids, pad) in enumerate(W):
            if wi == 0:
                continue
            X, Y = ids[:, :-1], ids[:, 1:]
            z_in, mem_in = stati[wi]
            if nome == 'senza-z':
                z_in = None
            elif nome == 'svuotato':
                mem_in = None
            elif nome == 'episodio-1':
                mem_in = [mem_in[(b + 1) % len(mem_in)]
                          for b in range(len(mem_in))]
            _, h = core(X, pad[:, :-1], causal=True,
                        z0=z_in, mem0=mem_in)
            lo = testa(h)
            sel_all, sel_prima = maschere(X, Y)
            for et, sel in (('giunzioni', sel_all),
                            ('prima', sel_prima)):
                if sel.any():
                    tot[et] += nn.functional.cross_entropy(
                        lo[sel], Y[sel]).item() \
                        * sel.sum().item()
                    cnt[et] += sel.sum().item()
        caso = torch.log(torch.tensor(float(NVAL))).item()
        print('lettore %-10s giunzioni %.3f  prima %.3f'
              '  (n=%d/%d, a caso %.3f)' % (
                  nome,
                  tot['giunzioni'] / max(cnt['giunzioni'], 1),
                  tot['prima'] / max(cnt['prima'], 1),
                  cnt['giunzioni'], cnt['prima'], caso),
              flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    letture = int(sys.argv[2]) if len(sys.argv) > 2 \
        else LETTURE
    passi = 2000
    print('=== N=%d letture/fatto=%d passi=%d ==='
          % (NFATTI, letture, passi), flush=True)
    test = dati(64, 2, NFATTI, letture)
    t0 = dati(8, 9, NFATTI, letture)
    tutti = [list(range(0, GIUN + NVAL + 2))]
    vocab, n = vocab_da_sequenze(
        t0.tolist() + test.tolist() + tutti)
    torch.manual_seed(0)
    core = MM.MultiUniverse(n, K=K)
    train = dati(passi * 8, 1, NFATTI, letture)
    W = finestre(train, vocab, dev)
    del train
    hist, testa = allena(
        core, W, vocab, dev, passi=passi, batch=8)
    print('train: %.3f -> %.3f' % (
        hist[0], sum(hist[-10:]) / 10), flush=True)
    lettore(core, testa, test, vocab, dev)
