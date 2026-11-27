"""Gli universi servono quando la finestra non basta?
Sequenze da 2 finestre (2*CTX simboli): l'argomento e' scritto
solo nel primo simbolo e il taglio puo' spezzare una parola sul
bordo. Nella seconda finestra l'encoder non vede niente di tutto
questo: solo z, lo stato degli universi, puo' portarlo.
Si allena K=3 passando z da una finestra all'altra (gradiente
attraverso il bordo: cosi' impara a metterci l'argomento), poi si
misura la loss della seconda finestra con z portato, dimenticato
o mescolato fra le sequenze, zittendo comunicazione e universi.
I primi simboli della finestra sono dove la memoria vale di piu':
prima che l'encoder raccolga indizi locali.
python murlen/tests/universi_oltre_finestra.py [cuda|cpu]"""
import copy
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, tensorizza,  # noqa: E402
                          TestaPredizione, PAD_ID,
                          BatchAdattivo, _vram,
                          _azzera_picco, _picco, _svuota,
                          ATTESA_MAX)

CTX, PASSI, K, BATCH = 1024, 200, 3, 1
PRIMI = 64          # primi simboli della finestra: il bordo


def dati(n, seed, L=2 * CTX):
    """Come confronto_init.dati ma lunga L: un argomento scelto
    all'inizio decide quali parole compaiono; il taglio a L puo'
    spezzare l'ultima parola a meta'."""
    r = random.Random(0)
    parole = [[r.randrange(2, 42) for _ in range(r.randint(3, 6))]
              for _ in range(60)]
    r = random.Random(seed)
    out = []
    for _ in range(n):
        arg = r.randrange(3)
        s = [100 + arg]
        while len(s) < L:
            s += parole[arg * 20 + r.randrange(20)] + [1]
        out.append(s[:L])
    return out


def finestre(seqs, vocab, dev):
    """Ogni sequenza -> finestre di CTX simboli [(ids,pad), ...]."""
    L = len(seqs[0])
    out = []
    for a in range(0, L, CTX):
        ids, pad = tensorizza([s[a:a + CTX] for s in seqs], vocab,
                              CTX)
        out.append((ids.to(dev), pad.to(dev)))
    return out


def _ce(logits, y):
    return nn.functional.cross_entropy(
        logits.flatten(0, 1), y.flatten(), ignore_index=PAD_ID)


def sorpresa_simbolo(lo, Y):
    """(B,N): quanto ha sorpreso il simbolo in ogni
    posizione di input = la loss con cui e' stato
    predetto dalla posizione prima. Posizione 0 = inf:
    il confine d'episodio e' sempre tracciato."""
    B, N = Y.shape
    s = torch.full((B, 1), float('inf'), device=lo.device)
    ce = nn.functional.cross_entropy(
        lo[:, :-1].flatten(0, 1), Y[:, :-1].flatten(),
        reduction='none', ignore_index=PAD_ID).view(B, -1)
    return torch.cat([s, ce], 1)


def passo(core, testa, opt, W, sel, dev):
    """Un passo sulle 2 finestre in sequenza: z esce dalla prima
    ed entra nella seconda, il gradiente attraversa il bordo.
    Con lui il foglietto (traccia episodica)."""
    z, mem, loss = None, None, 0.
    for ids, pad in W:
        X, Y = ids[sel, :-1], ids[sel, 1:]
        z, h = core(X, pad[sel, :-1], causal=True, z0=z,
                    mem0=mem)
        lo = testa(h)
        loss = loss + _ce(lo, Y)
        # consolida: le posizioni il cui simbolo era
        # imprevisto lasciano traccia per le finestre dopo
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


def allena(core, W, vocab, dev, passi=PASSI, batch=BATCH,
           seed=0, testa=None, lam_aperto=0):
    """lam_aperto>0: nei primi N passi il gate del foglietto
    non puo' chiudere sotto 1 — il canale ha un periodo di
    prova prima di essere giudicato (a N alto il retrieval
    immaturo spingeva lam sotto 1: stato assorbente)."""
    core.to(dev)
    if testa is None:
        testa = TestaPredizione(
            max(vocab.values()) + 1,
            core.enc.tok.embedding_dim)
    testa.to(dev)
    opt = torch.optim.AdamW(
        list(core.parameters()) + list(testa.parameters()),
        lr=3e-4)
    w = min(200, passi // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1., (i + 1) / max(1, w)))
    B = W[0][0].size(0)
    win = W[0][0].size(1)
    gen, ordine = torch.Generator(
    ).manual_seed(seed), []
    ad = BatchAdattivo(batch, win)
    hist = []
    for i in range(passi):
        libera = _vram(dev)
        nb = ad.scegli(libera)
        attesa = 0
        while True:
            nb = min(nb, B)
            if len(ordine) < nb:
                ordine = torch.randperm(
                    B, generator=gen).tolist()
            sel = torch.tensor(ordine[:nb])
            base = _azzera_picco(dev)
            try:
                hist.append(passo(core, testa, opt, W, sel, dev))
                ordine = ordine[nb:]
                break
            except RuntimeError as e:
                # VRAM piena o frammentata (OOM o
                # CUBLAS ALLOC_FAILED): si libera e
                # si ripete il passo con meno
                # sequenze — come impara_predizione
                if not ('ALLOC' in str(e) or
                        'out of memory' in str(e).lower()):
                    raise
            opt.zero_grad()
            _svuota(dev)
            dopo = _vram(dev)
            if nb == 1:
                if attesa >= ATTESA_MAX:
                    raise MemoryError(
                        'VRAM insufficiente anche per '
                        'una sequenza da %d s' % attesa)
                time.sleep(1)
                attesa += 1
            else:
                if dopo is None or libera is None \
                        or dopo >= libera:
                    ad.rifiuto()
                nb = min(ad.scegli(dopo), nb // 2)
            libera = dopo
        if i < lam_aperto and core.foglio is not None:
            with torch.no_grad():
                core.foglio.lam.clamp_(min=1.)
        sched.step()
        picco = _picco(dev, base)
        if picco is not None:
            ad.registra(nb, win, picco)
        if (i + 1) % 25 == 0:
            print('passo %d/%d loss %.3f (batch %d) lam=%.3f'
                  % (i + 1, passi, sum(hist[-25:]) / 25, nb,
                     core.foglio.lam.item()
                     if core.foglio is not None else -1),
                  flush=True)
    return hist, testa


@torch.no_grad()
def valuta(core, testa, W, dev):
    """Loss per finestra, poi sulla seconda: z portato,
    dimenticato, mescolato fra le sequenze; e mutando
    comunicazione e voce degli universi.
    -> lista di (nome, loss, loss primi PRIMI simboli)."""
    core.eval()
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]

    def seconda(m, z0, mem0):
        _, h2 = m(X2, p2[:, :-1], causal=True, z0=z0,
                  mem0=mem0)
        lo = testa(h2)
        return (_ce(lo, Y2).item(),
                _ce(lo[:, :PRIMI], Y2[:, :PRIMI]).item())

    def prova(nome, f=None, modo='porta'):
        m = copy.deepcopy(core).to(dev)
        if f:
            with torch.no_grad():
                f(m)
        m.eval()
        z1, h1 = m(X1, p1[:, :-1], causal=True)
        mem1 = m.annota(h1, sorpresa_simbolo(testa(h1), Y1),
                        p1[:, :-1])
        if modo == 'dimentica':
            z1, mem1 = None, None
        elif modo == 'mescola':
            z1 = z1.roll(1, 0)
            if mem1:
                mem1 = mem1[1:] + mem1[:1]
        elif modo == 'no_z':
            z1 = None
        elif modo == 'no_mem':
            mem1 = None
        l2, l2p = seconda(m, z1, mem1)
        print('%-42s %5.3f  %5.3f' % (nome, l2, l2p),
              flush=True)
        del m
        if dev == 'cuda':
            torch.cuda.empty_cache()

    _, h1 = core(X1, p1[:, :-1], causal=True)
    print('finestra 1 (riferimento): loss %.3f'
          % _ce(testa(h1), Y1).item())
    z, _ = core(X1, p1[:, :-1], causal=True)
    v = nn.functional.normalize(z.flatten(2), dim=-1)
    sim = (v @ v.transpose(1, 2))[:, *torch.triu_indices(
        K, K, 1)].mean().item()
    print('somiglianza fra universi: %.3f' % sim)
    print('%-42s %5s  %s' % ('finestra 2', 'loss',
                             'primi %d' % PRIMI))
    prova('z portato dalla finestra 1')
    prova('tutto dimenticato', modo='dimentica')
    prova('solo z dimenticato', modo='no_z')
    prova('solo foglietto dimenticato', modo='no_mem')
    prova('z+foglietto mescolati', modo='mescola')
    prova('senza comunicazione (lam=0)',
          lambda m: m.comm.lam.zero_())
    prova('universi muti (zinj=0)',
          lambda m: m.zinj.weight.zero_())
    d = core.zinj.out_features
    for k in range(core.K):
        prova('universo %d muto' % k,
              lambda m, k=k: m.zinj.weight[
                  :, k * d:(k + 1) * d].zero_())


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    train = dati(PASSI * BATCH, 1)
    test = dati(16, 2)
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev)
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    valuta(core, testa, finestre(test, vocab, dev), dev)
