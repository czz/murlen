"""Gli universi si parlano su dati veri?
Corpus: libri Wikijunior in TRE lingue (it/en/fr) — la
sotto-struttura esiste davvero: se gli universi si
spartissero il lavoro, zittirne uno farebbe male solo
sulle lingue che gli competono; se cooperassero,
zittire la comm costerebbe. La verifica e' per lingua.
python murlen/tests/universi_libro.py [cuda|cpu] [passi]"""
import copy
import glob
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze, impara_predizione,  # noqa: E402
                          tensorizza, _flusso, PAD_ID)

LIBRI = {'it': '/home/czz/ncm-lab/libri/it',
         'en': '/home/czz/ncm-lab/libri/en',
         'fr': '/home/czz/ncm-lab/libri/fr'}
L, PASSI = 512, 1500
K = 3


def libri(per_lingua=120, seed=0):
    """per_lingua file presi a caso per lingua."""
    r = random.Random(seed)
    testi = {}
    for lg, cartella in LIBRI.items():
        files = sorted(glob.glob(
            os.path.join(cartella, '*.txt')))
        r.shuffle(files)
        testi[lg] = []
        for f in files:
            try:
                t = open(f, encoding='utf-8',
                         errors='ignore').read()
            except OSError:
                continue
            if len(t) < L * 2:
                continue
            testi[lg].append(t)
            if len(testi[lg]) >= per_lingua:
                break
    return testi


def pezzi(testi, r):
    """(lingua, sequenza) per ogni pezzo di L caratteri."""
    out = []
    for lg, lista in testi.items():
        for t in lista:
            for a in range(0, len(t) - L, L):
                out.append((lg, [ord(c) for c in t[a:a + L]]))
    r.shuffle(out)
    return out


@torch.no_grad()
def loss_su(core, testa, seqs, vocab, dev, blocco=16):
    """Loss media, a blocchi per non riempire la VRAM."""
    core.eval()
    tot, n = 0., 0
    for a in range(0, len(seqs), blocco):
        ids, pad = tensorizza(seqs[a:a + blocco], vocab, L)
        ids, pad = ids.to(dev), pad.to(dev)
        X, Y = ids[:, :-1], ids[:, 1:]
        tot += nn.functional.cross_entropy(
            testa(_flusso(core, X, pad[:, :-1])
                  ).flatten(0, 1),
            Y.flatten(), ignore_index=PAD_ID,
            reduction='sum').item()
        n += int((Y != PAD_ID).sum())
    return tot / n


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else PASSI
    r = random.Random(1)
    testi = libri()
    train_t, test_t = {}, {}
    for lg, lista in testi.items():
        nt = max(1, len(lista) // 10)
        test_t[lg], train_t[lg] = lista[:nt], lista[nt:]
    train = [s for _, s in pezzi(train_t, r)]
    test = pezzi(test_t, r)
    per_lingua = {lg: [s for l2, s in test if l2 == lg]
                  for lg in LIBRI}
    print('%d sequenze train, %d test'
          % (len(train), len(test)))
    vocab, n = vocab_da_sequenze(train +
                                 [s for _, s in test])
    print('vocabolario: %d simboli' % n)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    _, testa = impara_predizione(core, train, vocab,
                                 passi=passi, device=dev,
                                 batch=16, diversita=0.5)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', 'universi_libro.pt'))

    print('forza comunicazione lam = %.3f (nasce 0.1)'
          % core.comm.lam.item())
    print('voce di ogni universo (norma zinj):',
          [round(x, 3) for x in core.zinj.weight.view(
              core.zinj.out_features, K, -1
          ).norm(dim=(0, 2)).tolist()])

    # chi ascolta chi, per lingua
    for lg in LIBRI:
        ids, pad = tensorizza(per_lingua[lg][:8], vocab, L)
        with torch.no_grad():
            z, _ = core(ids.to(dev), pad.to(dev))
            _, w = core.comm(z)
        w = w.mean(0).cpu()
        print('chi ascolta chi [%s]:' % lg)
        for k in range(K):
            print('  u%d' % k,
                  [round(float(x), 3) for x in w[k]])

    ab = [('normale', None),
          ('senza comm (lam=0)',
           lambda m: m.comm.lam.zero_()),
          ('universi muti (zinj=0)',
           lambda m: m.zinj.weight.zero_()),
          ('lettura muta (zread=0)',
           lambda m: m.zread.out_proj.weight.zero_())]
    d = core.zinj.out_features
    for k in range(K):
        ab.append(('universo %d muto' % k,
                   (lambda k: lambda m: m.zinj.weight[
                       :, k * d:(k + 1) * d].zero_())(k)))

    test_tutte = [s for _, s in test]
    print('%-28s %7s  %5s  %5s  %5s'
          % ('ablazione', 'tutte', 'it', 'en', 'fr'))
    for nome, f in ab:
        m = copy.deepcopy(core)
        if f:
            with torch.no_grad():
                f(m)
        riga = [loss_su(m, testa, test_tutte, vocab, dev)]
        riga += [loss_su(m, testa, per_lingua[lg],
                         vocab, dev) for lg in LIBRI]
        print('%-28s %7.3f  %5.3f  %5.3f  %5.3f'
              % ((nome,) + tuple(riga)), flush=True)
        del m
