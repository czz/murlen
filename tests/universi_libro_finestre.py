"""Gli organi servono su testo VERO?
Finestre consecutive dello stesso libro (2*WIN caratteri):
la seconda non vede la prima — i nomi, le parole rare,
il soggetto del discorso sono fuori contesto e possono
arrivare solo da z o dal foglietto. E' il compito di
memoria naturale, non sintetico: le entita' introdotte
in W1 ricompaiono in W2.
Misura per lingua: loss di finestra 2 con stato portato,
dimenticato, mescolato; ablazioni comm/foglietto/universi.
python murlen/tests/universi_libro_finestre.py [cuda|cpu] [passi]"""
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
from murlen.ponte import vocab_da_sequenze, PAD_ID  # noqa: E402
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import (  # noqa: E402
    K, allena, sorpresa_simbolo)
from universi_libro import LIBRI  # noqa: E402

WIN, PASSI = 256, 300


def libri(per_lingua=400, seed=0):
    """per_lingua file a caso per lingua, lunghi almeno
    4*WIN per avere coppie di finestre contigue."""
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
            if len(t) < 4 * WIN:
                continue
            testi[lg].append(t)
            if len(testi[lg]) >= per_lingua:
                break
    return testi


def coppie(testi, r, per_testo=8):
    """(lingua, sequenza di 2*WIN caratteri consecutivi)."""
    out = []
    for lg, lista in testi.items():
        for t in lista:
            possibili = list(range(0, len(t) - 2 * WIN,
                                   WIN))
            r.shuffle(possibili)
            for a in possibili[:per_testo]:
                out.append((lg, [ord(c)
                                 for c in t[a:a + 2 * WIN]]))
    r.shuffle(out)
    return out


@torch.no_grad()
def valuta(core, testa, W, dev, tag=''):
    """Loss finestra 2 con stato portato/dimenticato/
    mescolato e ablazioni — come il dialogo.
    Ogni condizione e' scomposta in fette derivate dai
    dati (non scelte): 'ric' = il simbolo da predire
    compariva gia' in W1; 'sor' = compariva fra le
    posizioni sorprendenti di W1 (quelle che il
    foglietto ha scritto, stesso fence); 'nuo' = mai
    visto in W1. La media globale diluisce: se la
    memoria lavora, il vantaggio di 'portati' deve
    concentrarsi su ric/sor."""
    core.eval()
    ids1, p1 = W[0]
    ids2, p2 = W[1]
    X1, Y1 = ids1[:, :-1], ids1[:, 1:]
    X2, Y2 = ids2[:, :-1], ids2[:, 1:]
    B, T = Y2.shape
    reale = ~p2[:, 1:]

    # insiemi per riga: simboli visti in W1; simboli che
    # erano sorprendenti in W1 (stesso fence del foglietto)
    _, h1 = core(X1, p1[:, :-1], causal=True)
    lo1 = testa(h1)
    V = lo1.size(-1)
    visti = torch.zeros(B, V, device=dev)
    visti.scatter_add_(1, ids1, torch.ones_like(ids1,
                                              dtype=torch.float))
    s1 = sorpresa_simbolo(lo1, Y1)
    fin = torch.isfinite(s1)
    ss = torch.where(fin, s1, s1.new_zeros(()))
    q1 = ss.quantile(.25, 1, keepdim=True)
    q3 = ss.quantile(.75, 1, keepdim=True)
    sm = fin & (s1 > q3 + 1.5 * (q3 - q1)) & ~p1[:, :-1]
    sm[:, 0] = ~p1[:, 0]
    sorprese = torch.zeros(B, V, device=dev)
    sorprese.scatter_add_(1, X1, sm.float())
    ric = visti.gather(1, Y2) > 0
    sor = sorprese.gather(1, Y2) > 0
    print('%sfette: %d ric, %d sor, %d nuo' %
          (tag, (ric & reale).sum().item(),
           (sor & reale).sum().item(),
           (~ric & reale).sum().item()), flush=True)

    def seconda(m, z0, mem0):
        _, h2 = m(X2, p2[:, :-1], causal=True, z0=z0,
                  mem0=mem0)
        ce = nn.functional.cross_entropy(
            testa(h2).flatten(0, 1), Y2.flatten(),
            reduction='none', ignore_index=PAD_ID
        ).view(B, T)

        def media(mask):
            mm = mask & reale
            return ce[mm].mean().item() if mm.any() \
                else float('nan')
        return (media(reale), media(ric), media(sor),
                media(~ric))

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
        t, r, s, n = seconda(m, z1, mem1)
        print('%-42s %5.3f  ric %5.3f  sor %5.3f  '
              'nuo %5.3f' % (tag + nome, t, r, s, n),
              flush=True)
        del m
        if dev == 'cuda':
            torch.cuda.empty_cache()

    prova('z+foglietto portati')
    prova('tutto dimenticato', modo='dimentica')
    prova('solo z dimenticato (foglietto tenuto)',
          modo='no_z')
    prova('solo foglietto dimenticato (z tenuto)',
          modo='no_mem')
    prova('z+foglietto mescolati', modo='mescola')
    prova('senza comunicazione (lam=0)',
          lambda m: m.comm.lam.zero_())
    prova('senza foglietto (mem.lam=0)',
          lambda m: m.foglio.lam.zero_())


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else PASSI
    r = random.Random(1)
    testi = libri()
    train_t, test_t = {}, {}
    for lg, lista in testi.items():
        nt = max(1, len(lista) // 10)
        test_t[lg], train_t[lg] = lista[:nt], lista[nt:]
    train = [s for _, s in coppie(train_t, r, per_testo=64)]
    test = coppie(test_t, r, per_testo=4)
    per_lingua = {lg: [s for l2, s in test if l2 == lg]
                  for lg in LIBRI}
    print('%d coppie train, %d test' % (len(train),
                                        len(test)))
    vocab, n = vocab_da_sequenze(train +
                                 [s for _, s in test])
    print('vocabolario: %d simboli' % n)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=K)
    testa = None
    ck_nome = os.path.join(os.path.dirname(
        os.path.abspath(__file__)),
        'dati', 'universi_libro_finestre.pt')
    if 'prosegui' in sys.argv and os.path.exists(ck_nome):
        ck = torch.load(ck_nome, map_location=dev,
                        weights_only=False)
        core.load_state_dict(ck['core'])
        from murlen.ponte import TestaPredizione
        testa = TestaPredizione(
            n, core.enc.tok.embedding_dim)
        testa.load_state_dict(ck['testa'])
        print('ripresa dal checkpoint')
    hist, testa = allena(core, finestre(train, vocab, dev),
                         vocab, dev, passi=passi, batch=8,
                         testa=testa)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(os.path.dirname(
                   os.path.abspath(__file__)),
                   'dati', 'universi_libro_finestre.pt'))
    print('train: %.3f -> %.3f' % (hist[0],
                                   sum(hist[-10:]) / 10))
    for lg in LIBRI:
        print('=== %s ===' % lg)
        valuta(core, testa,
               finestre(per_lingua[lg][:16], vocab, dev),
               dev)
