"""La saturazione vera esiste?

sonda_vigilanza ha misurato: la soglia assoluta e' cieca
alla saturazione, ma il segnale di stallo (pendenza) e'
separabile. Manca il presupposto: un carico dove la mente
si ferma DAVVERO — non "impara lento" (E2 si e' chiuso a
840 passi), ma "non ce la fa".

Metodo: il compito di scala.py scalato sulla difficolta'
(DIST, la distanza dei fatti — oltre la finestra il
trasporto tocca a z+foglietto). A ogni livello due corpi,
training lungo con valutazioni intermedie:

- se il piccolo stalla a ce_ric alta mentre il grande
  continua a scendere -> saturazione vera a quel livello:
  il carico per testare la vigilanza-stallo
- se entrambi scendono -> il corpo basta, salire di livello
- se entrambi stallano uguali -> il limite e' del compito/
  del metodo, non del corpo

python murlen/tests/sonda_saturazione.py [cuda|cpu]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import scala as SC  # noqa: E402
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import (vocab_da_sequenze,  # noqa: E402
                          TestaPredizione)

PASSI, OGNI = 400, 50
LIVELLI = [200, 600, 900]   # DIST < L-2: oltre la
                            # sequenza non ci sono richiami
CORPI = {
    'piccolo': dict(d=96, enc_layers=4, m=16, nhead=4, K=1),
    'grande': dict(d=384, enc_layers=8, m=32, nhead=8, K=8),
}


def prova(corpo, livello, dev):
    """Un corpo su un livello: ce_ric ogni OGNI passi."""
    SC.DIST = livello
    seqs, _ = SC.dati(SC.N_TRAIN, 0)
    ev, ev_rich = SC.dati(SC.N_EVAL, 999)
    vocab, nv = vocab_da_sequenze(seqs + ev)
    torch.manual_seed(0)
    cfg = dict(SC.DEF, **CORPI[corpo])
    core = MultiUniverse(
        nv, K=cfg['K'], T=cfg['T'], init='projected',
        enc_layers=cfg['enc_layers'], d=cfg['d'], m=cfg['m'],
        nhead=cfg['nhead'], ctx=cfg['ctx'],
        tempo=cfg['tempo'], jitter=cfg['jitter']).to(dev)
    W = SC.finestre(seqs, vocab, cfg['ctx'])
    testa = TestaPredizione(nv, cfg['d']).to(dev)
    opt = torch.optim.AdamW(
        list(core.parameters()) + list(testa.parameters()),
        lr=cfg['lr'])
    curva = []
    for i in range(PASSI):
        sel = torch.randperm(SC.N_TRAIN)[:SC.BATCH]
        SC.passo(core, testa, opt, W, sel, cfg['clip'])
        if (i + 1) % OGNI == 0:
            _, ric = SC.valuta(core, testa, ev, ev_rich,
                               vocab, cfg['ctx'])
            curva.append(ric)
            print('  %-7s dist=%-4d passo %-4d ce_ric %.3f'
                  % (corpo, livello, i + 1, ric), flush=True)
    del core, testa, opt, W
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return curva


def main(dev):
    SC.DEV = dev
    for liv in LIVELLI:
        print('=== distanza %d ===' % liv, flush=True)
        ris = {c: prova(c, liv, dev) for c in CORPI}
        for c, cv in ris.items():
            print('%-8s %s' % (c, '  '.join(
                '%.2f' % x for x in cv)), flush=True)
        # stallo: la crescita della curva nell'ultimo quarto
        pic, gra = ris['piccolo'], ris['grande']
        scende_g = gra[-2] - gra[-1] > 0.05
        fermo_p = pic[-2] - pic[-1] <= 0.05
        if fermo_p and scende_g and pic[-1] - gra[-1] > 0.5:
            print('-> SATURAZIONE: il piccolo si ferma a %.2f '
                  'mentre il grande scende (%.2f)' % (
                      pic[-1], gra[-1]), flush=True)
        elif not fermo_p and not scende_g:
            print('-> entrambi scendono ancora: corpo e '
                  'livello non bastano a saturare', flush=True)
        elif fermo_p and not scende_g:
            print('-> entrambi fermi: limite del compito, '
                  'non del corpo', flush=True)
        else:
            print('-> il piccolo scende piu\' del grande: '
                  'regime strano, da guardare', flush=True)


if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    main(dev)
