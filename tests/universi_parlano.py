"""Gli universi si diversificano e si parlano?
Allena un modello (K=3, comm, tempo=64), poi misura la loss su
sequenze mai viste zittendo comunicazione e voce degli universi.
python murlen/tests/universi_parlano.py [cuda|cpu]"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import vocab_da_sequenze, impara_predizione  # noqa: E402
from confronto_init import dati, valuta, PASSI  # noqa: E402

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    train, test = dati(PASSI * 16, 1), dati(64, 2)
    vocab, n = vocab_da_sequenze(train + test)
    torch.manual_seed(0)
    core = MultiUniverse(n, K=3)
    _, testa = impara_predizione(core, train, vocab, passi=PASSI,
                                 device=dev, batch=16, diversita=0.5)
    c = core.comm
    print('forza comunicazione lam = %.3f (nasce 0.1)' % c.lam.item())
    # chi ascolta chi: matrice (K,K) media sulle sequenze
    # di test — righe = slot dell'universo che ascolta,
    # colonne = voce ascoltata
    from murlen.ponte import tensorizza
    ids, pad = tensorizza(test[:8], vocab)
    core.eval()
    with torch.no_grad():
        z, _ = core(ids.to(dev), pad.to(dev))
        _, w = c(z)
    w = w.mean(0).cpu()
    print('chi ascolta chi:')
    for k in range(core.K):
        print('  universo %d ascolta' % k,
              [round(float(x), 3) for x in w[k]])
    print('voce di ogni universo nella previsione (norma zinj):',
          [round(x, 3) for x in core.zinj.weight.view(
              core.zinj.out_features, core.K, -1).norm(dim=(0, 2)).tolist()])

    def prova(nome, f=None):
        m = copy.deepcopy(core)
        if f:
            with torch.no_grad():
                f(m)
        loss, sim = valuta(m, testa, test, vocab, dev)
        print('%-40s loss %.3f  somiglianza universi %.3f'
              % (nome, loss, sim))

    prova('normale')
    prova('senza comunicazione (lam=0)', lambda m: m.comm.lam.zero_())
    prova('universi muti (zinj=0)', lambda m: m.zinj.weight.zero_())
    d = core.zinj.out_features
    for k in range(core.K):
        prova('universo %d muto' % k,
              lambda m, k=k: m.zinj.weight[:, k * d:(k + 1) * d].zero_())
