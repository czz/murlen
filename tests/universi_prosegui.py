"""Prosecuzione generica: carica il checkpoint di una
prova (dialogo, catena) e continua l'allenamento a
pos_modi=0, poi rivaluta. Per misurare se un meccanismo
(binding, catena) e' solo lento a scoprirsi.
python murlen/tests/universi_prosegui.py [dialogo|catena|dialogo_un] [cuda|cpu] [passi]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import TestaPredizione  # noqa: E402
import universi_dialogo as ud  # noqa: E402
import universi_catena as uc  # noqa: E402
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import allena, K  # noqa: E402

DATI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'dati')
COMPITI = {
    'dialogo': (ud.dati_dialogo, 'universi_dialogo.pt', {}),
    'catena': (uc.dati_catena, 'universi_catena.pt', {}),
    'dialogo_un': (ud.dati_dialogo, 'universi_dialogo_1f.pt',
                   {'un_fatto': True}),
}

if __name__ == '__main__':
    compito = sys.argv[1] if len(sys.argv) > 1 else 'dialogo'
    dev = sys.argv[2] if len(sys.argv) > 2 else 'cuda'
    passi = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    dati, ck_nome, kw = COMPITI[compito]
    ck = torch.load(os.path.join(DATI, ck_nome),
                    map_location=dev, weights_only=False)
    vocab = ck['vocab']
    core = MultiUniverse(max(vocab.values()) + 1, K=K)
    core.load_state_dict(ck['core'])
    testa = TestaPredizione(max(vocab.values()) + 1,
                            core.enc.tok.embedding_dim)
    testa.load_state_dict(ck['testa'])
    train = dati(passi * 8, 3, pos_modi=0, **kw)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8, testa=testa)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(DATI, ck_nome))
    print('prosecuzione: %.3f -> %.3f'
          % (hist[0], sum(hist[-10:]) / 10))
    test = dati(64, 2, **kw)
    ud.valuta(core, testa, finestre(test, vocab, dev),
              dev, vocab, un_fatto=kw.get('un_fatto', False))
