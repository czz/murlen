"""Opzione (a): il binding a due fatti e' solo lento a
scoprirsi? Carica il checkpoint del dialogo e continua
l'allenamento a pos_modi=0 (i modi alla distanza piena),
poi rivaluta. Se la 1a giunzione di finestra 2 scende,
serviva solo piu' gradiente.
python murlen/tests/universi_dialogo_prosegui.py [cuda|cpu] [passi]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.model import MultiUniverse  # noqa: E402
from murlen.ponte import TestaPredizione  # noqa: E402
import universi_dialogo as ud  # noqa: E402
from universi_chiave import finestre  # noqa: E402
from universi_oltre_finestra import allena, K  # noqa: E402

DATI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'dati')

if __name__ == '__main__':
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    passi = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ck = torch.load(os.path.join(DATI, 'universi_dialogo.pt'),
                    map_location=dev, weights_only=False)
    vocab = ck['vocab']
    core = MultiUniverse(max(vocab.values()) + 1, K=K)
    core.load_state_dict(ck['core'])
    testa = TestaPredizione(max(vocab.values()) + 1,
                            core.enc.tok.embedding_dim)
    testa.load_state_dict(ck['testa'])
    train = ud.dati_dialogo(passi * 8, 3, pos_modi=0)
    hist, testa = allena(
        core, finestre(train, vocab, dev), vocab, dev,
        passi=passi, batch=8, testa=testa)
    torch.save({'core': core.state_dict(),
                'testa': testa.state_dict(),
                'vocab': vocab},
               os.path.join(DATI, 'universi_dialogo.pt'))
    print('prosecuzione: %.3f -> %.3f'
          % (hist[0], sum(hist[-10:]) / 10))
    test = ud.dati_dialogo(64, 2)
    ud.valuta(core, testa, finestre(test, vocab, dev),
              dev, vocab)
