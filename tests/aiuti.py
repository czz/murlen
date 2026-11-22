"""Configurazioni piccole e dati di prova condivisi dai test."""
import torch

from murlen.model import MultiUniverse

D_PICCOLO, M_PICCOLO = 32, 4


def modello(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(K=2, comm=True, tempo=0, enc_layers=2,
               d=D_PICCOLO, m=M_PICCOLO)
    cfg.update(kw)
    return MultiUniverse(vocab, **cfg)


def sequenze_ripetitive():
    """Schema fisso e prevedibile: la loss deve scendere."""
    base = [10, 20, 30, 40, 50]
    return [base * 6, base[1:] + base * 5,
            base[2:] + base * 4]


def sequenze_variabili():
    """Lunghezze diverse: molto padding a destra."""
    return [[3, 5, 7, 9] * k for k in (2, 5, 9, 12)]
