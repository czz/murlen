"""MURLEN — Multi-Universe Learning/Reasoning Network.

Impara a prevedere sequenze di simboli (id interi):
l'errore di predizione e' il segnale di
apprendimento, nessuna etichetta.

Moduli: model (MultiUniverse: encoder condiviso +
K universi), ponte (simboli -> token, training
di predizione), store (memoria persistente delle
regole).
"""
