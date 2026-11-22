"""Quante letture per fatto offre un libro INTERO?

sonda_rif_lontani (pezzi da 2 finestre): riuso lontano ~
2.7 posizioni/pezzo, sotto il ginocchio di ~4 letture per
fatto che accende il canale. Ma i pezzi erano corti — un
libro intero puo' riusare nomi/termini per capitoli. Qui
la statistica pura sul testo completo, senza modello:

per ogni contesto di G caratteri ripetuto nel libro:
- occorrenze totali (le "letture per fatto")
- letture LONTANE: occorrenze a distanza > CAMPO dalla
  precedente — quelle dove la traccia e' l'unica via
- continuazione consistente con la prima occorrenza
  (deterministica) o no

Se i contesti con >=4 letture lontane sono molti, il carico
per il lettore esiste nel libro intero e la risposta e'
"leggere abbastanza basta". Se sono pochi anche su testi
lunghi, la densita' di riuso non-locale e' rara per
natura — e il canale va alimentato diversamente.

python murlen/tests/sonda_densita_libro.py [per_lingua|CARTELLA]"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from universi_libro import libri  # noqa: E402
from sonda_campo_libro import CAMPO  # noqa: E402

G = 16


def stat_libro(t):
    """Conta per contesto ripetuto: occorrenze totali e
    occorrenze lontane (>CAMPO dalla precedente) con
    continuazione = alla prima."""
    prima = {}
    occ, lon, ultimo = {}, {}, {}
    for q in range(G, len(t)):
        c = t[q - G:q]
        if c in prima:
            occ[c] = occ.get(c, 1) + 1
            q0, sym0 = prima[c]
            if q - ultimo[c] > CAMPO and t[q] == sym0:
                lon[c] = lon.get(c, 0) + 1
            ultimo[c] = q
        else:
            prima[c] = (q, t[q])
            ultimo[c] = q
    return occ, lon


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else '60'
    if os.path.isdir(arg):
        # libri lunghi scaricati a parte (fuori repo):
        # la densita' su romanzi veri vs wikijunior
        lib = [open(f, encoding='utf-8', errors='ignore'
                    ).read()
               for f in sorted(glob.glob(
                   os.path.join(arg, '*.txt')))]
    else:
        testi = libri(per_lingua=int(arg))
        lib = [t for lista in testi.values() for t in lista]
    print('%d libri interi' % len(lib), flush=True)

    n_occ, n_lon, occ_dist = [], [], []
    londisti, lunghi = [], []
    for t in lib:
        lunghi.append(len(t))
        occ, lon = stat_libro(t)
        occ_dist += list(occ.values())
        n_occ.append(len(occ))
        n_lon.append(sum(lon.values()))
        londisti += list(lon.values())
    occ_dist.sort()
    londisti.sort()
    lunghi.sort()

    def med(v):
        return v[len(v) // 2] if v else 0

    print('lunghezza mediana libro: %d caratteri' % med(
        lunghi))
    print('contesti ripetuti per libro: mediana %d' % med(
        n_occ))
    print('occorrenze per contesto: mediana %d, '
          '>=4: %d%%' % (
              med(occ_dist),
              100 * sum(1 for x in occ_dist if x >= 4)
              // max(1, len(occ_dist))))
    print('letture lontane deterministiche per libro: '
          'mediana %d, totale %d' % (med(n_lon),
                                     sum(n_lon)))
    if londisti:
        print('letture lontane per contesto: mediana %d, '
              '>=4: %d%% (%d contesti)' % (
                  med(londisti),
                  100 * sum(1 for x in londisti if x >= 4)
                  // len(londisti), len(londisti)))
