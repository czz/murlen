"""scala_store — le soglie dello Store, misurate.

M_CONSOLIDA (quante conferme prima di credere) e
recenti di dimentica (quanto oblio) sono state
dichiarate, mai misurate.

Mondo simulato: regole vere ricorrenti (zipf:
poche frequenti, molte rare) + rumore osservato
una volta. Misura: precisione (le consolidate
erano ricorrenti?) e richiamo (le ricorrenti si
sono consolidate?), piu' quanto oblio serve a
tenere le ultime senza cancellare il passato.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from murlen.store import Store, _rid      # noqa: E402

N_VERE, N_RUMORE, RICORRENTE = 40, 300, 3


def mondo(seed=0):
    """Osservazioni: 40 regole vere con frequenze
    zipf-like, 300 regole-rumore viste 1-2 volte.
    -> lista di (forma, e' vera)."""
    import random
    r = random.Random(seed)
    freq = sorted((1 / (i + 1) for i in
                   range(N_VERE)), reverse=True)
    tot = sum(freq)
    flusso = []
    for i, f in enumerate(freq):
        for _ in range(int(f / tot * 400) + 1):
            flusso.append(('v%d' % i, True))
    for i in range(N_RUMORE):
        for _ in range(r.randint(1, 2)):
            flusso.append(('n%d' % i, False))
    r.shuffle(flusso)
    return flusso


def prova_consolida(m, seed=0):
    st = Store(os.path.join(
        tempfile.mkdtemp(), 's.jsonl'),
        m_consolida=m)
    vere = set()
    for forma, vera in mondo(seed):
        if vera:
            vere.add(forma)
        st.add('f', forma, {'x': forma}, 'mondo',
               ident=[forma])
    cons = {r['forma'] for r in st.consolidate()}
    prec = len(cons & vere) / max(len(cons), 1)
    # ricorrenti: viste >= RICORRENTE volte
    ric = vere  # le vere sono tutte ricorrenti
    ric_cons = cons & ric
    return prec, len(ric_cons), len(cons)


def prova_recenti(recenti, n=2000, seed=0):
    """2000 episodi mai confermati + alcune
    regole confermate. dimentica deve tenere
    gli ultimi senza toccare le confermate."""
    st = Store(os.path.join(
        tempfile.mkdtemp(), 's.jsonl'))
    vive = []
    for i in range(n):
        r = st.add('ep', 'e%d' % i, {'i': i},
                   'mondo', ident=[i])
        if i % 100 == 7:      # alcune confermate
            st.conferma(r)
            vive.append('e%d' % i)
    prima = len(st.attive('ep'))
    st.dimentica('ep', recenti=recenti)
    dopo = st.attive('ep')
    conf_vive = all(any(r['forma'] == v
                        for r in dopo)
                    for v in vive)
    return prima, len(dopo), conf_vive


if __name__ == '__main__':
    print('scala_store — soglie dello Store')
    print()
    print('M_CONSOLIDA: consolidate vere / tutte '
          '(precisione) e quante ricorrenti '
          'consolidate')
    for m in (1, 2, 3, 5):
        prec, n_ric, n_cons = prova_consolida(m)
        print('  m=%d  prec=%.2f  ricorrenti '
              'consolidate=%d/%d  tot.cons=%d'
              % (m, prec, n_ric, N_VERE, n_cons))
    print()
    print('recenti (dimentica su 2000 episodi):')
    for r_ in (50, 500, 5000):
        prima, dopo, ok = prova_recenti(r_)
        print('  recenti=%-5d attive %d -> %d  '
              'confermate preservate: %s'
              % (r_, prima, dopo, ok))
