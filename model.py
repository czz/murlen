"""MURLEN — encoder condiviso, K universi, comunicazione.

Ogni universo ha stato e F (aggiornamento) propri;
encoder, init-base e A/G (comunicazione) sono
condivisi fra universi e fra iterazioni.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D, M, NHEAD, CTX = 384, 32, 8, 1024
ENC_LAYERS, T_DEFAULT = 8, 4
TEMPO_UNIVERSI = 64   # tempo di default con K>1
MEM_CAP = 256         # capienza del foglietto episodico
JITTER = 0.1     # rumore di ingresso per
                 # universo in training, in
                 # frazione della scala del
                 # segnale — 0 = input
                 # identico per tutti gli
                 # universi (collassano)


class Block(nn.Module):
    """Pre-LN attn + FFN. Se kv!=None -> cross-attn su kv."""

    def __init__(self, d=D, nhead=NHEAD):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Linear(4 * d, d))
        self._cmask = {}

    def forward(self, x, kv=None, kv_pad=None, causal=False):
        kv = x if kv is None else kv
        mask = None
        if causal:
            n = x.size(1)
            mask = self._cmask.get((n, x.device))
            if mask is None:
                mask = torch.triu(torch.ones(
                    (n, n), dtype=torch.bool,
                    device=x.device), 1)
                self._cmask[(n, x.device)] = mask
        a = self.attn(self.ln1(x), kv, kv, key_padding_mask=kv_pad,
                      attn_mask=mask, need_weights=False)[0]
        x = x + a
        return x + self.ff(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self, vocab, d=D, ctx=CTX, layers=ENC_LAYERS,
                 nhead=NHEAD):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, nhead)
                                    for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, ids, pad, causal=False):
        x = self.tok(ids) + self.pos.weight[: ids.size(1)]
        for b in self.blocks:
            x = b(x, kv_pad=pad, causal=causal)
        return self.ln(x)


class AttnPool(nn.Module):
    """1 query appresa -> riassunto (B,d)."""

    def __init__(self, d=D, nhead=NHEAD):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)

    def forward(self, x, pad=None):
        return self.attn(self.q.expand(x.size(0), -1, -1), x, x,
                         key_padding_mask=pad,
                         need_weights=False)[0].squeeze(1)


class UniverseInit(nn.Module):
    """z_k^(0) = I(h, e_k). kind in {'learned','projected'}."""

    def __init__(self, K, kind='learned', d=D, m=M, nhead=NHEAD):
        super().__init__()
        self.K, self.kind = K, kind
        if kind == 'learned':
            self.E = nn.Parameter(torch.randn(K, m, d) * 0.02)
            self.pool = AttnPool(d, nhead)
            self.P = nn.Linear(d, d)
        elif kind == 'projected':
            self.q0 = nn.Parameter(torch.randn(1, m, d) * 0.02)
            self.xa = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.W = nn.Parameter(torch.zeros(K, d, d))
            self.b = nn.Parameter(torch.zeros(K, 1, d))
            self.ln = nn.LayerNorm(d)
        else:
            raise ValueError(kind)

    def forward(self, h, pad=None):
        B = h.size(0)
        if self.kind == 'learned':
            pooled = self.P(self.pool(h, pad)).unsqueeze(1).unsqueeze(1)
            return self.E.unsqueeze(0) + pooled             # (B,K,M,d)
        z0 = self.xa(self.q0.expand(B, -1, -1), h, h,
                     key_padding_mask=pad,
                     need_weights=False)[0]                  # (B,M,d)
        return self.ln(torch.einsum('bmd,kde->bkme', z0, self.W)
                       + self.b.unsqueeze(0))


class UpdateBlock(nn.Module):
    """F di un universo: cross-attn su h +
    self-attn fra slot + FFN.
    In training ogni universo riceve una vista
    leggermente rumorosa dello stesso flusso
    (JITTER): insieme all'init e ai blocchi
    separati rompe la simmetria fra universi.
    In inferenza il rumore e' spento."""

    def __init__(self, d=D, nhead=NHEAD, jitter=JITTER):
        super().__init__()
        self.jitter = jitter
        self.xattn, self.self_attn, self.ff = (
            Block(d, nhead) for _ in range(3))
        # il gate a contenuto: quanto della proposta
        # diventa permanente lo decide lo slot, in base a
        # cio' che sta arrivando — non una velocita' fissa.
        # Nasce a zero: all'avvio conta solo il gate fisso
        # (self.gate di MultiUniverse) e i vecchi ckpt
        # caricano una funzione identica. In un fork del
        # generatore: non consuma casualita' condivisa
        with torch.random.fork_rng():
            self.gate_lin = nn.Linear(d, 1)
        nn.init.zeros_(self.gate_lin.weight)
        nn.init.zeros_(self.gate_lin.bias)

    def forward(self, z, h, hpad):
        B, K, M, d = z.shape
        zf = z.reshape(B * K, M, d)
        hf = h.unsqueeze(1).expand(B, K, *h.shape[1:]).reshape(
            B * K, *h.shape[1:])
        if self.training and self.jitter > 0:
            hf = hf + torch.randn_like(hf) * \
                hf.std(-1, keepdim=True) * self.jitter
        pf = hpad.unsqueeze(1).expand(B, K, hpad.size(1)).reshape(
            B * K, -1)
        zf = self.xattn(zf, hf, pf)
        zf = self.self_attn(zf)
        zf = self.ff(zf)
        g = self.gate_lin(zf)
        return zf.view(B, K, M, d), g.view(B, K, M, 1)


class UpdK(nn.Module):
    """K blocchi separati: un UpdateBlock per
    universo, nessun peso condiviso. La
    diversita' sta nell'architettura, non in
    una loss che la premia: con un F
    condiviso gli universi convergono allo
    stesso stato."""

    def __init__(self, K, d=D, nhead=NHEAD, jitter=JITTER):
        super().__init__()
        self.blocks = nn.ModuleList(
            [UpdateBlock(d, nhead, jitter) for _ in range(K)])

    def forward(self, z, h, hpad):
        zs, gs = [], []
        for k in range(len(self.blocks)):
            zk, gk = self.blocks[k](z[:, k:k + 1], h, hpad)
            zs.append(zk)
            gs.append(gk)
        return torch.cat(zs, 1), torch.cat(gs, 1)


class Comm(nn.Module):
    """Il palcoscenico: s_k=AttnPool(z_k) sono le voci.
    Sul canale passa solo la DEVIAZIONE di ogni voce dal
    coro (s - media): il contenuto comune a tutti gli
    universi non entra nel canale — la deviazione e' il
    segnale, non il bulk. Se le voci dicono la stessa
    cosa, il palco resta muto da solo.
    Ogni slot di ogni universo legge le K deviazioni per
    contenuto (cross-attn): chi ascolta chi dipende dal
    contesto — prima era un voto globale (mix), uguale
    per ogni posizione: niente da contendersi, infatti
    lam non imparava. Non dipende da K: in cresci() non
    serve toccarla. Il mutismo e' su lam, non su read:
    con read a zero anche il gradiente di lam sarebbe
    zero — comm resterebbe morta per sempre.
    Costruita in un fork del generatore per non
    consumare casualita' condivisa.
    Restituisce (z', w) con w (B,K,K): quanto gli slot
    di un universo ascoltano ogni voce — la matrice
    della competizione."""

    def __init__(self, d=D, nhead=NHEAD):
        super().__init__()
        self.pool = AttnPool(d, nhead)
        with torch.random.fork_rng():
            self.read = nn.MultiheadAttention(
                d, nhead, batch_first=True)
        self.lam = nn.Parameter(torch.tensor(0.1))

    def forward(self, z):
        B, K, M, d = z.shape
        s = torch.stack([self.pool(z[:, k])
                         for k in range(K)], 1)        # voci (B,K,d)
        s = s - s.mean(1, keepdim=True)    # solo deviazioni
        sf = s.unsqueeze(1).expand(B, K, K, d).reshape(
            B * K, K, d)
        r, w = self.read(z.reshape(B * K, M, d), sf, sf)
        w = w.view(B, K, M, K).mean(2)   # chi ascolta chi
        return z + self.lam * r.view(B, K, M, d), w


class Foglietto(nn.Module):
    """Traccia episodica indirizzata a contenuto.
    Le posizioni insolite del blocco lasciano una copia
    (snapshot, detach) del proprio h; le posizioni
    successive la interrogano per contenuto — come
    zread, ma su tracce che sopravvivono oltre il
    contesto e oltre la memoria di lavoro z.
    La scrittura non avviene dentro forward: dopo ogni
    finestra il chiamante consolida con annota(),
    passando la sorpresa per posizione — l'errore di
    predizione decide cosa lascia traccia (fence dei
    quartili sulla sua distribuzione, mai soglia a
    mano). Le tracce sono snapshot: il passato non si
    riscrive, il gradiente arriva solo al lato lettura
    (query e lam). Canale su lam come comm: se il
    foglietto serve, lam cresce; se no si chiude da
    solo. Costruito in un fork del generatore per non
    consumare casualita' condivisa.
    Con scrittura=True la scelta di cosa tracciare e'
    appresa, non affidata alla sorpresa: ogni posizione
    lascia una traccia pesata da un gate sigmoide
    (peso ~0 = non scritta), e il gradiente arriva al
    gate attraverso le finestre successive via mem0 —
    il modello impara a scegliere cosa servira'."""

    def __init__(self, d=D, nhead=NHEAD, scrittura=False,
                 coppia=False, ancora=False):
        super().__init__()
        self.ancora = ancora and coppia
        with torch.random.fork_rng():
            self.read = nn.MultiheadAttention(
                d, nhead, batch_first=True)
            self.scrittura = nn.Linear(d, 1) \
                if scrittura else None
            # chiavi a coppia di simboli: la traccia e'
            # indirizzata dal binding [s_{p-1}, s_p],
            # non da h — il query lato lettura proietta
            # h nello spazio delle chiavi (2d)
            if self.ancora:
                # meta' sinistra della query e' l'embedding
                # del simbolo corrente (filtro di famiglia
                # senza apprendimento); solo la meta'
                # destra si impara — nasce a zero: il
                # filtro domina finche' serve
                self.qpair = nn.Linear(d, d)
                nn.init.zeros_(self.qpair.weight)
                nn.init.zeros_(self.qpair.bias)
            else:
                self.qpair = nn.Linear(d, 2 * d) \
                    if coppia else None
        # con le chiavi a coppia il canale nasce rumoroso:
        # init muto si e' chiuso da solo (letture casuali
        # peggiorano la loss -> il gate si abbassa) —
        # partire alto da' a qpair il gradiente per
        # imparare l'indirizzamento
        self.lam = nn.Parameter(torch.tensor(
            1. if coppia else 0.1))

    def scegli_valori(self, v, pad_j):
        """(B,N) bool: posizioni sopra il fence dei quartili
        (Q3 + 1.5 IQR) della distribuzione di v nella
        sequenza — la soglia esce dai dati, non e' scelta."""
        q1 = v.quantile(.25, 1, keepdim=True)
        q3 = v.quantile(.75, 1, keepdim=True)
        return (v > q3 + 1.5 * (q3 - q1)) & ~pad_j


class MultiUniverse(nn.Module):
    """Core latente: ids -> (z_final (B,K,M,d), h (B,N,d)).
    K=1 -> baseline A; K>1 + comm -> varianti.
    Dimensioni impostabili (default = costanti
    in cima al file): d, m, nhead (d deve
    essere divisibile per nhead), ctx,
    enc_layers, jitter.
    Con K>1, se non indicati, comm=True e
    tempo=TEMPO_UNIVERSI: gli universi si
    parlano ed entrano nella previsione,
    quindi imparano. Con K=1: comm=False,
    tempo=0. Il foglietto episodico
    (mem) segue tempo: attivo quando il
    flusso va a blocchi, spento con
    tempo=0 — e con lui spenta anche la
    sua memoria."""

    def __init__(self, vocab, K=1, T=T_DEFAULT, init='learned',
                 comm=None, enc_layers=ENC_LAYERS, d=D, m=M,
                 tempo=None, nhead=NHEAD, ctx=CTX, jitter=JITTER,
                 mem=None, mem_gate=False, coppia=False,
                 ancora=False, simbolo=False, mem_cap=None):
        super().__init__()
        if d % nhead:
            raise ValueError('d=%d non e\' divisibile per nhead=%d'
                             % (d, nhead))
        # scelte non esplicite: cresci() le
        # accende al passaggio da 1 a 2 universi
        self._comm_auto = comm is None
        self._tempo_auto = tempo is None
        self._mem_auto = mem is None
        if comm is None:
            comm = K > 1
        if tempo is None:
            tempo = TEMPO_UNIVERSI if K > 1 else 0
        if mem is None:
            mem = tempo > 0    # il foglietto vive nel
                               # flusso a blocchi: senza
                               # tempo non c'e' dove
                               # appuntarsi
        self.K, self.T, self.comm_on = K, T, comm
        self.tempo = tempo
        self.mem_on = mem
        self.nhead, self.jitter = nhead, jitter
        self.enc = Encoder(vocab, d=d, ctx=ctx, layers=enc_layers,
                           nhead=nhead)
        self.init = UniverseInit(K, init, d=d, m=m, nhead=nhead)
        self.upd = UpdK(K, d, nhead, jitter)
        self.comm = Comm(d, nhead) if comm else None
        # la voce degli universi nel flusso: z del
        # passato entra in h — zero-init: all'avvio
        # muta, impara a parlare solo se serve
        self.zinj = nn.Linear(K * d, d)
        nn.init.zeros_(self.zinj.weight)
        nn.init.zeros_(self.zinj.bias)
        # ...e l'ascolto a contenuto: ogni posizione di h
        # interroga gli slot di z (K*M chiavi) invece di
        # riceverli a pioggia. Nasce muta (out_proj a
        # zero) e non dipende da K: in cresci() non serve
        # ridimensionarla. Costruita in un fork del
        # generatore: non consuma casualita' condivisa,
        # cosi' l'init di tutto il resto non cambia
        with torch.random.fork_rng():
            self.zread = nn.MultiheadAttention(
                d, nhead, batch_first=True)
        nn.init.zeros_(self.zread.out_proj.weight)
        nn.init.zeros_(self.zread.out_proj.bias)
        # il gate della memoria di lavoro: ogni slot ha una
        # sua velocita' di riscrittura imparata. sigmoide
        # 0.5 all'avvio; gli slot che imparano a chiudersi
        # (g->0) diventano le pagine protette: il contenuto
        # sopravvive alle riscritture dei blocchi dopo
        self.gate = nn.Parameter(torch.zeros(K, m, 1))
        # il foglietto: memoria episodica a contenuto,
        # sorella di z ma fatta di tracce discrete che
        # non si riscrivono. Vive solo nel flusso a
        # blocchi (tempo attivo): li' z e' memoria di
        # lavoro e il foglietto e' l'archivio. Non
        # dipende da K: in cresci() non serve toccarlo
        self.foglio = Foglietto(d, nhead, scrittura=mem_gate,
                                coppia=coppia, ancora=ancora) \
            if mem else None
        self._coppia = coppia
        self._ancora = ancora and coppia
        self._simbolo = simbolo
        self._mem_cap = mem_cap or MEM_CAP
        self._coppie = None
        self._mem = []

    @property
    def mem(self):
        """Il foglietto episodico: la lista di
        tracce per elemento del batch, stato di
        runtime da ripassare come `mem0` o dare
        a `salva_sicuro`. In modalita' a blocchi
        il flusso annota dentro forward e i
        ritorni di `annota` si perdono — questo
        e' l'accessor pubblico, `annota` basta
        solo nella modalita' a finestra."""
        return self._mem

    def cresci(self):
        """Neurogenesi: un universo in piu'.
        La voce nuova nasce muta (zinj a
        zero), il voto neutro in comm —
        impara in training come gli altri.
        Passando da 1 a 2 universi accende
        comm e tempo, se non scelti alla
        creazione. Dopo va ricreato
        l'optimizer."""
        init = self.init
        if init.kind == 'learned':
            E = init.E.data
            dev = E.device
            init.E = nn.Parameter(torch.cat(
                [E, torch.randn(
                    1, *E.shape[1:],
                    device=dev) * 0.02]))
        else:
            # projected: trasformazione nuova a
            # zero, come all'init di tutti
            W, b = init.W.data, init.b.data
            dev = W.device
            init.W = nn.Parameter(torch.cat(
                [W, torch.zeros_like(W[:1])]))
            init.b = nn.Parameter(torch.cat(
                [b, torch.zeros_like(b[:1])]))
        init.K += 1
        d = self.zinj.out_features
        W = self.zinj.weight.data
        z2 = nn.Linear((self.K + 1) * d, d).to(
            W.device)
        with torch.no_grad():
            z2.weight.zero_()
            z2.bias.copy_(self.zinj.bias)
            z2.weight[:, :self.K * d] = W
        self.zinj = z2
        # il neonato e' un blocco nuovo:
        # blocco proprio, init fresco — la
        # voce muta (zinj a zero) gli impedisce
        # di disturbare finche' non impara
        self.upd.blocks.append(
            UpdateBlock(d, self.nhead,
                        self.jitter).to(dev))
        self.gate = nn.Parameter(torch.cat(
            [self.gate.data, torch.zeros(
                1, self.gate.size(1), 1, device=dev)]))
        self.K += 1
        if self.K == 2:
            self._accendi_universi(d, dev)

    def _accendi_universi(self, d, device):
        """Da 1 a 2 universi: comunicazione e
        tempo, se non scelti alla creazione.
        La comunicazione nasce muta (lam=0)."""
        if not self.comm_on and self._comm_auto:
            self.comm = Comm(d, self.nhead).to(device)
            with torch.no_grad():
                self.comm.lam.zero_()
            self.comm_on = True
        if not self.tempo and self._tempo_auto:
            self.tempo = TEMPO_UNIVERSI
        if not self.mem_on and self._mem_auto and self.tempo:
            self.foglio = Foglietto(
                d, self.nhead, coppia=self._coppia,
                ancora=self._ancora).to(device)
            with torch.no_grad():
                self.foglio.lam.zero_()
            self.mem_on = True

    def load_state_dict(self, sd, strict=True):
        # ckpt antichi: un unico upd condiviso.
        # Mitosi — ogni universo eredita una
        # COPIA del blocco genitore: funzione
        # identica al caricamento, poi i
        # blocchi divergono in training
        if any(k.startswith('upd.') and not
               k.startswith('upd.blocks.')
               for k in sd):
            sd = dict(sd)
            for k_ in [k for k in sd
                       if k.startswith('upd.')
                       and not
                       k.startswith('upd.blocks.')]:
                v = sd.pop(k_)
                for kk in range(self.K):
                    sd['upd.blocks.%d.%s' % (
                        kk, k_[4:])] = v
        # ckpt senza la testa di lettura: si riempie coi
        # valori di default (muta) — funzione identica
        if not any(k.startswith('zread.') for k in sd):
            sd = dict(sd)
            for k, v in self.zread.state_dict().items():
                sd['zread.' + k] = v
        # ckpt senza il gate a contenuto: si riempie coi
        # default (tutto a zero) — contributo nullo,
        # funzione identica
        if not any('gate_lin.' in k for k in sd):
            sd = dict(sd)
            for k, v in self.upd.state_dict().items():
                if 'gate_lin.' in k:
                    sd['upd.' + k] = v
        # ckpt senza foglietto: si riempie coi default
        # (lam=0.1 ma read fresco) — la lettura riparte
        # da zero come un organo appena nato
        if self.foglio is not None and not any(
                k.startswith('foglio.') for k in sd):
            sd = dict(sd)
            for k, v in self.foglio.state_dict().items():
                sd['foglio.' + k] = v
        # ckpt senza gate: sigmoide 1 = aggiornamento pieno,
        # funzione identica al vecchio codice
        if 'gate' not in sd:
            sd = dict(sd)
            sd['gate'] = torch.full_like(
                self.gate.data, 10.)
        # ckpt col vecchio palco (voto globale mix+W1/W2):
        # quelle chiavi spariscono; la lettura a contenuto
        # si riempie coi default freschi — la nuova comm
        # riparte da zero training, guidata da lam
        if self.comm is not None and any(
                k.startswith('comm.') for k in sd) and \
                not any(k.startswith('comm.read.')
                        for k in sd):
            sd = dict(sd)
            for k in [k for k in sd if k.startswith(
                    ('comm.mix', 'comm.W1', 'comm.W2'))]:
                sd.pop(k)
            for k, v in self.comm.read.state_dict().items():
                sd['comm.read.' + k] = v
        return super().load_state_dict(
            sd, strict)

    def _leggi_foglietto(self, h_j, coppie=None):
        """Le posizioni interrogano le tracce per contenuto."""
        B, d = h_j.size(0), h_j.size(2)
        if not self._mem:
            return h_j
        n = [m.size(0) for m in self._mem]
        S = max(n)
        if not S:
            return h_j
        if self._coppia:
            # le chiavi delle tracce sono coppie di
            # simboli (2d): il lettore proietta h nello
            # spazio chiave con qpair, i valori restano
            # le h depositate
            kk = h_j.new_zeros(B, S, 2 * d)
            vv = h_j.new_zeros(B, S, d)
            kpm = torch.ones(B, S, dtype=torch.bool,
                             device=h_j.device)
            for b in range(B):
                if n[b]:
                    kk[b, :n[b]] = \
                        self._mem[b][:, :2 * d]
                    vv[b, :n[b]] = \
                        self._mem[b][:, 2 * d:]
                    kpm[b, :n[b]] = False
            if self._ancora:
                # il cue e' il simbolo corrente: la meta'
                # sinistra della query e' il suo embedding
                # — seleziona gratis la famiglia di tracce
                # con lo stesso simbolo a sinistra
                if coppie is None:
                    coppie = self._coppie
                if coppie is None or \
                        coppie.size(1) != h_j.size(1):
                    return h_j
                q = torch.cat(
                    [coppie[:, :, d:],
                     self.foglio.qpair(h_j)], -1)
            else:
                q = self.foglio.qpair(h_j)
            att = q @ kk.transpose(1, 2) / (2 * d) ** .5
            att = att.masked_fill(
                kpm[:, None, :], float('-inf')
            ).softmax(-1).nan_to_num(0.)
            return h_j + self.foglio.lam * (att @ vv)
        kv = h_j.new_zeros(B, S, d)
        kpm = torch.ones(B, S, dtype=torch.bool,
                         device=h_j.device)
        for b in range(B):
            if n[b]:
                kv[b, :n[b]] = self._mem[b]
                kpm[b, :n[b]] = False
        return h_j + self.foglio.lam * self.foglio.read(
            h_j, kv, kv, key_padding_mask=kpm,
            need_weights=False)[0]

    def annota(self, h, sorpresa, pad, coppie=None):
        """Consolida: le posizioni il cui simbolo ha sorpreso
        il modello lasciano una traccia nel foglietto —
        l'errore di predizione decide cosa merita memoria
        (la novita' geometrica non bastava: i modi del
        dialogo non erano "strani" in h e non venivano
        mai scritti). Il confine d'episodio (posizione 0)
        e' sempre tracciato. Chiamare dopo forward con la
        sorpresa per posizione; le tracce servono ai
        forward successivi (mem0). Dentro forward, in
        modalita' a blocchi, la chiama il flusso stesso
        a ogni blocco se il chiamante passa `scrivi`
        (o se la scrittura e' appresa): le tracce
        servono ai blocchi dopo.
        -> le tracce consolidate, da ripassare come
        `mem0` al forward successivo (None se il
        foglietto e' spento).
        Con `coppia` attiva la regola cambia: la posizione
        p lascia traccia se la sua coppia di simboli
        (s_{p-1}, s_p) non e' gia' coperta nel buffer —
        ogni binding distinto viene scritto una volta,
        le ripetizioni no; la riga e' (chiave 2d, h) —
        con `simbolo` il valore e' emb(s_p) al posto
        di h: il contenuto richiamato, non lo stato."""
        if not self.mem_on:
            return
        largh = 3 * h.size(2) if self._coppia \
            else h.size(2)
        if not self._mem:
            self._mem = [h.new_zeros(0, largh)
                         for _ in range(h.size(0))]
        if self._coppia:
            if coppie is None:
                coppie = self._coppie
            if coppie is None:
                return self._mem or None
            dd = h.size(2)
            for b in range(h.size(0)):
                pos = torch.nonzero(~pad[b]).flatten()
                if not pos.numel():
                    continue
                ck = coppie[b][pos]
                keep = torch.ones(
                    pos.numel(), dtype=torch.bool,
                    device=h.device)
                if self._mem[b].size(0):
                    eq = (ck[:, None, :] ==
                          self._mem[b][None, :, :2 * dd]
                          ).all(-1)
                    keep &= ~eq.any(1)
                dup = (ck[:, None, :] ==
                       ck[None, :, :]).all(-1)
                keep &= ~dup.triu(1).any(0)
                if keep.any():
                    # simbolo: il contenuto richiamato
                    # e' il simbolo stesso (emb(s_p),
                    # meta' destra della chiave) — un
                    # uomo ricorda la parola, non lo
                    # stato che la prediceva
                    val = ck[keep][:, dd:].detach() \
                        if self._simbolo else \
                        h[b][pos][keep].detach()
                    nuove = torch.cat(
                        [ck[keep], val], -1)
                    self._mem[b] = torch.cat(
                        [self._mem[b], nuove])[-self._mem_cap:]
            return self._mem or None
        if self.foglio.scrittura is not None:
            # scelta appresa: tracce pesate dal gate,
            # niente detach — il credito torna al gate
            # dalle finestre che leggono le tracce
            g = torch.sigmoid(
                self.foglio.scrittura(h)).squeeze(-1)
            for b in range(h.size(0)):
                ok = ~pad[b]
                nuove = h[b][ok] * g[b][ok].unsqueeze(-1)
                if nuove.size(0):
                    self._mem[b] = torch.cat(
                        [self._mem[b], nuove])[-self._mem_cap:]
        else:
            s = sorpresa.clone()
            s[:, 0] = float('inf')
            sel = self.foglio.scegli_valori(s, pad)
            for b in range(h.size(0)):
                nuove = h[b][sel[b]].detach()
                if nuove.size(0):
                    self._mem[b] = torch.cat(
                        [self._mem[b], nuove])[-self._mem_cap:]
        return self._mem or None

    def forward(self, ids, pad, causal=False, z0=None,
                mem0=None, scrivi=None):
        h = self.enc(ids, pad, causal=causal)
        B, N = h.shape[:2]
        if self.mem_on:
            # mem0 valido: B tracce (S, d o 3d con
            # chiavi a coppia). Se la forma non torna
            # si riparte da vuoto — stessa regola di z0
            largh = 3 * h.size(2) if self._coppia \
                else h.size(2)
            ok = mem0 is not None and len(mem0) == B
            if ok:
                try:
                    ok = all(m.dim() == 2 and
                             m.size(1) == largh
                             for m in mem0)
                except AttributeError:
                    ok = False
            self._mem = [m.clone() for m in mem0] if ok \
                else [h.new_zeros(0, largh)
                      for _ in range(B)]
            if self._coppia:
                # chiavi delle tracce: la posizione p
                # e' indirizzata dal binding dei due
                # simboli che la producono (s_{p-1}, s_p)
                emb = self.enc.tok(ids)
                prev = torch.cat(
                    [emb.new_zeros(B, 1, emb.size(2)),
                     emb[:, :-1]], 1)
                self._coppie = torch.cat([prev, emb], -1)
        z = z0
        if z is not None:
            if z.size(0) != B or z.size(1) > self.K:
                z = None
            elif z.size(1) < self.K:
                z = F.pad(z, (0, 0, 0, 0, 0,
                              self.K - z.size(1)))
        if not self.tempo or self.tempo >= N:
            if z is None:
                z = self.init(h, pad)
            for _ in range(self.T):
                z_new, g_c = self.upd(z, h, pad)
                z = z + (self.gate + g_c).sigmoid() * (z_new - z)
                if self.comm_on:
                    z, _ = self.comm(z)
            return z, h
        hs = []
        for a in range(0, N, self.tempo):
            h_j = h[:, a:a + self.tempo]
            p_j = pad[:, a:a + self.tempo]
            if z is not None:
                h_j = h_j + self.zinj(
                    z.mean(2).reshape(B, -1)).unsqueeze(1)
                zs = z.flatten(1, 2)
                h_j = h_j + self.zread(
                    h_j, zs, zs, need_weights=False)[0]
            else:
                z = self.init(h_j, p_j)
            if self.mem_on:
                h_j = self._leggi_foglietto(
                    h_j,
                    self._coppie[:, a:a + self.tempo]
                    if self._ancora else None)
            for _ in range(self.T):
                z_new, g_c = self.upd(z, h_j, p_j)
                z = z + (self.gate + g_c).sigmoid() * (
                    z_new - z)
                if self.comm_on:
                    z, _ = self.comm(z)
            # omeostasi: lo stato portato resta
            # limitato — senza, la ricorrenza
            # diverge in una passata
            z = F.layer_norm(z, (z.size(-1),))
            hs.append(h_j)
            # il blocco appena visto lascia traccia:
            # la leggono i blocchi dopo. scrivi(a,
            # h_j, p_j) -> sorpresa per posizione
            # decide cosa consolidare; con la
            # scrittura appresa decide il gate.
            # Cosi' il foglietto si allena anche
            # dentro una finestra sola, non solo
            # fra finestre (mem0)
            if self.mem_on and (
                    scrivi is not None or
                    self._coppia or
                    self.foglio.scrittura
                    is not None):
                self.annota(
                    h_j,
                    scrivi(a, h_j, p_j)
                    if scrivi else None, p_j,
                    self._coppie[:, a:a + self.tempo]
                    if self._coppia else None)
        return z, torch.cat(hs, 1)


def param_count(m):
    return sum(p.numel() for p in m.parameters())
