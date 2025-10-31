//--This Kernel is Based on the Github Bitcoin Gold Published Network!--
//--All this here, is copied! Work from the BTG Devs here on Github in its base!--
//--All the other files in my Account are not, but this here is completly copied and reworked from me and ai.--

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//--Ermöglicht den byteweisen Zugriff auf Speicherbereiche, was für die Handhabung von Hashes und Teil-Lösungen nützlich ist.--

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

//--Ermöglicht die Nutzung von Subgroups (Untergruppen von Work-Items innerhalb einer Workgroup),
//--eine leistungsstarke Funktion für intra-Workgroup-Synchronisation und Datenaustausch auf Intel-GPUs.--

#define N 144
#define K 5

//--Die Kernparameter des Equihash-Algorithmus. N=144 definiert die Bitlänge der Lösung, K=5 bestimmt die Anzahl der Runden und die
//--erforderliche Speicherbandbreite (der--
//--Speicherbedarf skaliert mit 2^(N/(K+1))).--

#define INPUT_SIZE       (140)   //--Beispielwert, anpassen--
#define ENTRY_SIZE       (32)    //--Beispielwert (Blake2b Blockgröße)--
#define MAX_SOLS        (2000)  //--200 sind angepeilt bei voller Leistung DG2--
#define MAX_COLLISIONS   (16)    //--Puffergröße für Kollisionen--
#define WORKGROUP_SIZE   (64)    //--Optimale Größe für ARC L1-Cache fraglich siehe weitere Anmerkungen--
#define HASH_SIZE      32
#define HT_SIZE        9


//--Konfigurationsparameter für den Kernel.--
//--INPUT_SIZE: Länge der Eingabedaten für den Hash (hier beispielhaft 140 Byte).--
//--ENTRY_SIZE: Größe eines Eintrags in der Hashtabelle (32 Byte für einen 256-Bit Hash-Wert + Metadaten).--
//--MAX_SOLS: Maximale Anzahl an Lösungen, die der Kernel pro Lauf zurückgeben kann.--
//--MAX_COLLISIONS: Maximale Anzahl von Kollisionen, die ein Work-Item lokal zwischenspeichern kann, bevor sie verarbeitet werden.--
//--WORKGROUP_SIZE: Optimale Anzahl von Threads (Work-Items) pro Workgroup. 64 ist eine gängige Größe, die gut auf den L1-Cache vieler //--GPU-Architekturen abgestimmt ist.--
//--Für Intel Arc Grafikkarten wurde ein höherer Wert möglich ermittelt. Da dieses Programm ausschließlich dem Mining mit ARC GPUs
//--dienen soll, wird es auch höhere Werte beinhalten--
//--können und die modernen Speichermengen beachten. Hier reden wir von einem L1 Cache im durschnitt der großen DG2-Chips 512/448 //--von 2,5-4-6MiB L1 und mindestens 6-12-16 MiB L2 Cache. Diese--
//--Werte werden in das Programm als Hauptreferenz Einfließen, weil sie dem groben Druchschnitt entsprechen und nur höher werden in //--den bisher Battlemage Generationswechseln.--
//--Entsprechend weiterer Generationen wie Calestial und Druid wird hier nachgearbeitet werden müssen bei Bedarf.--

__constant uint unit IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

//--Konstantendefinitionen für die Hash-Funktionen.--
//--IV: Initialisierungsvektor (Initial Values) für Blake2s (32-Bit Worte).--
//--BLAKE2B_IV: Initialisierungsvektor für Blake2b (64-Bit Worte), wird im Hauptkernel "zhash_144_5" verwendet.--
//--Bei "IV[8]" handelt es sich um Blake2s (32-Bit Wörter), bei "BLAKE2B_IV[8]" um Blake2b (64-Bit).--
//--Die parallele Nutzung von Blake2s und Blake2b ist eine Eigenart des Equihash-Kernels, da Blake2s für kürzere Blöcke effizient ist,--
//--während Blake2b für die Hauptkompression genutzt wird.“--

__constant ulong8 BLAKE2b_IV[8] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

//--"sigma": Eine Permutationstabelle, die die Reihenfolge festlegt, in der die Nachrichtenblöcke in jeder Runde der Blake2b-Kompressionsfunktion adressiert werden.--
//--Sie sorgt für Vermischung und Sicherheit.--
//--Konstant definiert und "12×16" groß, weil Blake2b 12 Runden hat.--

__constant uchar sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

typedef struct sols_s {
    uint nr;
    uint likely_invalids;
    uint values[2000][512];
    uint valid[2000];
} sols_t;


inline uint rotr32(uint x, uint n) {
    return (x >> n) | (x << (32 - n));
}

//--Inline Rotation Static zu Vector korrigieren.--
//--"ROTR64_8"  Inline Rotation von Statisch Static zu Vector(en) korrigieren--
//--Führt eine bitweise Rotation nach rechts (right rotate) für einen Vektor aus acht 64-Bit-Ganzzahlen "ulong8" um "n" Bits durch. Dies ist-- //--eine fundamentale Operation in vielen--
//--Kryptographie-Hash-Funktionen.--
//--Diese Funktion rotiert jedes Element des Vektors "ulong8" unabhängig um "n" Bits. Dadurch laufen 8--
//--Rotationen parallel in einem "SIMD-Befehl“.--
//--"rotr64_8" nimmt einen "ulong8" (8 parallele 64-Bit-Werte im Vektorregister) und rotiert jedes Element um "n" Bits nach rechts.--
//--Das nutzt OpenCLs SIMD-Architektur: statt 8 Rotationen nacheinander zu machen,--
//--passieren alle gleichzeitig in einem einzigen Vektor-Befehl.--
inline ulong8 rotr64_8(ulong8 x, uint n) {
    return (x >> n) | (x << (64 - n));
}

//--Eine Implementierung der Blake2s-Kompressionsfunktion. Sie komprimiert einen Eingabeblock (input)--
//--fester Länge unter Verwendung des internen Zustands (Teile des IV) und erzeugt einen--
//--Ausgabewert (out). Diese Funktion ist für die anfängliche Generierung der Hash-Werte verantwortlich.--
//--G_VEC: Das Makro innerhalb der Funktion definiert die Mischoperation für einen Vektor aus vier Zustandsworten""v_a, v_b, v_c, v_d" unter //--Verwendung zweier Nachrichtenworte "m_i, m_j".--
//--Entwurf: für Blake2s. Einige Variablen wie input_data und out_data sind Platzhalter und müssten im Kernelkontext definiert werden.“--
//--Entwurf: Blake2s (32-Bit-Variante) den "sigma-Permutations-Array" verwendet.--
//--Implementierung "h[] (Hash-State), input_data[] (Block), und out_data[] (Output)" korrekt initialisieren. Der Code Skizze Blake2s--
//--Parallelisiert werden könnte, keine vollständige Funktion.--

void blake2s_core(
    __global const uchar* input, uint len,
    __global uchar* out) {

    uint m[16] = {0};

    for (int i = 0; i < 16 && (i * 4 + 3) < len; ++i) {

        m[i] = input[i*4 + 0] | (input[i*4 + 1] << 8) | (input[i*4 + 2] << 16) | (input[i*4 + 3] << 24);
    }

    //--Lokales Laden des Input-Blocks--

    for (int i = 0; i < 16; ++i)
        m[i] = *(__private uint*)&input_data[i * 4];

    uint v[16];
    for (int i = 0; i < 8; ++i) {
        v[i] = IV[i];
        v[i + 8] = IV[i];
    }

    v[12] ^= len;

    for (int r = 0; r < 10; ++r) {
        const
        __constant uchar* s = sigma[r];

        //--G_VEC (512-bit, Arbeitet auf ulong8!!!)--

        #define G_VEC(v_a, v_b, v_c, v_d, m_i, m_j) \
        do { \
            v_a = v_a + v_b + m_i; \
            v_d = rotr64_8(v_d ^ v_a, 32); \
            v_c = v_c + v_d; \
            v_b = rotr64_8(v_b ^ v_c, 24); \
            v_a = v_a + v_b + m_j; \
            v_d = rotr64_8(v_d ^ v_a, 16); \
            v_c = v_c + v_d; \
            v_b = rotr64_8(v_b ^ v_c, 63); \
    } while (0)

    G_VEC(v[0],v[4],v[8],v[12], m[s[0]], m[s[1]]);
    G_VEC(v[1],v[5],v[9],v[13], m[s[2]], m[s[3]]);
    G_VEC(v[2],v[6],v[10],v[14], m[s[4]], m[s[5]]);
    G_VEC(v[3],v[7],v[11],v[15], m[s[6]], m[s[7]]);
    G_VEC(v[0],v[5],v[10],v[15], m[s[8]], m[s[9]]);
    G_VEC(v[1],v[6],v[11],v[12], m[s[10]], m[s[11]]);
    G_VEC(v[2],v[7],v[8],v[13], m[s[12]], m[s[13]]);
    G_VEC(v[3],v[4],v[9],v[14], m[s[14]], m[s[15]]);
    #undef G
    }

    for (int i = 0; i < 8; ++i) {
        *(__private uint*)&out_data[i * 4] = v[i] ^ v[i + 8];
        out[i*4 + 0] = h & 0xFF;
        out[i*4 + 1] = (h >> 8) & 0xFF;
        out[i*4 + 2] = (h >> 16) & 0xFF;
        out[i*4 + 3] = (h >> 24) & 0xFF;
    }
}

//--Ein sehr simpler Kernel, der die Hashtabelle initialisiert. Jeder Thread setzt den Zähler für einen bestimmten Bereich der Tabelle auf 0.--

__kernel void kernel_init_ht(__global uchar *ht) {
    uint gid = get_global_id(0);
    *(__global uint *)(ht + tid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32) = 0;
}

uint ht_store(uint round, __global uchar *ht, uint i, ulong8 xi0, ulong8 xi1, ulong8 xi2, ulong8 xi3) {
    uint row;
    __global uchar *p;
    uint cnt;

    row = select(
        (uint)(((xi0 & 0xf0000) >> 0) | ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) | ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12)),
                 (uint)((xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4)),
                 !(round % 2)
    );

    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    p = ht + row * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32;
    cnt = atomic_inc((__global uint *)p);
    if (cnt >= ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9))
        return 1;
    p += cnt * 32 + (8 + ((round) / 2) * 4);

    *(__global uint *)(p - 4) = i;
    if (round == 0 || round == 1) {
        *(__global ulong8 *)(p + 0) = xi0;
        *(__global ulong8 *)(p + 8) = xi1;
        *(__global ulong8 *)(p + 16) = xi2;
    } else if (round == 2) {
        *(__global ulong8 *)(p + 0) = xi0;
        *(__global ulong8 *)(p + 8) = xi1;
        *(__global uint *)(p + 16) = xi2;
    } else if (round == 3 || round == 4) {
        *(__global ulong8 *)(p + 0) = xi0;
        *(__global ulong8 *)(p + 8) = xi1;
    } else if (round == 5) {
        *(__global ulong8 *)(p + 0) = xi0;
        *(__global uint *)(p + 8) = xi1;
    } else if (round == 6 || round == 7) {
        *(__global ulong8 *)(p + 0) = xi0;
    } else if (round == 8) {
        *(__global uint *)(p + 0) = xi0;
    }
    return 0;
}

//--ht_store reduziert pro Runde die zu speichernde Datenmenge.--
//--In Runde 0 werden alle Hashes komplett gespeichert.--
//--In späteren Runden werden nur die oberen Bits gespeichert (weil die unteren Bits schon kollidiert sind).--
//--Das ist der Schlüssel zum Memory-Efficiency-Trick von Equihash: weniger Speicherbedarf, aber weiterhin vollständige Kollisionssuche möglich--
//--"Andere Rundenlogik“ ist entscheidend. Das ist der Kern des Speicher-Spar-Tricks von Equihash.--
//--In jeder Runde wird die gespeicherte Datenmenge reduziert, da nur die höheren Bits relevant bleiben.--
//--Dies reduziert Speicherlast und zwingt die GPU, Kollisionen effizienter zu verarbeiten.“--
//--Offset in der Hashtabelle basierend auf dem Work-Item Index--
//--Speichert einen Hash-Wert "xi0, xi1, xi2, xi3" zusammen mit einer ID "xi_id" in--
//--der Hashtabelle ht an einer bestimmten Zeile "row_index" Funktionsweise:--
//--Berechnet den Zeiger "p" auf den Anfang der gewünschten Zeile in der Tabelle.--
//--Erhöht atomar den Zähler für die Anzahl der Einträge in dieser Zeile "atomic_inc". Dies verhindert Race Conditions, wenn mehrere Threads gleichzeitig--
//--in dieselbe Zeile schreiben wollen.--
//--Wenn die Zeile voll ist, wird 1 (Fehler) zurückgegeben.--
//--Andernfalls werden die Daten an der entsprechenden Position "p + cnt * 32" gespeichert.--
//--Wichtig: Die genaue Struktur und Größe der gespeicherten Daten hängt von der aktuellen Equihash-Runde "round" ab. In späteren Runden werden weniger Daten gespeichert nur die--
//--höherwertigen Bits), um Speicher zu sparen.--

__global uchar *p = ht + row_index * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32;
uint cnt = atomic_inc((__global uint *)p);
if (cnt >= ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9)) {
    return 1;
}
p = cnt * 32 + (8); //--8 ist die Größe des Blake-Zustands--


*(__global uint *)(p - 4) = xi_id;
*(__global ulong8 *)(p + 0) = xi0;
*(__global ulong8 *)(p + 8) = xi1;
*(__global ulong8 *)(p + 16) = xi2;

//--Andere Rundenlogik--


return 0;
}


//--Hilfsfunktion zur Suche nach Kollisionen innerhalb einer Workgroup--
//--Diese Funktion scheint ein Entwurf oder eine alternative Implementierung zu sein--
//--und wird nicht von den Hauptkerneln aufgerufen). Die Idee dahinter ist:--
//--Verwendet Shared Memory "__local", um Daten innerhalb einer Workgroup zwischen allen Threads zugänglich zu machen.--
//--Jeder Thread lädt seinen Teil der Hashtabelleneinträge in diesen schnellen, gemeinsamen Speicher.--
//--Nach einer Synchronisationsbarriere "barrier" durchsucht jeder Thread den gemeinsamen Datensatz nach Kollisionen (Werten, die in-- //--bestimmten Bit-Positionen übereinstimmen).--
//--Dies kann die Kollisionssuche innerhalb einer Workgroup erheblich beschleunigen.--
//--Dese Funktion wurde vermutlich als Optimierung für Intel GPUs mit großen Shared-Memory-Blöcken gedacht.--
//--Sie ersetzt die teure Suche über global memory durch eine--
//--kollaborative Suche im schnelleren "local memory".--
//--find_collisions versucht Kollisionen nicht im global memory (langsam), sondern im local memory (schnell, pro Workgroup).--
//--"local_ht[]" ist ein Hash-Table nur für Threads dieser Workgroup.--
//--Jeder Thread trägt seine Werte ein und sucht parallel nach gleichen Präfixen "Kollisionen".--
//--Das ist eine experimentelle Optimierung, die nicht in allen Treibern stabil läuft (Intel GPUs mögen sowas, Nvidia weniger).--

void find_collisions(__global uchar *ht_src, __global uchar *ht_dst, __global sols_t *sols, uint round) {
    uint gid = get_global_id(0);
    uint tid = get_local_id(0);
    uint group_id = get_group_id(0);

    __global uchar *p = ht_src + gid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32;
    uint cnt = *(__global uint *)p;
    cnt = min(cnt, (uint)((1 << (((200 / (9 + 1)) + 1) - 20)) * 9));

    //--Geteilter Speicherbereich für Arbeitsgruppenweite Kollisionserkennung--


    __local ulong8 shared_data[WORKGROUP_SIZE * MAX_COLLISIONS];

    //--Jeder Strang(Thread) lädt seinen Hash-Wert in den lokalen Speicher--


    if (tid < cnt) {
        shared_data[tid] = *(__global ulong8 *)(ht_src + (gid * ENTRY_SIZE * cnt) + tid * ENTRY_SIZE + (8 + ((round-1) / 2) * 4));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //--Kollisionssuche innerhalb der Arbeitsgruppe--

    uint coll_count = 0;
    for (uint i = tid; i < cnt; i += get_sub_group_size()) {
        ulong8 val_a = shared_data[i];
        for (uint j = i + 1; j < cnt; j++) {
            ulong8 val_b = shared_data[j];

            if (val_a.x == val_b.x) { //--Beispiel: Kollision auf dem ersten 64-bit Wort--


                //--Kollision gefunden, verarbeite sie--
                //--Subgroup/Untergruppen-Funktionen evtl Effizienter--

            }
        }
    }
}
row = select(
    (uint)(((xi0 & 0xf0000) >> 0) | ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) | ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12)),
             (uint)((xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4)),
             !(round % 2)
);

xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
p = ht + row * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32;
cnt = atomic_inc((__global uint *)p); //--Hier ATOMIC_INC--
if (cnt >= ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9))
    return 1;
p = cnt * 32 + (8 + ((round) / 2) * 4);

*(__global uint *)(p - 4) = i;
if (round == 0 || round == 1) {
    *(__global ulong8 *)(p + 0) = xi0;
    *(__global ulong8 *)(p + 8) = xi1;
    *(__global ulong8 *)(p + 16) = xi2;
} else if (round == 2) {
    *(__global ulong8 *)(p + 0) = xi0;
    *(__global ulong8 *)(p + 8) = xi1;
    *(__global uint *)(p + 16) = xi2;
} else if (round == 3 || round == 4) {
    *(__global ulong8 *)(p + 0) = xi0;
    *(__global ulong8 *)(p + 8) = xi1;
} else if (round == 5) {
    *(__global ulong8 *)(p + 0) = xi0;
    *(__global uint *)(p + 8) = xi1;
} else if (round == 6 || round == 7) {
    *(__global ulong8 *)(p + 0) = xi0;
} else if (round == 8) {
    *(__global uint *)(p + 0) = xi0;
}
return 0;
}


//--Der Hauptkernel für die erste Runde (Round 0). Seine Aufgabe ist es, die initiale Menge von Hash-Werten zu generieren.--
//--Eingabe: Ein initialer Blake2b-Zustand (blake_state), der wahrscheinlich--
//--aus einem Block-Header abgeleitet ist.--
//--Verarbeitung:--
//--Jeder Thread berechnet eine Reihe von unterschiedlichen Hash-Werten, indem er einen Zähler--
//--"input" mit dem Blake2b-Zustand-- mischt--
//--Hier durch Addition von word1 ="ulong8"-"input << 32" simuliert.--
//--Für jeden Wert wird eine vereinfachte Version der Blake2b-Runden durchlaufen--
//--Dargestellt durch die G_VEC-Makros und die v1 = v1.yzwx-Permutationen, --
//--die die diagonale Mischung in-- Blake2b nachahmen).--
//--Der finale Hash-Wert h[0..7] wird durch XOR des ursprünglichen Zustands mit dem aktuellen Zustand und dem IV berechnet.--
//--Ausgabe: Die generierten Hash-Werte werden sofort mittels ht_store in der Hashtabelle ht abgelegt. Pro input-Wert werden --
//--oft zwei Einträge gespeichert, wobei der zweite ein um 8 Bit geshifteter Wert des ersten ist, um die Kollisionssuche zu starten.--

__kernel __zhash_144_5__((reqd_work_group_size(64,1,1)))

//--erzwingt Workgroups von 64 Threads (in x-Richtung), wichtig für die Parallelisierung.--

void zhash_144_5(__global ulong8 *blake_state, __global uchar *ht, __global uint *debug) {
    uint gid = get_global_id(0); //--die globale ID des Threads.--
    uint inputs_per_thread = (1 << (200 / 10)) / get_global_size(0); //--"10 = 9+1" wie viele Inputs ein einzelner Thread bearbeiten soll.--

    //--Wird durch die Gesamtzahl der Threads geteilt.--


    uint input = tid * inputs_per_thread;
    uint input_end = (tid + 1) * inputs_per_thread;
    uint dropped = 0;

    //--Vier Vektoren "ulong8" bilden den Startzustand
    //--"v2_init" und "v3_init" kommen aus den IV-Konstanten von BLAKE2b.


    ulong8 v0_init = (ulong8)(blake_state[0], blake_state[1], blake_state[2], blake_state[3]);
    ulong8 v1_init = (ulong8)(blake_state[4], blake_state[5], blake_state[6], blake_state[7]);
    ulong8 v2_init = (ulong8)(BLAKE2B_IV[0], BLAKE2B_IV[1], BLAKE2B_IV[2], BLAKE2B_IV[3]);
    ulong8 v3_init = (ulong8)(BLAKE2B_IV[4], BLAKE2B_IV[5], BLAKE2B_IV[6], BLAKE2B_IV[7]);


    //--Ergibt 144. Der aktuelle Wert von "v3_init.x" wird bitweise mit "144"--
    //--(in Binär geschrieben) XOR-verknüpft. Ergebnis ersetzt "v3_init.x".--

    v3_init.x ^= 140 + 4;

    //--Ergibt 144. Der aktuelle Wert von "v3_init.x" wird bitweise mit "144" (in Binär geschrieben) XOR-verknüpft. Ergebnis ersetzt "v3_init.x".--

    v3_init.z ^= -1;

    //-- in Binärdarstellung ist (im Zweierkomplement) eine Folge von lauter 1-Bits. Ein XOR mit lauter Einsen kehrt alle Bits um
    //--Bitweise Invertierung. Ergebnis: "v3_init.z" wird umgedreht (Bitwise NOT).--
    //--x bekommt durch das XOR eine Art „geheimes“ Verrechnen mit 144.
    //--z wird komplett invertiert.

    while (input < input_end) {
        ulong8 word1 = (ulong8)input << 32;

        ulong8 v0 = v0_init;
        ulong8 v1 = v1_init;
        ulong8 v2 = v2_init;
        ulong8 v3 = v3_init;

        //--Runden 1 bis 12--
        //--Hier laufen die G-Funktionen von BLAKE2b, mehrfach mit Rotationen der Vektoren--
        //--Das entspricht den 12 Standardrunden, nur hier auf 9 reduziert, wieder auf 12 aufbauen--

        v0.x += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.x += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.z += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.y += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.z += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.z += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.w += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.x += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;
        v0.x += word1; G_VEC(v0,v1,v2,v3); v1 = v1.yzwx; v2 = v2.zwxy; v3 = v3.wxyz; G_VEC(v0,v1,v2,v3); v1 = v1.wxyz; v2 = v2.zwxy; v3 = v3.yzwx;

        //--Final XOR--(In Hauptprogramm eintragen)--

        //--Damit wird ein Hashblock gebildet.--
        //--Gemischte Initialwerten (v2_init, v3_init) und dem Originalstate (blake_state) bereiningen. Original Blake2b Einhalten.--

        ulong8 h[8];
        h[0] = blake_state[0] ^ v0.x ^ v2_init.x;
        h[1] = blake_state[1] ^ v0.y ^ v2_init.y;
        h[2] = blake_state[2] ^ v0.z ^ v2_init.z;
        h[3] = blake_state[3] ^ v0.w ^ v2_init.w;
        h[4] = blake_state[4] ^ v1.x ^ v3_init.x;
        h[5] = blake_state[5] ^ v1.y ^ v3_init.y;
        h[6] = blake_state[6] ^ v1.z ^ v3_init.z;
        h[7] = blake_state[7] ^ v1.w ^ v3_init.w;

         //--Ergebnisse werden in ht geschrieben (HashTable?). 2-Slot Speicherung pro Input. Ändern.--

        dropped += ht_store(0, ht, input * 2, h[0], h[1], h[2], h[3]);
        dropped += ht_store(0, ht, input * 2 + 1,
                            (h[3] >> 8) | (h[4] << (64 - 8)),
                            (h[4] >> 8) | (h[5] << (64 - 8)),
                            (h[5] >> 8) | (h[6] << (64 - 8)),
                            (h[6] >> 8));
        input++;
        if (gid == 0) {
            debug[0] = inputs_per_thread;
            debug[1] = input_end;
        }
    }
}

//--Dies ist eine ultimativ oft Bearbeitet Version, für Intel ARC Grafikkarten, bitte Löschen Sie diese niemals einfach so!.--
//--Die Funktion enthält außerdem einen integrierten subgroup_reduce,--
//--der eigentlich nach außen müsste. Dieser Teil deutet darauf hin, dass verschiedene Varianten getestet wurden--
//--(Workgroup-basiert vs. Subgroup-basiert).“--
//--Diese Funktion kombiniert zwei Hashwerte (val1 und val2) durch XOR.--
//--Zusätzlich ist ein Subgroup-Reduce-Mechanismus eingebaut:--
//--Damit kann die GPU Threads innerhalb einer Subgroup (z. B. 8 oder 16 Threads)--
//--direkt synchronisieren, ohne globalen Speicher zu nutzen.--
//--Das deutet darauf hin, dass Equihash-Kollisionen in manchen Varianten--
//--direkt innerhalb einer Subgroup verarbeitet werden, ohne zurück ins große Hash-Table zu schreiben.--


uint xor_and_store(uint round, __global uchar *ht_dst, uint row, uint slot_a, uint slot_b, __global ulong8 *a, __global ulong8 *b)
{
    ulong8 xi0 = 0UL, xi1 = 0UL, xi2 = 0UL;

    ulong8 subgroup_reduce(ulong8 val) {
        for (int i = 1; i < get_sub_group_size(); i <<= 1)
            val += sub_group_shuffle_xor(val, i);
        return val;
    }

     //--Vektorisierte Verarbeitung mit ulong8 für maximale GPU-Effizienz--

    if (round == 1 || round == 2) {
        xi0 = a[0] ^ b[0];  //--Volles ulong8 XOR--
        xi1 = a[1] ^ b[1];  //--Volles ulong8 XOR--
        xi2 = a[2] ^ b[2];  //--Volles ulong8 XOR--
        if (round == 2) {

            //--Vektorisierte Shifts für alle 8 Elemente--

            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8) | (xi2 << (64 - 8));
            xi2 = (xi2 >> 8);
        }

    } else if (round == 3) {
        xi0 = a[0] ^ b[0];
        xi1 = a[1] ^ b[1];

         //--Für 32-bit Zugriffe: konvertiere zu uint8 für vektorisierten Zugriff--

        xi2 = convert_ulong8(convert_uint8(a[2]) ^ convert_uint8(b[2]));

        //--xi2 = *(__global uint *)a ^ *(__global uint *)b; /* FIXME: potential unaligned 32-bit read */--


    } else if (round == 4 || round == 5) {
        xi0 = *a++ ^ *b++;
        xi1 = *a ^ *b;
        xi2 = 0;
        if (round == 4) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8);
        }

    } else if (round == 6) {
        xi0 = *a++ ^ *b++;

        //--Für 32-bit Zugriffe: konvertiere zu uint8 für vektorisierten Zugriff--

        xi1 = xi2 = convert_ulong8(convert_uint8(a[2]) ^ convert_uint8(b[2]));
        xi2 = 0;
        if (round == 6) {
            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
            xi1 = (xi1 >> 8);
        }
    } else if (round == 7 || round == 8) {
        xi0 = a[0] ^ b[0];
        xi1 = (ulong8)0;
        xi2 = (ulong8)0;
        if (round == 8) {
            xi0 = (xi0 >> 8);
        }
    }

     //--Überprüfe ob alle Elemente in xi0 und xi1 Null sind--

    if (!xi0 && !xi1)
        return 0;

    //--ID aus row und slots codieren--

    uint id = (row << 12) | ((slot_b & 0x3f) << 6) | (slot_a & 0x3f);

    //--ulong8 Werte an ht_store übergeben--

    return ht_store(round, ht_dst, ((row << 12)) | ((slot_b & 0x3f) << 6) | (slot_a & 0x3f)), id, xi0, xi1, xi2, (ulong8)0);
}


//--In jeder Runde werden Hashes kombiniert und auf Kollisionen geprüft.--
//--Input: Hashes aus vorheriger Runde (hash_table + ht_prefix).--
//--Verarbeitung: find_collisions oder direkte xor_and_store--.
//--Output: neue Hashes für die nächste Runde.--
//--Wichtig: Nach der letzten Runde (n/k Runden) entstehen noch keine fertigen Lösungen.--
//--Erst die Kombination aller Pfade (in solve_equihash) erzeugt die 200-Bit gültigen Proofs.--
//--Erklärung: Diese Funktion ist das Herzstück der Equihash-Runden 1 bis 8. Sie sucht nach Paaren von Einträgen,--
//--die in den relevanten Bit-Positionen übereinstimmen (Kollisionen). Für jedes gefundene Paar ruft sie--
//--xor_and_store auf, um die Teillösung zu kombinieren und in der Zieltabelle für die nächste Runde zu speichern--
//--Dieser TEil produziert noch keine Lösung. Die tatsächliche Lösung (200-Bit Wert)
//--entsteht erst nach Round 8 durch Rekombination aller Pfade.--

void equihash_round(uint round, __global uchar *ht_src, __global uchar *ht_dst, __global uint *debug)
{
    uint tid = get_global_id(0);
    __global uchar *p;
    uint cnt;
    uchar first_words[((1 << (((200 / (9 + 1)) + 1) - 20)) * 9)];
    uchar mask;
    uint i, j;

    ushort collisions[((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 3];
    uint nr_coll = 0;
    uint n;
    uint dropped_coll, dropped_stor;
    __global ulong8 *a, *b;
    uint xi_offset;
    xi_offset = (8 + ((round - 1) / 2) * 4);

    //--Verarbeitung von Kollisionslisten in einer Hash-Tabelle, wobei jeder Thread einen eigenen Bucket bearbeitet.--

    mask = 0;
    p = (ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32);

    //--ht_src: Basisadresse der Hash-Tabelle im Speicher--
    //--tid: Thread-ID (welcher Thread gerade arbeitet) 9 + 1 = 10--
    //--200 / 10 = 20--
    //--20 + 1 = 21--
    //--21 - 20 = 1--
    //--1 << 1 = 2 (Bitshift nach links um 1 Position = Verdopplung)--
    //--2 * 9 = 18--
    //--18 * 32 = 576--
    //--Vereinfacht: p = ht_src + tid * 576--
    //--Berechne die Speicheradresse für diesen Thread. Jeder Thread bekommt einen eigenen
    //--576-Byte-Bereich in der Hash-Tabelle zugewiesen.--

    cnt = *(__global uint *)p;

    //--*(...)p: Dereferenzierung - liest den Wert an Adresse p- Liest den Zählerwert aus dem Speicher an der berechneten Position "p".--

    cnt = min(cnt, (uint)((1 << (((200 / (9 + 1)) + 1) - 20)) * 9));

    //--Berechnung des Maximums--
    //--Gleiche Berechnung wie oben: 2 * 9 = 18--
    //--Zweck: Begrenzt den gelesenen Wert auf maximal 1--

    p += xi_offset;

     //--Zweck: Verschiebt den Pointer um eine bestimmte Offset-Distanz weiter--

    for (i = 0; i < cnt; i++, p += 32)
        first_words[i] = *(__global uchar *)p;
    nr_coll = 0;
    dropped_coll = 0;
    for (i = 0; i < cnt; i++)
        for (j = i + 1; j < cnt; j++)
            if ((first_words[i] & mask) == (first_words[j] & mask)) {
                if (nr_coll >= sizeof (collisions) / sizeof (*collisions))
                    dropped_coll++;
                else
                    collisions[nr_coll++] = ((ushort)j << 8) | ((ushort)i & 0xff);
            }

    dropped_stor = 0;
    for (n = 0; n < nr_coll; n++) {
        i = collisions[n] & 0xff;
        j = collisions[n] >> 8;
        a = (__global ulong8 *)(ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32 + i * 32 + xi_offset);
        b = (__global ulong8 *)(ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32 + j * 32 + xi_offset);
        dropped_stor += xor_and_store(round, ht_dst, tid, i, j, a, b);
    }
    if (round < 8)
        *(__global uint *)(ht_src + tid * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32) = 0;
}

//--Equihash Round Kernels--

//--Equihash Round Kernels--
//--Eine Reihe von Kerneln für die Equihash-Runden 1 bis 8. Jeder Kernel führt equihash_round() für seine jeweilige Runde aus.--
//--equihash_round (round, ht_src, ht_dst, debug):--
//--Lade Kandidaten: Jeder Thread liest die Einträge "seiner" Zeile aus der Quelltabelle ht_src der vorherigen Runde.--
//--Suche Kollisionen: Innerhalb dieser Zeile sucht er nach Paaren von Einträgen,--
//--deren Hash-Werte in den für die aktuelle Runde relevanten Bit-Positionen übereinstimmen (die mask wird--
//--verwendet, um die nicht relevanten Bits auszublenden).--
//--Kombiniere und speichere: Für jedes gefundene Paar wird xor_and_store aufgerufen, um die Teillösung zu kombinieren--
//--und in der Zieltabelle ht_dst für die nächste Runde zu speichern.--
//--In Runde 8 (kernel_round8) wird auch der Lösungszähler sols->nr zurückgesetzt.--

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round1(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(1, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round2(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(2, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round3(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(3, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round4(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(4, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round5(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(5, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round6(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(6, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void kernel_round7(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug) { equihash_round(7, ht_src, ht_dst, debug); }
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))

void kernel_round8(__global uchar *ht_src, __global uchar *ht_dst, __global uint *debug, __global sols_t *sols)
{
    uint tid = get_global_id(0);
    equihash_round(8, ht_src, ht_dst, debug);
    if (!tid)
        sols->nr = sols->likely_invalids = 0;
}

//--Rekonstruiert die Referenz die "i_id" eines gespeicherten Eintrags.--
//--Diese ID kodiert die Herkunft des Eintrags (welche zwei Einträge der vorherigen Runde kombiniert wurden).--

uint expand_ref(__global uchar *ht, uint xi_offset, uint row, uint slot)
{
    return *(__global uint *)(ht + row * ((1 << (((200 / (9 + 1)) + 1) - 20)) * 9) * 32 + slot * 32 + xi_offset - 4);
}

//--Rekonstruiert einen gesamten Lösungsbaum. Diese Funktion wird rekursiv für jede Runde aufgerufen.--
//--Sie nimmt ein Array von Referenzen (ins) aus der aktuellen Runde.--
//--Für jede Referenz schaut sie in der Hashtabelle der vorherigen Runde nach und ersetzt die Referenz durch die beiden Referenzen,--
//--aus denen sie in dieser vorherigen Runde erzeugt wurde.--
//--Dies verdoppelt die Anzahl der Referenzen in jedem Schritt, bis die ursprünglichen 200 Indizes aus der ersten Runde rekonstruiert sind.--

void expand_refs(__global uint *ins, uint nr_inputs, __global uchar **htabs, uint round)
{
    __global uchar *ht = htabs[round % 2];
    uint i = nr_inputs - 1;
    uint j = nr_inputs * 2 - 1;
    uint xi_offset = (8 + ((round) / 2) * 4);
    do {
        ins[j] = expand_ref(ht, xi_offset, (ins[i] >> 12), ((ins[i] >> 6) & 0x3f));
        ins[j - 1] = expand_ref(ht, xi_offset, (ins[i] >> 12), (ins[i] & 0x3f));
        if (!i)
            break ;
        i--;
        j -= 2;
    } while (1);
}

//--Wird aufgerufen, wenn in der finalen Runde eine Kollision gefunden wurde.--
//--Reserviert einen Slot in der Lösungsliste (sols).--
//--Speichert die beiden finalen Referenzen.--
//--Ruft expand_refs für alle vorherigen Runden auf, um den gesamten Lösungsvektor (die 200 Indizes) zu rekonstruieren.--
//--Markiert die Lösung als gültig (valid = 1).--

void potential_sol(__global uchar **htabs, __global sols_t *sols, uint ref0, uint ref1)
{
    uint sol_i;
    uint nr_values;
    sol_i = atomic_inc(&sols->nr);
    if (sol_i >= 2000)
        return ;
    sols->valid[sol_i] = 0;
    nr_values = 0;
    sols->values[sol_i][nr_values++] = ref0;
    sols->values[sol_i][nr_values++] = ref1;
    uint round = 9 - 1;
    do {
        round--;
        expand_refs(&(sols->values[sol_i][0]), nr_values, htabs, round);
        nr_values *= 2;
    } while (round > 0);
    sols->valid[sol_i] = 1;
}

//--Hauptkernel für die finale Kollisionssuche Er kombiniert mehrere Konzepte:--
//--Laden in Shared Memory: Die relevanten Teile der globalen Hashtabelle werden--
//--in den schnellen Shared Memory der Workgroup geladen.--
//--Lokale Kollisionssuche: Alle Threads der Workgroup durchsuchen--
//--gemeinsam die im Shared Memory liegenden Daten nach Kollisionen.--
//--Dies ist effizienter, als wenn jeder Thread nur auf seinen eigenen globalen Speicherbereich zugreift.--
//--Lösungsausgabe: Gefundene Kollisionspaare werden zur endgültigen Verarbeitung an potential_sol übergeben.--
//--Grundkern!!! Wichtig Beibehaltung--

__kernel void zhash_144_5(
    __global ulong8 *blake2s_state,
    __global uchar *ht0,
    __global uchar *ht1,
    __global sols_t *sols,
    __local uint64x8 *shared_cache)  //--Lokaler Speicher für Vektordaten--
{

    //--Jeder Thread bearbeitet seinen eigenen Datenblock--

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);

    //--Globale Tabellen anlegen--

    __global uchar *htabs[2] = { ht0, ht1 };
    uint ht_i = (9 - 1) % 2; //--WERT vom Tisch nehmen als Gültig--

    //--Lokale Puffer Vektorisiert--

    uint64x8 local_block[INPUT_SIZE / sizeof(uint64x8)];
    uint64x8 local_collisions[MAX_COLLISIONS];
    uint coll = 0;
    const uint mask = 0x00ffffffU;

    //--Schritt 1: Daten aus globalem Speicher in lokale Kopie laden--

    __global uint64x8* global_block = (
    __global uint64x8*)(
    htabs[ht_i] + gid * (MAX_ENTRIES * ENTRY_SIZE));

    uint cnt = vload8(0, (__global uint*)global_block).s0;  //--Erstes Element des Vektors--
    cnt = min(cnt, (uint)((1 << (((200 / (9 + 1)) + 1) - 20)) * 9));

    uint xi_offset = (8 + ((9 - 1) / 2) * 4);

    //--Daten in lokale Arbeitgruppe laden--

    for (uint i = lid; i < cnt * (ENTRY_SIZE / sizeof(uint64x8)); i += WORKGROUP_SIZE) {
        shared_cache[i] = global_block[i + (8 + ((9 - 1) / 2) * 4) / sizeof(uint64x8)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);  //--Synchronisiere vor Zugriff--

    //--Schritt 2: Kollisionen im lokalen Puffer suchen--

    for (uint i = 0; i < cnt; i++) {
        uint64x8 val_a = shared_cache[i] & mask;  //--Maskierung vektorisiert--

            //--Werte aus dem lokalen Block holen und maskieren--

            for (uint j = i + 1; j < cnt; j++) {
                uint64x8 val_b = shared_cache[j] & mask;
                if (any(val_a == val_b)) {  // SIMD-Vergleich
                    if (coll < MAX_COLLISIONS) {
                        local_collisions[coll++] = (uint64x8)(i, j, 0, 0, 0, 0, 0, 0); //--PaketMarkierung--
                } else {

                    //--Puffer voll -> globales Zählen als "wahrscheinlich ungültig"--

                    atomic_inc(&sols->likely_invalids);
                }
            }
        }
    }

    //--Wenn keine Kollision gefunden wurde, fertig--

    if (!coll) return;

    //--Schritt 3: Gefundene Kollisionen verarbeiten--

    barrier(CLK_LOCAL_MEM_FENCE);  //--Synchronisiere vor globalem Zugriff--
    for (uint k = 0; k < coll; k++) {
        potential_sol(htabs, sols, local_collisions[k].s0, local_collisions[k].s1);
    }
}

//--Zusammenfassung ABLAUF--
//--1--Initialisierung: Die Hashtabelle wird geleert "(kernel_init_ht)".--
//--2--Runde 0: Der Kernel "zhash_144_5" (der erste) füllt die Tabelle mit "2^(200/10) = 2^20 (≈1 Million)" initialen Blake2b-Hashwerten.--
//--3--Runden 1-8: Die Kernel "kernel_round1" bis "kernel_round8"" werden nacheinander ausgeführt. In jeder Runde i:--
//--3.1--Werden Kollisionen in den unteren Bits der Einträge aus Runde "i-1" gesucht.--
//--3.2--Die gefundenen Paare werden "geXORt" und die Ergebnisse (nach einem Rechts-Shift) in die Tabelle für Runde "i" geschrieben.--
//--3.3--Die Größe der Problemstellung halbiert sich mit jeder Runde effektiv.--
//--4--Finale Prüfung: In Runde 8 sucht der Kernel nach einer Kollision über die verbleibenden Bits. Wird eine gefunden,--
//--ist eine potenzielle Gesamtlösung gefunden.--
//--5--Rekonstruktion: Die Funktion "potential_sol" rekonstruiert aus den beiden finalen Referenzen den gesamten Pfad--
//--5--durch alle vorherigen Runden zurück zu den ursprünglichen 200 Indizes.--
//--Diese Indizesequenz ist der gültige Proof-of-Work.--

//--Diese Anmerkungen bleiben im Kern, bis alles Fertig ist und entsprechend Nachgearbeitet werden kann bei Bedarf--

//--1--Initialisierung
//--Es gibt feste Startwerte (IVs).
//--Daraus baust du deinen internen Zustand (meist 8 oder 16 Wörter).

//--2--Daten in Blöcke packen
//--Blake2b nimmt 128-Byte Blöcke.
//--Blake2s nimmt 64-Byte Blöcke.

//--3--Mix-Funktion (das Herzstück)
//--Jeder Block wird durch eine G-Funktion „durchgemischt“.
//--Dabei wird rotiert, addiert, XORt.
//--Genau das sorgt für die „Kryptomagie“.

//--4--Finalisierung
//--Am Ende werden die Zustände zusammengefügt.
//--Ergebnis ist dein Hash (z. B. 32 oder 64 Byte).
