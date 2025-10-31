#!/bin/bash

export GPU_MAX_HEAP_SIZE=100 //--
export GPU_MAX_USE_SYNC_OBJECTS=1 //--
export GPU_SINGLE_ALLOC_PERCENT=100 //--
export GPU_MAX_ALLOC_PERCENT=100 //--
export GPU_MAX_SINGLE_ALLOC_PERCENT=100 //--
export GPU_ENABLE_LARGE_ALLOCATION=100 //--
export GPU_MAX_WORKGROUP_SIZE=64 //--

./xbtgpuarc \ //--Startet das XBTGPUARC Mining Programm.--
    --platform 1 \ //--Wählt die Plattform aus, auf wlecher Gemined werden soll.--
    --device 0 \ //-Wählt die genaue Recheneinheit-in Form der Intel ARC Grafikkarte mit dem DG2 Chip aus.--
    --algo zhash_144_5 \ //--Wählt den zu minenden Algoritmus aus--
    --pool solo-btg.2miners.com \ //--Wählt den Pool oder den Server aus, um mit dem Netzwerk zu Kommunizieren.--
    --port 4040 \ //--Wählt den PoolPort aus.--
    --wallet Gb4V4a9Jk3p8aH6jkW3Aq3sq8rQCuJQ6S8 \ //--Hier fügen Sie ihre eigene Mining Adressen ein!--
                                                 //--Sehr Wichtig, da ansonsten die Belohnung an meine Person geht!--
    --worker A730m \ //--Hier sehen Sie die genaue Bezeichnung des ausgewählten ARC Computer Chips.--
    --password x \ //--Hier können Sie den Wert bei "x" Behalten. Es würde wohl kaum jemand ihre Wallet mit Geld durch seine Arbeit bei ihnen füllen wollen.--
    --intensity 256 //--HIer kann man Einstellen, wie Stark die Grafikkarte arbeiten soll!--
                    //--Ein Mittelhoher Wert istin der Regel zu Präferieren, genaue Details zur ARC GPU Architektur stehen noch aus!--
