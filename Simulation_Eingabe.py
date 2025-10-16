from Simulation_Lernkurve import Lernkurve
from Simulation_Vergessenskurve import Vergessenskurve




'''Allgemeine-Parameter''' 

Betrachtungszeitraum = 9 # inkl. Wochenende angeben
Abwesenheit_1 = 3600*16 # Abwesenheit zwischen den Schichten = 16h
Abwesenheit_2 = 3600*72 # Abwesenheit am Wochenende = 72h sind das nicht eigentlich 64h?

'''Vergessenskurvenparameter'''
ZI = 3600 # hat einfluss auf die Vergessenskurve (je größer das Zeitintervall [ZI] gewählt desto langsamer wird vergessen)

'''Job-Parameter'''
##########
# EINGABE ERFORDERLICH:
jobs = ["J1", "J2", "J3"]  # Jobs und ihre Zuordnung zu Tätigkeiten
taetigkeiten_liste = ["T1", "T2", "T3", "T4", "T5"]
#taetigkeiten_liste = ["T1", "T2", "T3", "T4", "T5", "T6"]
jobs_zuordnung = {
    "J1": ["T1", "T2"],  # J1 erfordert T1 und T3
    "J2": ["T2", "T3"],  # J2 erfordert T2 und T3
    "J3": ["T3", "T4", "T5"]


}

"""Übungsfaktor (Routineaufbau)"""
uebungsfaktor_parameter = {
    "m_min": 1.0,
    "m_max": 5.0,
    "alpha": 1.0,
    "n_norm": 300.0,
    "q_min_stunden": 1.0,
}

# Komplexität je Tätigkeit (p-Wert, 1 = sehr einfach, 5 = sehr komplex)
taetigkeit_komplexitaet = {
    "T1": 3.0,
    "T2": 4.0,
    "T3": 3.0,
    "T4": 2.0,
    "T5": 5.0,
}

# Geplante Änderungen der Tätigkeitsprofile während der Simulation
# Ab dem jeweiligen Starttag gelten die hier definierten neuen Zuordnungen
# für die angegebenen Jobs. Falls neue Tätigkeiten genutzt werden sollen,
# müssen diese zuvor in "taetigkeiten_liste" ergänzt werden.
jobprofil_aenderungen = [
    # {
    #     "start_tag": 4,
    #     "beschreibung": "Produktwechsel in Linie J1",  # optional
    #     "jobs": {
    #         "J1": ["T1", "T4", "T5"],
    #         "J2": ["T2"],
    #     },
    # },
    # {
    #     "start_tag": 6,
    #     "beschreibung": "Produktwechsel in Linie J1",  # zurück zum ursprünglichen Tätigkeitsprofil
    #     "jobs": {
    #         "J1": ["T1", "T2"],  
    #         "J2": ["T2", "T3"]
    #     },
    # },
]

# Ziele, die einen vorzeitigen Stopp der Simulation auslösen können
# (leer lassen, wenn keine automatischen Stopps gewünscht sind)
simulationsziele = {
    # "fehlerquote": 0.063,  # Durchschnittliche Fehlerquote aller Mitarbeitenden
    # "produktivitaet": 250,  # Gesamtoutput (Output_gut) pro Tag über alle Jobs
    # "kompetenzziel": {"durchschnitt": 3.0, "varianz": 0.20},
    "max_tage": 20,  # Maximale Anzahl an Tagen, falls Ziele nicht erreicht werden
}


# Geplante Produktionsstörungen (Stillstände je Tag/Runde/Job)
produktionsstoerungen = [
    # {
    #     "start_tag": 3,
    #     "dauer": 1,
    #     "runden": [2],
    #     "jobs": ["J1"],
    #     "beschreibung": "Maschinenausfall an Linie J1",
    # },
    # {
    #     "start_tag": 2,
    #     "dauer": 1,
    #     "runden": [2, 3],
    #     "jobs": ["J2"],
    #     "beschreibung": "Maschinenausfall an Linie J1",
    # },
    # {
    #     "start_tag": 4,
    #     "dauer": 1,
    #     "runden": "alle",
    #     "jobs": "alle",
    #     "beschreibung": "Werkweiter Stromausfall",
    # },
]

##########

'''Mitarbeitenden-Parameter'''
##########
# Liste der aktiven Mitarbeitenden in der Simulation
mitarbeitende = ["MA1", "MA2", "MA3"]

# Standardisierte Planung von Personalrisiken (Ausfälle & Fluktuationen)
personalrisiken = {
    # "ausfaelle": [
    #     {"mitarbeiter": "MA1", "start_tag": 2, "dauer": 2},
    #     #{"mitarbeiter": "MA2", "start_tag": 5, "dauer": 1},
    # ],
    # "fluktuation": [
    #     {"mitarbeiter": "MA2", "start_tag": 3},
    #     #{"mitarbeiter": "MA1", "start_tag": 8},
    # ],
}

# Standardparameter für Ersatzkräfte (EMA)
ersatz_standardparameter = {
    "t_initiale_AFZ": {
        "T1": 123.333,
        "T2": 117.0,
        "T3": 122.0,
        "T4": 100.0,
        "T5": 110.0,
    },
    "vor_AFA": {taetigkeit: 1 for taetigkeit in taetigkeiten_liste},
    "GW": {
        "T1": 0.3,
        "T2": 0.4,
        "T3": 0.2,
        "T4": 0.25,
        "T5": 0.25,
    },
    "LF": {
        "T1": -0.45,
        "T2": -0.29,
        "T3": -0.243333,
        "T4": -0.25,
        "T5": -0.25,
    },
    "VF": {
        "T1": -0.045,
        "T2": -0.029,
        "T3": -0.0243333,
        "T4": -0.025,
        "T5": -0.025,
    },
}


# Lernfaktoren für Mitarbeitende (pro Tätigkeit)
LF_MA = {
    "MA1": {"T1": -0.50, "T2": -0.32, "T3": -0.25, "T4": -0.25, "T5": -0.25},
    "MA2": {"T1": -0.45, "T2": -0.30, "T3": -0.28, "T4": -0.25, "T5": -0.25},
    "MA3": {"T1": -0.4, "T2": -0.25, "T3": -0.2, "T4": -0.25, "T5": -0.25},
}

# Vergessensfaktoren werden aus den Lernfaktoren abgeleitet
teiler = 10
VF_MA = {
    ma: {taetigkeit: wert / teiler for taetigkeit, wert in werte.items()}
    for ma, werte in LF_MA.items()
}

# Grenzwerte für Mitarbeitende sind maschinell bedingt und daher identisch
GW_MA = {
    "MA1": {"T1": 0.3, "T2": 0.4, "T3": 0.2, "T4": 0.25, "T5": 0.25},
    "MA2": {"T1": 0.3, "T2": 0.4, "T3": 0.2, "T4": 0.25, "T5": 0.25},
    "MA3": {"T1": 0.3, "T2": 0.4, "T3": 0.2, "T4": 0.25, "T5": 0.25},
}

# Initiale Ausführungszeiten (AFZ) je Mitarbeitendem
t_initiale_AFZ = {
    "MA1": {"T1": 120, "T2": 115, "T3": 120, "T4": 100, "T5": 100},
    "MA2": {"T1": 125, "T2": 118, "T3": 123, "T4": 100, "T5": 110},
    "MA3": {"T1": 125, "T2": 118, "T3": 123, "T4": 100, "T5": 120},
}

# Vorerfahrung (AFA) je Mitarbeitendem, Startwert für die Ausführungsanzahl
vor_AFA = {
    "MA1": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1},
    "MA2": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1},
    "MA3": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1},
}

# Standardisierte Fehlerquoten pro Kompetenzstufe (1 = geringste Kompetenz)
standard_fehlerquote = {
    1: 0.18,
    2: 0.12,
    3: 0.07,
    4: 0.04,
    5: 0.02,
}

# Fehlerquoten können pro Mitarbeitendem und Tätigkeit angepasst werden
fehlerquote_parameter = {
    ma: {taetigkeit: dict(standard_fehlerquote) for taetigkeit in taetigkeiten_liste}
    for ma in mitarbeitende
}

# Geplanter Output pro Ausführung einer Tätigkeit (Stückzahl)
output_pro_ausfuehrung = {taetigkeit: 1 for taetigkeit in taetigkeiten_liste}

# Arbeitsrunden und Job-Zuordnung
arbeitsrunden = [
    {
        "name": "Runde 1",
        "arbeitszeit": 3 * 3600,  # 3 Stunden Arbeit in Sekunden
        "pause": 20 * 60,  # 20 Minuten Pause
        "jobs": {
            "MA1": "J1",
            "MA2": "J2",
            "MA3": "J3",
        },
    },
    {
        "name": "Runde 2",
        "arbeitszeit": 2 * 3600,  # 2 Stunden Arbeit
        "pause": 40 * 60,  # 40 Minuten Pause
        "jobs": {
            "MA1": "J2",
            "MA2": "J3",
            "MA3": "J1",
        },
    },
    {
        "name": "Runde 3",
        "arbeitszeit": 2 * 3600,  # 2 Stunden Arbeit
        "pause": 0,  # Keine Pause am Ende der Schicht
        "jobs": {
            "MA1": "J3",
            "MA2": "J1",
            "MA3": "J2",
        },
    },
]

# Aus den Arbeitsrunden abgeleitete Dauerparameter
gesamt_pausenzeit = sum(runde.get("pause", 0) for runde in arbeitsrunden)
arbeitszeit_pro_tag = sum(runde["arbeitszeit"] for runde in arbeitsrunden)
Schichtdauer = arbeitszeit_pro_tag + gesamt_pausenzeit  # Gesamtdauer inkl. Pausen (8h)
Pausen = gesamt_pausenzeit  # Gesamtpausenzeit pro Schicht (60 Minuten)


##########Lern- und Vergessenskurven Initialisieren##########
# Initialisierung der Lern- und Vergessenskurven pro Mitarbeitendem
lernkurve_mitarbeiter = {
    ma: Lernkurve(
        t_initiale_AFZ=t_initiale_AFZ[ma],
        M=GW_MA[ma],
        k=LF_MA[ma],
        prozesstaetigkeiten=taetigkeiten_liste,
        m_min=uebungsfaktor_parameter["m_min"],
        m_max=uebungsfaktor_parameter["m_max"],
    )
    for ma in mitarbeitende
}

vergessenskurve_mitarbeiter = {
    ma: Vergessenskurve(
        t_initiale_AFZ=t_initiale_AFZ[ma],  # Initiale Bearbeitungszeiten
        c=VF_MA[ma],   # Vergessensfaktoren
        prozesstaetigkeiten=taetigkeiten_liste,
    )
    for ma in mitarbeitende
}
##########
