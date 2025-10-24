import Simulation_Berechnungen as Berechnung
import Simulation_Eingabe as Eingabe
import pandas as pd

ANZEIGE_MITARBEITENDE = list(getattr(Berechnung, "simulierte_mitarbeitende", []))
if not ANZEIGE_MITARBEITENDE:
    ANZEIGE_MITARBEITENDE = list(Eingabe.mitarbeitende)


''' Vorbereitung der Job-Zusammenfassung und Aggregation vollständiger Durchläufe für die Job-Protokolle'''
# Ermitteln der Bearbeitungszeiten auf Jobbasis
# Neue Matrix für die Jobs und ihre aufsummierte Bearbeitungszeiten
Jobbearbeitungszeit = {}

for ma in ANZEIGE_MITARBEITENDE:
    historie = Berechnung.job_history.get(ma, [])
    job_df = pd.DataFrame(
        columns=[
            "Job",
            "Bearbeitungszeit",
            "Aufsummierte Bearbeitungszeit",
            "Output_geplant",
            "Output_gut",
            "Ausschuss",
            "Output_geplant_ohne_Fehler",
            "Output_gut_ohne_Fehler",
            "Ausschuss_ohne_Fehler",
            "Ø_Fehlerquote",
        ]
    )
    gesamt_bearbeitungszeit = 0
    alle_jobs = getattr(Eingabe, "jobs", None)
    if alle_jobs is None:
        alle_jobs = list(Eingabe.jobs_zuordnung.keys())
    job_progress = {job: [] for job in alle_jobs}
    job_dauer = {job: 0 for job in alle_jobs}

    for eintrag in historie:
        job = eintrag.get("Job")
        taetigkeit = eintrag.get("Tätigkeit")
        bearbeitungszeit = eintrag.get("Bearbeitungszeit", 0)
        if job is None:
            continue
        erforderliche_taetigkeiten = Eingabe.jobs_zuordnung.get(job, [])
        if not erforderliche_taetigkeiten:
            continue
        job_progress.setdefault(job, [])
        job_dauer.setdefault(job, 0)
        job_progress[job].append(eintrag)
        job_dauer[job] += bearbeitungszeit

        if len(job_progress[job]) == len(erforderliche_taetigkeiten):
            erster_schritt = job_progress[job][0]
            letzter_schritt = job_progress[job][-1]
            output_geplant = erster_schritt.get(
                "Output_input",
                erster_schritt.get("Output_geplant", 0),
            )
            output_geplant_ohne = erster_schritt.get(
                "Output_input_ohne_Fehler",
                erster_schritt.get("Output_geplant_ohne_Fehler", output_geplant),
            )
            output_gut = letzter_schritt.get("Output_gut", 0)
            output_gut_ohne = letzter_schritt.get(
                "Output_gut_ohne_Fehler",
                output_geplant_ohne,
            )
            ausschuss = output_geplant - output_gut
            if ausschuss < 0:
                ausschuss = 0
            ausschuss_ohne = output_geplant_ohne - output_gut_ohne
            if ausschuss_ohne < 0:
                ausschuss_ohne = 0
            if output_geplant > 0:
                durchschnitt_fehlerquote = ausschuss / output_geplant
            else:
                durchschnitt_fehlerquote = 0
            gesamt_bearbeitungszeit += job_dauer[job]
            job_df.loc[len(job_df)] = [
                job,
                job_dauer[job],
                gesamt_bearbeitungszeit,
                output_geplant,
                output_gut,
                ausschuss,
                output_geplant_ohne,
                output_gut_ohne,
                ausschuss_ohne,
                durchschnitt_fehlerquote,
            ]
            job_progress[job] = []
            job_dauer[job] = 0

    Jobbearbeitungszeit[ma] = job_df

''' Detailtabelle für Tätigkeiten '''
# Erstellen der Liste für die Ausführungszeiten der einzelnen Tätigkeiten
Tätigkeitenbearbeitungszeit = {} # Initialisiert den DataFrame für die Jobs

# Ermittle die Bearbeitungszeiten pro Produkt auf Basis der Tätigkeitenzuordnung
for ma in ANZEIGE_MITARBEITENDE:
    historie = Berechnung.job_history.get(ma, [])
    taetigkeiten_df = pd.DataFrame(
        columns=[
            "Job",
            "Tätigkeit",
            "Runde",
            "Simulationszeit",
            "Bearbeitungszeit",
            "Kompetenzstufe",
            "Reduktion_%",
            "Fehlerquote",
            "Output_geplant",
            "Output_gut",
            "Ausschuss",
            "Output_geplant_ohne_Fehler",
            "Output_gut_ohne_Fehler",
            "Ausschuss_ohne_Fehler",
        ]
    )
    for eintrag in historie:
        taetigkeiten_df.loc[len(taetigkeiten_df)] = [
            eintrag.get("Job"),
            eintrag.get("Tätigkeit"),
            eintrag.get("Runde"),
            eintrag.get("Simulationszeit"),
            eintrag.get("Bearbeitungszeit"),
            eintrag.get("Kompetenzstufe"),
            eintrag.get("Reduktion_%"),
            eintrag.get("Fehlerquote"),
            eintrag.get("Output_geplant"),
            eintrag.get("Output_gut"),
            eintrag.get("Ausschuss"),
            eintrag.get("Output_geplant_ohne_Fehler"),
            eintrag.get("Output_gut_ohne_Fehler"),
            eintrag.get("Ausschuss_ohne_Fehler"),
        ]

    Tätigkeitenbearbeitungszeit[ma] = taetigkeiten_df

''' Excel-Export '''
excel_datei = "Bearbeitungszeiten.xlsx"
# Exportiere die DataFrames in eine Excel-Datei mit zwei Tabellenblättern pro Mitarbeitendem
with pd.ExcelWriter(excel_datei) as writer:
    for ma in ANZEIGE_MITARBEITENDE:
        job_df = Jobbearbeitungszeit.get(
            ma,
            pd.DataFrame(
                columns=[
                    "Job",
                    "Bearbeitungszeit",
                    "Aufsummierte Bearbeitungszeit",
                    "Output_geplant",
                    "Output_gut",
                    "Ausschuss",
                    "Output_geplant_ohne_Fehler",
                    "Output_gut_ohne_Fehler",
                    "Ausschuss_ohne_Fehler",
                    "Ø_Fehlerquote",
                ]
            ),
        )
        sheet_name = f"{ma}_Jobs"[:31]
        job_df.to_excel(writer, sheet_name=sheet_name, index=True)
    for ma in ANZEIGE_MITARBEITENDE:
        taetigkeiten_df = Tätigkeitenbearbeitungszeit.get(
            ma,
            pd.DataFrame(
                columns=[
                    "Job",
                    "Tätigkeit",
                    "Runde",
                    "Simulationszeit",
                    "Bearbeitungszeit",
                    "Kompetenzstufe",
                    "Reduktion_%",
                    "Fehlerquote",
                    "Output_geplant",
                    "Output_gut",
                    "Ausschuss",
                    "Output_geplant_ohne_Fehler",
                    "Output_gut_ohne_Fehler",
                    "Ausschuss_ohne_Fehler",
                ]
            ),
        )
        sheet_name = f"{ma}_Tätigkeiten"[:31]

        taetigkeiten_df.to_excel(writer, sheet_name=sheet_name, index=True)





