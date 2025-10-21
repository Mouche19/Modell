from typing import Any, Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import Simulation_Eingabe as Eingabe

from Simulation_Engine import (
    PrintLogger,
    SimulationLogger,
    SimulationRunner,
    Simulationsergebnis,
    berechne_outputsumme_fuer_tag,
)

pd.options.display.float_format = "{:.3f}".format  # Zeigt 3 Nachkommastellen für alle Pandas-Ausgaben


aktive_mitarbeitende = list(Eingabe.mitarbeitende)
arbeitsrunden = Eingabe.arbeitsrunden
if not arbeitsrunden:
    raise ValueError("Es muss mindestens eine Arbeitsrunde definiert sein.")

alle_jobs = getattr(Eingabe, "jobs", None)
if alle_jobs is None:
    alle_jobs = list(getattr(Eingabe, "jobs_zuordnung", {}).keys())

personalrisiken_vorgabe = getattr(Eingabe, "personalrisiken", None)
produktionsstoerungen_vorgabe = getattr(Eingabe, "produktionsstoerungen", None)

# Vorberechnung der Kompetenzgrenzen pro Mitarbeitendem und Tätigkeit
kompetenz_parameter: Dict[str, Dict[str, Dict[str, float]]] = {}
for ma in aktive_mitarbeitende:
    kompetenz_parameter[ma] = {}
    gw_ma = getattr(Eingabe, "GW_MA", {}).get(ma, {})
    for t in Eingabe.taetigkeiten_liste:
        initial = Eingabe.t_initiale_AFZ[ma][t]
        grenzwert = initial * gw_ma.get(t, 1)
        differenz = max(initial - grenzwert, 0)
        kompetenz_parameter[ma][t] = {
            "initial": initial,
            "grenzwert": grenzwert,
            "differenz": differenz,
        }


def berechne_kompetenzstufe(ma, taetigkeit, aktuelle_afz):
    parameter = kompetenz_parameter[ma][taetigkeit]
    initial = parameter["initial"]
    differenz = parameter["differenz"]

    if differenz <= 0:
        return 5 if aktuelle_afz <= initial else 1

    reduktion = (initial - aktuelle_afz) / differenz
    reduktion = max(0, min(reduktion, 1))

    if reduktion < 0.5:
        return 1
    if reduktion < 0.6:
        return 2
    if reduktion < 0.7:
        return 3
    if reduktion < 0.8:
        return 4
    return 5


def ermittle_fehlerquote(ma, taetigkeit, kompetenzstufe, reduktion_rel=None):
    parameter = getattr(Eingabe, "fehlerquote_parameter", {})
    standard = getattr(Eingabe, "standard_fehlerquote", {})

    stufenwerte = parameter.get(ma, {}).get(taetigkeit, {})
    if kompetenzstufe in stufenwerte:
        return max(0.0, min(stufenwerte[kompetenzstufe], 1.0))
    if stufenwerte:
        naechste_stufe = max(stufenwerte.keys())
        return max(0.0, min(stufenwerte[naechste_stufe], 1.0))

    if kompetenzstufe in standard:
        return max(0.0, min(standard[kompetenzstufe], 1.0))
    if standard:
        return max(0.0, min(standard[max(standard.keys())], 1.0))

    if reduktion_rel is not None:
        return max(0.0, min(1.0, (1 - reduktion_rel)))

    return 0.0


def ermittle_outputmenge(ma, taetigkeit):
    konfig = getattr(Eingabe, "output_pro_ausfuehrung", {})

    if isinstance(konfig, dict):
        if taetigkeit in konfig and not isinstance(konfig[taetigkeit], dict):
            return konfig[taetigkeit]
        if ma in konfig and isinstance(konfig[ma], dict):
            return konfig[ma].get(taetigkeit, 1)
        if taetigkeit in konfig and isinstance(konfig[taetigkeit], dict):
            werte = konfig[taetigkeit]
            if isinstance(werte, dict):
                return werte.get(ma, werte.get("default", 1))

    return 1


def fuehre_simulation(
    *,
    fehlerfrei: bool = False,
    logger: Optional[SimulationLogger] = None,
    event_handlers: Optional[Dict[str, Iterable[Callable[..., None]]]] = None,
    personalrisiken: Optional[Dict[str, Iterable[Dict[str, Any]]]] = None,
    produktionsstoerungen: Optional[Iterable[Dict[str, Any]]] = None,
    ziel_tag_limit: Optional[int] = None,
    ignoriere_ziele: bool = False,
) -> Simulationsergebnis:
    """Fassade für die Simulation, die den :class:`SimulationRunner` verwendet."""

    runner = SimulationRunner(
        Eingabe,
        logger=logger or PrintLogger(),
        event_handlers=event_handlers,
        personalrisiken=personalrisiken,
        produktionsstoerungen=produktionsstoerungen,
    )
    return runner.run(
        fehlerfrei=fehlerfrei,
        ziel_tag_limit=ziel_tag_limit,
        ignoriere_ziele=ignoriere_ziele,
    )

simulation_mit_fehler = fuehre_simulation(
    fehlerfrei=False,
    personalrisiken=personalrisiken_vorgabe,
    produktionsstoerungen=produktionsstoerungen_vorgabe,
)

letzter_tag_mit_fehler = simulation_mit_fehler.letzter_simulationstag
ziel_tag_limit = (
    int(letzter_tag_mit_fehler)
    if letzter_tag_mit_fehler is not None and letzter_tag_mit_fehler > 0
    else None
)
ignoriere_ziele_fehlerfrei = bool(ziel_tag_limit and simulation_mit_fehler.ziel_status)

simulation_ohne_fehler = fuehre_simulation(
    fehlerfrei=True,
    personalrisiken=personalrisiken_vorgabe,
    produktionsstoerungen=produktionsstoerungen_vorgabe,
    ziel_tag_limit=ziel_tag_limit,
    ignoriere_ziele=ignoriere_ziele_fehlerfrei,
)

output_data_all = simulation_mit_fehler.output_data_all
job_history = simulation_mit_fehler.job_history
kompetenz_protokoll = simulation_mit_fehler.kompetenz_protokoll
uebungsfaktor_protokoll = simulation_mit_fehler.uebungsfaktor_protokoll

output_data_all_ohne_fehler = simulation_ohne_fehler.output_data_all
job_history_ohne_fehler = simulation_ohne_fehler.job_history

alle_labels: Set[str] = set(output_data_all.keys()) | set(output_data_all_ohne_fehler.keys())
alle_labels |= set(job_history.keys()) | set(job_history_ohne_fehler.keys())
alle_labels |= set(kompetenz_protokoll.keys())
simulierte_mitarbeitende = sorted(alle_labels)
if not simulierte_mitarbeitende:
    simulierte_mitarbeitende = list(aktive_mitarbeitende)

# Ergänze die job_history um die Ergebnisse der fehlerfreien Simulation
for ma, historie in job_history.items():
    fehlerfrei_map = {
        eintrag.get("DurchlaufNr"): eintrag
        for eintrag in job_history_ohne_fehler.get(ma, [])
    }
    for eintrag in historie:
        durchlauf = eintrag.get("DurchlaufNr")
        ff = fehlerfrei_map.get(durchlauf, {})
        eintrag["Output_input_ohne_Fehler"] = ff.get("Output_input", 0.0)
        eintrag["Output_geplant_ohne_Fehler"] = ff.get("Output_geplant", 0.0)
        eintrag["Output_gut_ohne_Fehler"] = ff.get("Output_gut", 0.0)
        eintrag["Ausschuss_ohne_Fehler"] = ff.get("Ausschuss", 0.0)


# Füge die fehlerfreie Simulation als zusätzliche Spalten in den Tätigkeits-DataFrames hinzu
for ma in simulierte_mitarbeitende:
    for taetigkeit in Eingabe.taetigkeiten_liste:
        df_real = output_data_all.get(ma, {}).get(taetigkeit)
        df_ff = output_data_all_ohne_fehler.get(ma, {}).get(taetigkeit)
        if df_real is None:
            continue
        if df_ff is None or df_ff.empty:
            df_real["Output_input_ohne_Fehler"] = 0.0
            df_real["Output_geplant_ohne_Fehler"] = 0.0
            df_real["Output_gut_ohne_Fehler"] = 0.0
            df_real["Ausschuss_ohne_Fehler"] = 0.0
            df_real["Output_input_ohne_Fehler_kumuliert"] = 0.0
            df_real["Output_gut_ohne_Fehler_kumuliert"] = 0.0
            df_real["Ausschuss_ohne_Fehler_kumuliert"] = 0.0
            continue
        ff_subset = df_ff[[
            "DurchlaufNr",
            "Output_input",
            "Output_geplant",
            "Output_gut",
            "Ausschuss",
        ]].copy()
        ff_subset.rename(
            columns={
                "Output_input": "Output_input_ohne_Fehler",
                "Output_geplant": "Output_geplant_ohne_Fehler",
                "Output_gut": "Output_gut_ohne_Fehler",
                "Ausschuss": "Ausschuss_ohne_Fehler",
            },
            inplace=True,
        )
        df_real = df_real.merge(ff_subset, on="DurchlaufNr", how="left")
        for spalte in [
            "Output_input_ohne_Fehler",
            "Output_geplant_ohne_Fehler",
            "Output_gut_ohne_Fehler",
            "Ausschuss_ohne_Fehler",
        ]:
            df_real[spalte] = pd.to_numeric(df_real.get(spalte), errors="coerce").fillna(0.0)
        df_real["Output_input_ohne_Fehler_kumuliert"] = df_real[
            "Output_input_ohne_Fehler"
        ].cumsum()
        df_real["Output_gut_ohne_Fehler_kumuliert"] = df_real[
            "Output_gut_ohne_Fehler"
        ].cumsum()
        df_real["Ausschuss_ohne_Fehler_kumuliert"] = df_real[
            "Ausschuss_ohne_Fehler"
        ].cumsum()
        output_data_all[ma][taetigkeit] = df_real


''' Ergebnis für alle Tätigkeiten anzeigen '''
for ma, daten in output_data_all.items():
    for t, df in daten.items():
        print(f"Ergebnisse für Mitarbeiter {ma} - Tätigkeit {t}:\n{df}\n")


''' Kombinieren aller AFZ aus den Datenframes in eine Tabelle je Mitarbeitendem '''
output_data_compressed = {}
for ma in simulierte_mitarbeitende:
    compressed_df: Optional[pd.DataFrame] = None
    for t in Eingabe.taetigkeiten_liste:
        df_task = output_data_all.get(ma, {}).get(t)
        if df_task is None or df_task.empty:
            continue
        df_task = df_task.copy()
        df_task["DurchlaufNr"] = pd.to_numeric(df_task.get("DurchlaufNr"), errors="coerce")
        df_task["AFZ"] = pd.to_numeric(df_task.get("AFZ"), errors="coerce")
        df_task = df_task.dropna(subset=["DurchlaufNr"])
        if df_task["AFZ"].dropna().empty:
            continue
        df_task = df_task.sort_values("DurchlaufNr")
        df_task["AFZ"] = df_task["AFZ"].ffill()
        df_task = df_task.dropna(subset=["AFZ"])

        df_subset = df_task[["DurchlaufNr", "AFZ"]].copy()
        df_subset.rename(columns={"AFZ": t}, inplace=True)
        if compressed_df is None:
            compressed_df = df_subset
        else:
            compressed_df = compressed_df.merge(df_subset, on="DurchlaufNr", how="outer")
    if compressed_df is None:
        output_data_compressed[ma] = pd.DataFrame(columns=["DurchlaufNr", *Eingabe.taetigkeiten_liste])
    else:
        compressed_df = compressed_df.sort_values("DurchlaufNr").reset_index(drop=True)
        output_data_compressed[ma] = compressed_df

''' Konsolidierte Tabelle über alle Mitarbeitenden und Tätigkeiten '''
gesamt_output_rows = []
for ma, taetigkeiten in output_data_all.items():
    for taetigkeit, df in taetigkeiten.items():
        if df.empty:
            continue
        df_copy = df.copy()
        numeric_spalten = [
            "Output_input",
            "Output_input_ohne_Fehler",
            "Output_geplant",
            "Output_geplant_ohne_Fehler",
            "Output_gut",
            "Output_gut_ohne_Fehler",
            "Ausschuss",
            "Ausschuss_ohne_Fehler",
        ]
        for spalte in numeric_spalten:
            df_copy[spalte] = pd.to_numeric(df_copy.get(spalte), errors="coerce").fillna(0.0)
        df_copy["Ausschuss_kumuliert"] = df_copy["Ausschuss"].cumsum()
        df_copy["Output_geplant_kumuliert"] = df_copy["Output_geplant"].cumsum()
        df_copy["Output_input_ohne_Fehler_kumuliert"] = df_copy[
            "Output_input_ohne_Fehler"
        ].cumsum()
        df_copy["Output_gut_ohne_Fehler_kumuliert"] = df_copy[
            "Output_gut_ohne_Fehler"
        ].cumsum()
        df_copy["Ausschuss_ohne_Fehler_kumuliert"] = df_copy[
            "Ausschuss_ohne_Fehler"
        ].cumsum()
        df_copy["Mitarbeiter"] = ma
        df_copy["Tätigkeit"] = taetigkeit
        gesamt_output_rows.append(df_copy)

if gesamt_output_rows:
    output_data_flat = pd.concat(gesamt_output_rows, ignore_index=True)
    spalten = [
        "Mitarbeiter",
        "Tätigkeit",
        "DurchlaufNr",
        "AFZ",
        "AFA_pre",
        "AFZ_post",
        "AFA_post",
        "VG_Dauer",
        "Ausgefuehrt",
        "Sim_zeit",
        "Output_input",
        "Output_input_ohne_Fehler",
        "Output_geplant",
        "Output_geplant_ohne_Fehler",
        "Output_gut",
        "Output_gut_ohne_Fehler",
        "Ausschuss",
        "Ausschuss_ohne_Fehler",
        "Output_geplant_kumuliert",
        "Output_input_ohne_Fehler_kumuliert",
        "Output_gut_ohne_Fehler_kumuliert",
        "Ausschuss_kumuliert",
        "Ausschuss_ohne_Fehler_kumuliert",
        "Fehlerquote",
    ]
    output_data_flat = output_data_flat.reindex(columns=spalten)
else:
    output_data_flat = pd.DataFrame(
        columns=[
            "Mitarbeiter",
            "Tätigkeit",
            "DurchlaufNr",
            "AFZ",
            "AFA_pre",
            "AFZ_post",
            "AFA_post",
            "VG_Dauer",
            "Ausgefuehrt",
            "Sim_zeit",
            "Output_input",
            "Output_input_ohne_Fehler",
            "Output_geplant",
            "Output_geplant_ohne_Fehler",
            "Output_gut",
            "Output_gut_ohne_Fehler",
            "Ausschuss",
            "Ausschuss_ohne_Fehler",
            "Output_geplant_kumuliert",
            "Output_input_ohne_Fehler_kumuliert",
            "Output_gut_ohne_Fehler_kumuliert",
            "Ausschuss_kumuliert",
            "Ausschuss_ohne_Fehler_kumuliert",
            "Fehlerquote",
        ]
    )


# Gesamtausschuss über alle Mitarbeitenden und Tätigkeiten hinweg ausgeben
has_output_data = isinstance(output_data_flat, pd.DataFrame) and not output_data_flat.empty
if has_output_data:
    gesamt_ausschuss = output_data_flat["Ausschuss"].sum()
    gesamt_output_ohne_fehler = output_data_flat["Output_gut_ohne_Fehler"].sum()
    if gesamt_output_ohne_fehler > 0:
        ausschuss_quote = gesamt_ausschuss / gesamt_output_ohne_fehler * 100
    else:
        ausschuss_quote = 0.0
    print(
        "Kumulierter Ausschuss aller Mitarbeitenden und Tätigkeiten: "
        f"{gesamt_ausschuss:.3f} von {gesamt_output_ohne_fehler:.3f}, "
        f"{ausschuss_quote:.2f}% Ausschuss"
    )

letzter_arbeitstag: Optional[int] = None
feierabend_tage: List[int] = []
for protokolle in kompetenz_protokoll.values():
    for eintrag in protokolle:
        if eintrag.get("Ereignis") != "Feierabend":
            continue
        try:
            tag_wert = int(float(eintrag.get("Tag")))
        except (TypeError, ValueError):
            continue
        feierabend_tage.append(tag_wert)

if feierabend_tage:
    letzter_arbeitstag = max(feierabend_tage)

produkt_last_day: Optional[float] = None
fehlerwerte_last_day: List[float] = []
kompetenzwerte_last_day: List[float] = []

if letzter_arbeitstag is not None:
    produkt_last_day = berechne_outputsumme_fuer_tag(
        job_history, letzter_arbeitstag
    )

    for protokolle in kompetenz_protokoll.values():
        for eintrag in protokolle:
            if eintrag.get("Ereignis") != "Feierabend":
                continue
            try:
                tag_wert = int(float(eintrag.get("Tag")))
            except (TypeError, ValueError):
                continue
            if tag_wert != letzter_arbeitstag:
                continue
            fehlerquote = eintrag.get("Fehlerquote")
            if fehlerquote is not None:
                try:
                    fehlerwerte_last_day.append(float(fehlerquote))
                except (TypeError, ValueError):
                    pass
            kompetenz = eintrag.get("Kompetenzstufe")
            if kompetenz is not None:
                try:
                    kompetenzwerte_last_day.append(float(kompetenz))
                except (TypeError, ValueError):
                    pass

if letzter_arbeitstag is not None:
    if produkt_last_day is not None:
        print(
            f"Produktivität am letzten Tag (Tag {letzter_arbeitstag}): "
            f"{produkt_last_day:.3f} Output-Einheiten"
        )
    else:
        print(
            f"Produktivität am letzten Tag (Tag {letzter_arbeitstag}): "
            "Keine Daten verfügbar"
        )

    if fehlerwerte_last_day:
        durchschnitt_fehler = float(np.mean(fehlerwerte_last_day))
        print(
            "Durchschnittliche Fehlerquote aller Mitarbeitenden am letzten Tag: "
            f"{durchschnitt_fehler:.4f} ({durchschnitt_fehler * 100:.2f}%)"
        )
    else:
        print(
            "Durchschnittliche Fehlerquote aller Mitarbeitenden am letzten Tag: "
            "Keine Daten verfügbar"
        )

    if kompetenzwerte_last_day:
        durchschnitt_kompetenz = float(np.mean(kompetenzwerte_last_day))
        varianz_kompetenz = float(np.var(kompetenzwerte_last_day))
        print(
            "Durchschnittliches Kompetenzniveau am letzten Tag: "
            f"{durchschnitt_kompetenz:.3f} (Varianz: {varianz_kompetenz:.4f})"
        )
    else:
        print(
            "Durchschnittliches Kompetenzniveau am letzten Tag: "
            "Keine Daten verfügbar"
        )
else:
    print("Es konnten keine Kennzahlen für den letzten Arbeitstag ermittelt werden.")


# Kompetenzdaten aufbereiten
kompetenz_dfs = {}
gesamt_kompetenz_rows = []
for ma, protokolle in kompetenz_protokoll.items():
    df = pd.DataFrame(protokolle)
    if not df.empty:
        if "Ereignis" not in df.columns:
            df["Ereignis"] = "Feierabend"
        df.insert(0, "Mitarbeiter", ma)
        kompetenz_dfs[ma] = df
        gesamt_kompetenz_rows.append(df)
    else:
        kompetenz_dfs[ma] = df

if gesamt_kompetenz_rows:
    kompetenz_flat = pd.concat(gesamt_kompetenz_rows, ignore_index=True)
else:
    kompetenz_flat = pd.DataFrame(
        columns=[
            "Mitarbeiter",
            "Tag",
            "Tätigkeit",
            "AFZ",
            "Kompetenzstufe",
            "Reduktion_%",
            "Fehlerquote",
            "Ereignis",
        ]
    )

uebungsfaktor_dfs: Dict[str, pd.DataFrame] = {}
uebungsfaktor_flat = pd.DataFrame(
    columns=[
        "Mitarbeiter",
        "Tag",
        "Tätigkeit",
        "m_abbau",
        "m_aufbau",
        "q_stunden",
        "n_ges",
        "Komplexität",
    ]
)
if isinstance(uebungsfaktor_protokoll, dict):
    for label, eintraege in uebungsfaktor_protokoll.items():
        df = pd.DataFrame(eintraege or [])
        if df.empty:
            continue
        if "Mitarbeiter" not in df.columns:
            df.insert(0, "Mitarbeiter", label)
        df = df[[
            col
            for col in [
                "Mitarbeiter",
                "Tag",
                "Tätigkeit",
                "m_abbau",
                "m_aufbau",
                "q_stunden",
                "n_ges",
                "Komplexität",
            ]
            if col in df.columns
        ]].copy()
        uebungsfaktor_dfs[label] = df
    if uebungsfaktor_dfs:
        uebungsfaktor_flat = pd.concat(uebungsfaktor_dfs.values(), ignore_index=True)

''' Schreiben in Excel '''
with pd.ExcelWriter("output_data_all.xlsx", engine='xlsxwriter') as writer:
    for ma, df in output_data_compressed.items():
        df.to_excel(writer, sheet_name=f"{ma}_AFZ", index=False)
        for taetigkeit, taetigkeits_df in output_data_all[ma].items():
            if not taetigkeits_df.empty:
                sheet_name = f"{ma}_{taetigkeit}"[:31]
                taetigkeits_df = taetigkeits_df.copy()
                for spalte in [
                    "Output_input",
                    "Output_geplant",
                    "Output_gut",
                    "Ausschuss",
                    "Output_input_ohne_Fehler",
                    "Output_geplant_ohne_Fehler",
                    "Output_gut_ohne_Fehler",
                    "Ausschuss_ohne_Fehler",
                ]:
                    taetigkeits_df[spalte] = pd.to_numeric(
                        taetigkeits_df.get(spalte), errors="coerce"
                    ).fillna(0.0)
                taetigkeits_df["Ausschuss_kumuliert"] = taetigkeits_df["Ausschuss"].cumsum()
                taetigkeits_df["Output_geplant_kumuliert"] = taetigkeits_df["Output_geplant"].cumsum()
                taetigkeits_df["Output_input_ohne_Fehler_kumuliert"] = taetigkeits_df[
                    "Output_input_ohne_Fehler"
                ].cumsum()
                taetigkeits_df["Output_gut_ohne_Fehler_kumuliert"] = taetigkeits_df[
                    "Output_gut_ohne_Fehler"
                ].cumsum()
                taetigkeits_df["Ausschuss_ohne_Fehler_kumuliert"] = taetigkeits_df[
                    "Ausschuss_ohne_Fehler"
                ].cumsum()
                taetigkeits_df.to_excel(writer, sheet_name=sheet_name, index=False)
        kompetenz_df = kompetenz_dfs.get(ma)
        if kompetenz_df is not None and not kompetenz_df.empty:
            sheet_name = f"{ma}_Kompetenz"[:31]
            kompetenz_df.to_excel(writer, sheet_name=sheet_name, index=False)
    for label, df in uebungsfaktor_dfs.items():
        if df is None or df.empty:
            continue
        sheet_name = f"{label}_Uebungsfaktor"[:31]
        df_sorted = df.sort_values(["Tag", "Tätigkeit"]).reset_index(drop=True)
        df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
    if not output_data_flat.empty:
        output_data_flat.to_excel(writer, sheet_name="Gesamtübersicht", index=False)
    if not kompetenz_flat.empty:
        kompetenz_flat.to_excel(writer, sheet_name="Kompetenzgesamt", index=False)
    if not uebungsfaktor_flat.empty:
        uebungsfaktor_flat.sort_values(["Mitarbeiter", "Tag", "Tätigkeit"]).to_excel(
            writer,
            sheet_name="Uebungsfaktor_Gesamt",
            index=False,
        )
if not output_data_flat.empty:
    output_data_flat.to_csv("output_data_all.csv", index=False)

if not kompetenz_flat.empty:
    kompetenz_flat.to_csv("kompetenz_protokoll.csv", index=False)

if not uebungsfaktor_flat.empty:
    uebungsfaktor_flat.to_csv("uebungsfaktor_protokoll.csv", index=False)

""" Flexibilitäts- und Kompetenzanalyse """
flex_excel_datei = "Flexibilitaet_Kompetenz_Auswertung.xlsx"

kompetenz_feierabend = kompetenz_flat[
    kompetenz_flat.get("Ereignis") == "Feierabend"
].copy()

tages_matrizen: List[Dict[str, Any]] = []
individuelle_gesamt_rows: List[pd.DataFrame] = []
kollektive_gesamt_rows: List[pd.DataFrame] = []

if not kompetenz_feierabend.empty:
    for tag, df_tag in kompetenz_feierabend.groupby("Tag"):
        matrix = (
            df_tag.pivot(index="Mitarbeiter", columns="Tätigkeit", values="Kompetenzstufe")
            .reindex(index=simulierte_mitarbeitende, columns=Eingabe.taetigkeiten_liste)
        )

        individuelle_df = pd.DataFrame(
            {
                "Ø Kompetenz individuell": matrix.mean(axis=1, skipna=True),
                "Varianz Kompetenz": matrix.var(axis=1, skipna=True, ddof=0),
            }
        )
        individuelle_df.index.name = "Mitarbeiter"
        individuelle_df["Tag"] = tag
        individuelle_gesamt_rows.append(individuelle_df.reset_index())

        kollektive_df = pd.DataFrame(
            {
                "Ø Kompetenz Tätigkeit": matrix.mean(axis=0, skipna=True),
                "Varianz Kompetenz Tätigkeit": matrix.var(axis=0, skipna=True, ddof=0),
            }
        )
        kollektive_df.index.name = "Tätigkeit"

        gesamt_werte = matrix.to_numpy(dtype=float)
        gesamt_werte = gesamt_werte[~np.isnan(gesamt_werte)]
        if gesamt_werte.size > 0:
            gesamt_mittel = float(np.mean(gesamt_werte))
            gesamt_varianz = float(np.var(gesamt_werte))
            kollektive_df.loc["Gesamt"] = {
                "Ø Kompetenz Tätigkeit": gesamt_mittel,
                "Varianz Kompetenz Tätigkeit": gesamt_varianz,
            }
        kollektive_df["Tag"] = tag
        kollektive_gesamt_rows.append(
            kollektive_df.reset_index().rename(columns={"index": "Tätigkeit"})
        )

        tages_matrizen.append(
            {
                "Tag": tag,
                "Matrix": matrix,
                "Individuell": individuelle_df,
                "Kollektiv": kollektive_df,
            }
        )

with pd.ExcelWriter(flex_excel_datei, engine="xlsxwriter") as writer:
    if tages_matrizen:
        for eintrag in tages_matrizen:
            tag = eintrag["Tag"]
            matrix = eintrag["Matrix"].copy()
            matrix.insert(0, "Mitarbeiter", matrix.index)
            matrix.reset_index(drop=True, inplace=True)
            matrix_sheet = f"Tag_{tag}_Matrix"[:31]
            matrix.to_excel(writer, sheet_name=matrix_sheet, index=False)

            individuell = eintrag["Individuell"].reset_index().rename(
                columns={"index": "Mitarbeiter"}
            )
            individuell_sheet = f"Tag_{tag}_Individuell"[:31]
            individuell.to_excel(writer, sheet_name=individuell_sheet, index=False)

            kollektiv = eintrag["Kollektiv"].reset_index().rename(
                columns={"index": "Tätigkeit"}
            )
            kollektiv_sheet = f"Tag_{tag}_Kollektiv"[:31]
            kollektiv.to_excel(writer, sheet_name=kollektiv_sheet, index=False)

    gesamt_individuell_df = (
        pd.concat(individuelle_gesamt_rows, ignore_index=True)
        if individuelle_gesamt_rows
        else pd.DataFrame(columns=["Mitarbeiter", "Ø Kompetenz individuell", "Varianz Kompetenz", "Tag"])
    )
    gesamt_individuell_df.to_excel(writer, sheet_name="Gesamt_Individuell", index=False)

    gesamt_kollektiv_df = (
        pd.concat(kollektive_gesamt_rows, ignore_index=True)
        if kollektive_gesamt_rows
        else pd.DataFrame(
            columns=["Tätigkeit", "Ø Kompetenz Tätigkeit", "Varianz Kompetenz Tätigkeit", "Tag"]
        )
    )
    gesamt_kollektiv_df.to_excel(writer, sheet_name="Gesamt_Kollektiv", index=False)

