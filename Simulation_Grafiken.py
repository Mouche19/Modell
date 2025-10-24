import math
from itertools import cycle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.patches import Patch

# Einheitliche Schriftart und -größe für alle Diagramme 
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
})

import Simulation_Berechnungen as Berechnung
import Simulation_Eingabe as Eingabe
import Simulation_Output as Output_Tabellen


# Schwellenwerte für die Kompetenzstufen aus Simulation_Engine.SimulationRunner
# Die Werte geben an, ab welcher relativen Reduktion (in Dezimalform) die nächste
# Kompetenzstufe erreicht wird. Die Angaben werden für die Achsenbeschriftung der
# Kompetenzentwicklung-Diagramme genutzt. Bei Änderungen der zugrunde liegenden
# Logik (also Schwellwerte) in Simulation_Engine sollte diese Liste ebenfalls angepasst werden.
KOMPETENZSTUFEN_SCHWELLEN = { 
    1: 0.5,
    2: 0.6,
    3: 0.7,
    4: 0.8,
}

def _formatiere_prozentwert(wert: float) -> str: # gibt einen Prozentwert ohne überflüssige Nachkommastellen zurück
    
    prozent = wert * 100
    if abs(prozent - round(prozent)) < 1e-9:
        return f"{int(round(prozent))} %"
    return f"{prozent:.1f} %"


def _ermittle_kompetenz_ticklabels() -> List[str]: # Erzeugt Achsenbeschriftungen für Kompetenzstufen inkl. Prozentwert
    
    labels: List[str] = []
    fuer_stufe_5 = KOMPETENZSTUFEN_SCHWELLEN.get(4, 0.9)
    for stufe in range(1, 6):
        if stufe < 5:
            schwelle = KOMPETENZSTUFEN_SCHWELLEN.get(stufe)
            if schwelle is None:
                labels.append(str(stufe))
                continue
            labels.append(f"{stufe} (< {_formatiere_prozentwert(schwelle)})")
        else:
            labels.append(f"{stufe} (≥ {_formatiere_prozentwert(fuer_stufe_5)})")
    return labels

ANZEIGE_MITARBEITENDE = list(getattr(Berechnung, "simulierte_mitarbeitende", []))
if not ANZEIGE_MITARBEITENDE:
    ANZEIGE_MITARBEITENDE = list(Eingabe.mitarbeitende)
if not ANZEIGE_MITARBEITENDE:
    raise ValueError(
        "Es sind keine Mitarbeitenden für die grafische Darstellung definiert."
    )


# IFA-Farbpalette für Grafiken
FARBPALETTE_HEX = [
    "#d8234f",  # kräftiges Rot
#    "#101010",  # Schwarz, erschwert Analyse einiger Grafiken
    "#004f9d",  # kräftiges Blau
    "#7b0f2f",  # dunkles Rot
    "#6f6f73",  # Mittelgrau
    "#0b2f60",  # dunkles Blau
    "#f27d9b",  # Pink
    "#b6b7ba",  # helles Grau
    "#78a7d3",  # Hellblau
    "#f6c1d3",  # zartes Rosa
    "#dedfe0",  # sehr helles Grau
    "#b9cee6",  # sehr helles Blau
]

# Konfigurationsoptionen für die Farbgebung der Kurven.
# modus="liste" → es wird die definierte HEX-Liste (IFA-Farbpalette) verwendet
# modus="colormap" → Farben werden aus einer Matplotlib-Colormap erzeugt 
FARBSCHEMA = {
    "modus": "liste",  # mögliche Werte: "liste" oder "colormap"
    "liste": FARBPALETTE_HEX,
    "colormap": "tab10",  # wird nur verwendet, wenn modus="colormap"
}

# Spezielle Farbgebung für die Gesamtproduktivität (kann nach Bedarf verändert werden), nutzt momentan dieselbe Palette,
# damit alle Diagramme konsistent eingefärbt werden
FARBSCHEMA_GESAMT = {
    "modus": "liste",
    "liste": FARBPALETTE_HEX,
    "colormap": "Dark2",
}


def _sample_colormap(cmap, anzahl_farben: int): # für die Nutzung der Matplotlib-Farben (optional)
    
    if anzahl_farben <= 0:
        return []

    if anzahl_farben == 1:
        sample_points = [0.5]
    else:
        sample_points = [i / (anzahl_farben - 1) for i in range(anzahl_farben)]

    return [colors.to_hex(cmap(point)) for point in sample_points]


def _ermittle_colormap(cmap_name: str): # für die Nutzung der Matplotlib-Farben, gibt eine Matplotlib-Colormap anhand ihres Namens zurück (optional)
    
    try:
        registry = plt.colormaps()
    except Exception:
        registry = None

    if registry is not None:
        try:
            return registry[cmap_name]
        except (KeyError, TypeError):
            if hasattr(registry, "get"):
                cmap = registry.get(cmap_name)
                if cmap is not None:
                    return cmap

    for registry_name in ("cmap_d", "_cmap_registry"):
        cmap_dict = getattr(cm, registry_name, None)
        if isinstance(cmap_dict, dict) and cmap_name in cmap_dict:
            return cmap_dict[cmap_name]

    return None


def _farben_aus_colormap(cmap_name: str, anzahl_farben: int): # für die Nutzung der Matplotlib-Farben, versucht, eine Matplotlib-Colormap anhand des Namens zu nutzen (optional)

    if anzahl_farben <= 0:
        return []

    cmap = _ermittle_colormap(cmap_name)
    if cmap is None:
        return []

    return _sample_colormap(cmap, anzahl_farben)


def generiere_farben(anzahl_farben: int, schema: dict | None = None): # erzeugt eine Farbpalette gemäß der Konfiguration

    if anzahl_farben <= 0:
        return []

    if schema is None:
        schema = FARBSCHEMA

    modus = schema.get("modus", "liste")

    if modus == "colormap":
        cmap_name = schema.get("colormap", "tab10")
        farben = _farben_aus_colormap(cmap_name, anzahl_farben)
    else:
        farben = schema.get("liste", [])

    if not farben:
        farben = _farben_aus_colormap("tab10", anzahl_farben) # Fallback auf tab10, falls keine Farben konfiguriert wurden

    if not farben:
        prop_cycle = plt.rcParams.get("axes.prop_cycle")
        if prop_cycle is not None:
            cycle_colors = [eintrag.get("color") for eintrag in prop_cycle if "color" in eintrag]
            farben = [farbe for farbe in cycle_colors if farbe]

    if len(farben) < anzahl_farben and farben:          # bei Bedarf wird die Liste verlängert, sodass ausreichend Farben zur Verfügung stehen
        vielfaches = -(-anzahl_farben // len(farben))  
        farben = (farben * vielfaches)[:anzahl_farben]

    return farben


TAETIGKEITEN_FARBEN_BASIS = generiere_farben(len(Eingabe.taetigkeiten_liste))       # erzeugt eine Liste, die definiert, welche Farbe welcher Tätigkeit zugeordnet wird
if TAETIGKEITEN_FARBEN_BASIS:
    TAETIGKEIT_FARBEN = {
        taetigkeit: TAETIGKEITEN_FARBEN_BASIS[idx % len(TAETIGKEITEN_FARBEN_BASIS)]
        for idx, taetigkeit in enumerate(Eingabe.taetigkeiten_liste)
    }
else:
    TAETIGKEIT_FARBEN = {}


def _farbe_fuer_taetigkeit(     # liefert die konkrete Farbe für eine Tätigkeit
    taetigkeit: str, fallback_farben: Optional[List[str]], index: int
) -> Optional[str]:
    """Gibt die konfigurierte Farbe für ``taetigkeit`` zurück."""

    if taetigkeit in TAETIGKEIT_FARBEN:
        return TAETIGKEIT_FARBEN[taetigkeit]

    if fallback_farben:
        return fallback_farben[index % len(fallback_farben)]

    return None


# Festlegen des Zeitraums in Stunden (Schichtzeit)
Betrachtungsintervall = 8  # Angabe im Stundenformat -> Bsp.: 30min -> Betrachtungsintervall = 0.5

# Angabe zu Jobspezifischem Output:
JOB_FILTER = 'alle' # Angabe entweder als Jobname: 'J1' oder 'alle'


# Sammelcontainer für die spätere Gesamtproduktivität über alle Mitarbeitenden
gesamt_produktivitaet_rows = []

# Grafiken für alle definierten Mitarbeitenden werden erzeugt
for mitarbeiter_id in ANZEIGE_MITARBEITENDE:
    if mitarbeiter_id not in Berechnung.output_data_all:
        raise KeyError(f"Keine Simulationsergebnisse für Mitarbeitenden '{mitarbeiter_id}' vorhanden.")

    mitarbeitenden_details = Berechnung.output_data_all[mitarbeiter_id]

    # Importieren der Datenframes für die ausgewählte Person:
    AFZ_Tabelle = Berechnung.output_data_compressed.get(mitarbeiter_id, pd.DataFrame())

    Zeiten_Jobs = Output_Tabellen.Jobbearbeitungszeit.get(
        mitarbeiter_id,
        pd.DataFrame(columns=["Job", "Bearbeitungszeit", "Aufsummierte Bearbeitungszeit"]),
    )
    Zeiten_Tätigkeiten = Output_Tabellen.Tätigkeitenbearbeitungszeit.get(
        mitarbeiter_id,
        pd.DataFrame(columns=["Job", "Tätigkeit", "Runde", "Simulationszeit", "Bearbeitungszeit"]),
    )

    ''' Aufbereiten der erforderlichen Daten '''
    # Zeitraum definieren
    zeitraum = 3600 * Betrachtungsintervall
   
    # Neue Kategorie-Spalte basierend auf dem Zeitraum berechnen
    Zeiten_Jobs = Zeiten_Jobs[Zeiten_Jobs['Bearbeitungszeit'] != 0].copy()  # Filtern von allen Jobs, die außerhalb des Betrachtungsrahmens liegen

    
    # Grafische Aufbereitung der Ausführungszeiten auf Basis der Lern- und Vergessenseffekte

    # DataFrame erstellen
    plot_AFZ_Tabelle = pd.DataFrame(AFZ_Tabelle)
    spaltenreihenfolge = ["DurchlaufNr", *Eingabe.taetigkeiten_liste]
    if not plot_AFZ_Tabelle.empty:
        plot_AFZ_Tabelle = plot_AFZ_Tabelle.copy()
        if "DurchlaufNr" in plot_AFZ_Tabelle.columns:
            plot_AFZ_Tabelle["DurchlaufNr"] = pd.to_numeric(
                plot_AFZ_Tabelle["DurchlaufNr"], errors="coerce"
            )
            plot_AFZ_Tabelle = plot_AFZ_Tabelle.dropna(subset=["DurchlaufNr"])
            plot_AFZ_Tabelle = plot_AFZ_Tabelle.sort_values("DurchlaufNr").reset_index(
                drop=True
            )
        for t in Eingabe.taetigkeiten_liste:
            if t not in plot_AFZ_Tabelle.columns:
                plot_AFZ_Tabelle[t] = pd.NA
        taetigkeits_spalten = []
        for t in Eingabe.taetigkeiten_liste:
            if t in plot_AFZ_Tabelle.columns:
                plot_AFZ_Tabelle[t] = pd.to_numeric(
                    plot_AFZ_Tabelle[t], errors="coerce"
                )
                plot_AFZ_Tabelle[t] = plot_AFZ_Tabelle[t].ffill()
                taetigkeits_spalten.append(t)
        if taetigkeits_spalten:
            gueltige_zeilen = plot_AFZ_Tabelle[taetigkeits_spalten].notna().any(axis=1)
            plot_AFZ_Tabelle = plot_AFZ_Tabelle.loc[gueltige_zeilen]
        plot_AFZ_Tabelle = plot_AFZ_Tabelle.reindex(columns=spaltenreihenfolge)
    else:
        plot_AFZ_Tabelle = plot_AFZ_Tabelle.reindex(columns=spaltenreihenfolge)
    plot_Zeiten_Jobs = pd.DataFrame(Zeiten_Jobs)
    plot_Zeiten_Tätigkeiten = pd.DataFrame(Zeiten_Tätigkeiten)

    # Ermitteln der Durchlaufnummern, an denen eine Arbeitsrunde endet
    round_boundaries = []
    job_history_df = pd.DataFrame(Berechnung.job_history.get(mitarbeiter_id, []))
    if not job_history_df.empty and {'DurchlaufNr', 'Runde'}.issubset(job_history_df.columns):
        job_history_df = job_history_df.sort_values('DurchlaufNr').dropna(subset=['DurchlaufNr', 'Runde'])
        job_history_df['DurchlaufNr'] = pd.to_numeric(job_history_df['DurchlaufNr'], errors='coerce')
        job_history_df['Runde'] = pd.to_numeric(job_history_df['Runde'], errors='coerce')
        job_history_df = job_history_df.dropna(subset=['DurchlaufNr', 'Runde'])
        job_history_df['Runde'] = job_history_df['Runde'].astype(int)
        round_change_mask = job_history_df['Runde'].ne(job_history_df['Runde'].shift(-1))
        round_boundaries = job_history_df.loc[round_change_mask, ['DurchlaufNr', 'Runde']].to_dict('records')

    # Grafik erstellen
    fig, ax = plt.subplots(figsize=(10, 6))


    numeric_columns = plot_AFZ_Tabelle.reindex(columns=Eingabe.taetigkeiten_liste).apply(
        pd.to_numeric, errors='coerce'
    )
    has_curve_data = (
        not plot_AFZ_Tabelle.empty
        and not plot_AFZ_Tabelle.get('DurchlaufNr', pd.Series(dtype=float)).dropna().empty
        and numeric_columns.notna().any().any()
    )

    
    basis_taetigkeitsfarben = TAETIGKEITEN_FARBEN_BASIS or generiere_farben(
        len(Eingabe.taetigkeiten_liste)
    )
    farben = [
        _farbe_fuer_taetigkeit(t, basis_taetigkeitsfarben, idx)
        for idx, t in enumerate(Eingabe.taetigkeiten_liste)
    ]

    if has_curve_data:
        for i, t in enumerate(Eingabe.taetigkeiten_liste):
            ax.plot(
                plot_AFZ_Tabelle['DurchlaufNr'],
                plot_AFZ_Tabelle[t],
                label=t,
                color=farben[i % len(farben)] if farben else None,
                linestyle="-"
            )

        boundary_label_added = False
        for boundary in round_boundaries:
            x_position = boundary['DurchlaufNr'] + 0.5  # Linie zwischen den Durchläufen platzieren
            label = 'Rundenende' if not boundary_label_added else None
            ax.axvline(
                x=x_position,
                color='#676767',
                linestyle='--',
                linewidth=1,
                alpha=0.6,
                label=label,
            )
            boundary_label_added = True
    else:
        ax.text(
            0.5,
            0.5,
            'Keine Ausführungsdaten vorhanden',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )


    # Titel und Achsenbeschriftungen hinzufügen
    ax.set_title(f"Lern- und Vergessensverhalten der Tätigkeiten ({mitarbeiter_id})")
    ax.set_xlabel('Ausführungsanzahl')
    ax.set_ylabel('Ausführungzeit [s]')

    if has_curve_data:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True)
        max_durchlauf = pd.to_numeric(
            plot_AFZ_Tabelle.get('DurchlaufNr'), errors='coerce'
        ).dropna()
        if not max_durchlauf.empty:
            max_wert = int(math.ceil(max_durchlauf.max()))
            if max_wert <= 0:
                ax.set_xlim(left=0, right=1)
                ax.set_xticks([0])
            else:
                obergrenze = max(500, int(math.ceil(max_wert / 500.0) * 500))
                ax.set_xlim(left=0, right=obergrenze)
                xticks = list(range(0, obergrenze + 1, 500))
                if not xticks:
                    xticks = [0]
                ax.set_xticks(xticks)
        else:
            ax.set_xticks([0])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    plot_filename = f"Lern- und Vergessensverhalten_{mitarbeiter_id}.svg"
 
    fig.savefig(plot_filename, dpi=300)  # Speichert die Grafik im SVG-Format mit einer Auflösung von 300 DPI
    plt.close(fig)

    # Einzelgrafiken für jede Tätigkeit des Mitarbeitenden erzeugen
    for index, taetigkeit in enumerate(Eingabe.taetigkeiten_liste):
        taetigkeits_df = mitarbeitenden_details.get(
            taetigkeit,
            pd.DataFrame(
                columns=[
                    "DurchlaufNr",
                    "AFZ",
                    "AFA_pre",
                    "AFZ_post",
                    "AFA_post",
                    "VG_Dauer",
                    "Ausgefuehrt",
                    "Sim_zeit",
                ]
            ),
        )

        if taetigkeits_df.empty:
            continue

        taetigkeits_df = taetigkeits_df.copy()
        taetigkeits_df["DurchlaufNr"] = pd.to_numeric(
            taetigkeits_df["DurchlaufNr"], errors="coerce"
        )
        taetigkeits_df = taetigkeits_df.dropna(subset=["DurchlaufNr"])
        taetigkeits_df = taetigkeits_df.sort_values("DurchlaufNr").reset_index(drop=True)

        ausgefuehrt_df = taetigkeits_df.copy()
        if "Ausgefuehrt" in taetigkeits_df.columns:
            taetigkeits_df["Ausgefuehrt"] = pd.to_numeric(
                taetigkeits_df["Ausgefuehrt"], errors="coerce"
            )
            ausgefuehrt_df = taetigkeits_df[taetigkeits_df["Ausgefuehrt"] == 1].copy()
        else:
            ausgefuehrt_df = taetigkeits_df.copy()

        if taetigkeit not in plot_AFZ_Tabelle.columns:
            continue

        timeline_df = plot_AFZ_Tabelle[["DurchlaufNr", taetigkeit]].copy()
        timeline_df["DurchlaufNr"] = pd.to_numeric(
            timeline_df["DurchlaufNr"], errors="coerce"
        )
        timeline_df = timeline_df.dropna(subset=["DurchlaufNr"])
        timeline_df = timeline_df.sort_values("DurchlaufNr").reset_index(drop=True)
        timeline_df[taetigkeit] = timeline_df[taetigkeit].apply(
            pd.to_numeric, errors="coerce"
        )
        timeline_df[taetigkeit] = timeline_df[taetigkeit].ffill()
        timeline_df = timeline_df.dropna(subset=[taetigkeit])

        if timeline_df[taetigkeit].dropna().empty:
            continue

        afz_values = timeline_df[taetigkeit]
        fig, ax1 = plt.subplots(figsize=(10, 6))

        kurvenfarbe_basis = _farbe_fuer_taetigkeit(
            taetigkeit, farben, index
        )

        x_werte = timeline_df["DurchlaufNr"].to_numpy()
        ax1.plot(
            x_werte,
            afz_values,
            label="AFZ (vor Ausführung)",
            color=kurvenfarbe_basis,
            linestyle="-",
        )

        if not ausgefuehrt_df.empty:
            executed_points = ausgefuehrt_df[["DurchlaufNr", "AFZ"]].copy()
            executed_points["AFZ"] = executed_points["AFZ"].apply(
                pd.to_numeric, errors="coerce"
            )
            executed_points = executed_points.dropna(subset=["DurchlaufNr", "AFZ"])
            if not executed_points.empty:
                ax1.scatter(
                    executed_points["DurchlaufNr"],
                    executed_points["AFZ"],
                    color=kurvenfarbe_basis,
                    marker="o",
                    s=7,
                    label="Ausführung",
                )


        ax1.set_title(
            f"Lern- und Vergessensverhalten ({mitarbeiter_id} - {taetigkeit})"
        )
        ax1.set_xlabel("Ausführungsanzahl")
        ax1.set_ylabel("Ausführungszeit [s]")
        ax1.grid(True)

        min_x = 0
        max_x = 0
        anzeige_max = min_x
        if len(x_werte) > 0:
            max_x = int(math.ceil(float(np.nanmax(x_werte))))
            if max_x <= min_x:
                ax1.set_xlim(left=min_x, right=min_x + 1)
                ax1.set_xticks([min_x])
            else:
                obergrenze = max(500, int(math.ceil(max_x / 500.0) * 500))
                anzeige_max = obergrenze
                ax1.set_xlim(left=min_x, right=obergrenze)
                xticks = list(range(min_x, obergrenze + 1, 500))
                if not xticks:
                    xticks = [min_x]
                ax1.set_xticks(xticks)

        if max_x > min_x and anzeige_max == min_x:
            anzeige_max = max_x

        if round_boundaries:
            boundary_label_added = False
            for boundary in round_boundaries:
                x_position = boundary["DurchlaufNr"] + 0.5
                if x_position < min_x or x_position > anzeige_max + 1:
                    continue
                label = "Rundenende" if not boundary_label_added else None
                ax1.axvline(
                    x=x_position,
                    color="#000000",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    label=label,
                )
                boundary_label_added = True


        lines_1, labels_1 = ax1.get_legend_handles_labels()
        if lines_1:
            ax1.legend(lines_1, labels_1, loc="best")

        einzel_plot_filename = (
            f"Lern- und Vergessensverhalten_{mitarbeiter_id}_{taetigkeit}.svg"
        )
        fig.savefig(einzel_plot_filename, dpi=300)
        plt.close(fig)


    # Kompetenzentwicklung über die Arbeitstage visualisieren
    kompetenz_records = Berechnung.kompetenz_protokoll.get(mitarbeiter_id, [])
    kompetenz_df = pd.DataFrame(kompetenz_records)

    if not kompetenz_df.empty:
        kompetenz_df = kompetenz_df.copy()
        if "Ereignis" not in kompetenz_df.columns:
            kompetenz_df["Ereignis"] = "Feierabend"
        kompetenz_df["Ereignis"] = (
            kompetenz_df["Ereignis"].fillna("Feierabend").astype(str)
        )
        kompetenz_df["Ereignis_Bereinigt"] = kompetenz_df["Ereignis"].str.strip()
        erlaubte_ereignisse = {"feierabend", "wochenende"}
        kompetenz_df = kompetenz_df[
            kompetenz_df["Ereignis_Bereinigt"].str.lower().isin(erlaubte_ereignisse)
        ]
        kompetenz_df["Tag"] = pd.to_numeric(kompetenz_df["Tag"], errors="coerce")
        kompetenz_df = kompetenz_df.dropna(subset=["Tag"])
        kompetenz_df["Tag"] = kompetenz_df["Tag"].astype(int)
        if kompetenz_df.empty:
            kompetenz_df = pd.DataFrame()
        else:
            kompetenz_df = kompetenz_df.reset_index(drop=True)
            ereignis_sort_map = {"feierabend": 0, "wochenende": 1}
            kompetenz_df["Ereignis_Sort"] = (
                kompetenz_df["Ereignis_Bereinigt"].str.lower().map(ereignis_sort_map).fillna(0).astype(int)
            )
            kompetenz_df = kompetenz_df.sort_values(["Tag", "Ereignis_Sort"])
            kompetenz_df["Ereignis_Bereinigt_Lower"] = kompetenz_df["Ereignis_Bereinigt"].str.lower()
            ereignis_order = (
                kompetenz_df[["Tag", "Ereignis_Sort", "Ereignis_Bereinigt_Lower"]]
                .drop_duplicates()
                .sort_values(["Tag", "Ereignis_Sort"])
            )
            ereignis_order["Ereignis_Reihenfolge"] = (
                ereignis_order.groupby("Tag").cumcount()
            )
            kompetenz_df = kompetenz_df.merge(
                ereignis_order[["Tag", "Ereignis_Bereinigt_Lower", "Ereignis_Reihenfolge"]],
                on=["Tag", "Ereignis_Bereinigt_Lower"],
                how="left",
            )

            def _labeliere_ereignis(row):
                basis = f"Tag {int(row['Tag'])}"
                ereignis = row["Ereignis_Bereinigt"].strip().lower()
                if ereignis == "wochenende":
                    return f"(Wochenende)"
                if row["Ereignis_Reihenfolge"] > 0:
                    return f"{basis} (Ereignis {int(row['Ereignis_Reihenfolge']) + 1})"
                return basis

            kompetenz_df["Tag_Ereignis"] = kompetenz_df.apply(_labeliere_ereignis, axis=1)
            tag_event_order = (
                kompetenz_df[["Tag_Ereignis", "Tag", "Ereignis_Sort"]]
                .drop_duplicates()
                .sort_values(["Tag", "Ereignis_Sort", "Tag_Ereignis"])
            )

            kompetenz_pivot = (
                kompetenz_df.pivot_table(
                    index="Tag_Ereignis",
                    columns="Tätigkeit",
                    values="Kompetenzstufe",
                    aggfunc="last",
                )
                .reindex(tag_event_order["Tag_Ereignis"])
            )
            kompetenz_pivot = kompetenz_pivot.dropna(how="all")

        if not kompetenz_pivot.empty:
            spaltenreihenfolge = [
                taetigkeit
                for taetigkeit in Eingabe.taetigkeiten_liste
                if taetigkeit in kompetenz_pivot.columns
            ]
            spaltenreihenfolge += [
                spalte
                for spalte in kompetenz_pivot.columns
                if spalte not in spaltenreihenfolge
            ]
            kompetenz_pivot = kompetenz_pivot.reindex(columns=spaltenreihenfolge)

        if not kompetenz_pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            basis_farben = TAETIGKEITEN_FARBEN_BASIS or generiere_farben(
                kompetenz_pivot.shape[1]
            )
            kompetenz_farben = [
                _farbe_fuer_taetigkeit(
                    column,
                    basis_farben,
                    idx,
                )
                for idx, column in enumerate(kompetenz_pivot.columns)
            ]
            markerzyklus = cycle(["o", "^", "s", "D", "v", "*"])

            x_positionen = np.arange(len(kompetenz_pivot.index))

            for idx, column in enumerate(kompetenz_pivot.columns):
                ax.plot(
                    x_positionen,
                    kompetenz_pivot[column].to_numpy(),
                    marker=next(markerzyklus),
                    linestyle='-',
                    color=kompetenz_farben[idx % len(kompetenz_farben)] if kompetenz_farben else None,
                    label=column,
                )

            ax.set_title(f"Kompetenzentwicklung nach Arbeitstagen ({mitarbeiter_id})")
            ax.set_xlabel('Arbeitstag')
            ax.set_ylabel('Kompetenzstufe (Reduktionsgrad)')
            ax.set_xticks(x_positionen)
            ax.set_xticklabels(kompetenz_pivot.index, rotation=45, ha='right')
            ax.set_ylim(1, 5)
            ax.set_yticks(range(1, 6))
            ax.set_yticklabels(_ermittle_kompetenz_ticklabels())
            ax.grid(True, axis='both', linestyle='--', alpha=0.3)
            ax.legend(title='Tätigkeit (Markerformen wiederholen sich bei Bedarf)', loc='best')

            fig.tight_layout()
            fig.savefig(f"Kompetenzentwicklung_{mitarbeiter_id}.svg", dpi=300)
            plt.close(fig)


    # Produktivitätsdarstellung pro Arbeitstag und Runde
    produktivitaet_df = job_history_df.copy()
    if not produktivitaet_df.empty and {'Tag', 'Runde', 'Job'}.issubset(produktivitaet_df.columns):
        produktivitaet_df['Tag'] = pd.to_numeric(produktivitaet_df['Tag'], errors='coerce')
        produktivitaet_df['Runde'] = pd.to_numeric(produktivitaet_df['Runde'], errors='coerce')
        produktivitaet_df = produktivitaet_df.dropna(subset=['Tag', 'Runde'])
        produktivitaet_df['Tag'] = produktivitaet_df['Tag'].astype(int)
        produktivitaet_df['Runde'] = produktivitaet_df['Runde'].astype(int)

        for spalte in ["Output_geplant", "Output_gut", "Output_gut_ohne_Fehler"]:
            if spalte not in produktivitaet_df.columns:
                produktivitaet_df[spalte] = 0.0
            else:
                produktivitaet_df[spalte] = pd.to_numeric(
                    produktivitaet_df[spalte], errors="coerce"
                ).fillna(0.0)

        if JOB_FILTER != 'alle':
            produktivitaet_df = produktivitaet_df[produktivitaet_df['Job'] == JOB_FILTER]

        if not produktivitaet_df.empty:
            gesamt_produktivitaet_rows.append(
                produktivitaet_df[
                    ['Tag', 'Runde', 'Job', 'Output_gut', 'Output_gut_ohne_Fehler']
                ].assign(Mitarbeiter=mitarbeiter_id)
            )

        produktivitaet_ohne = produktivitaet_df.pivot_table(
            index=['Tag', 'Runde'],
            columns='Job',
            values='Output_gut_ohne_Fehler',
            aggfunc='sum',
            fill_value=0.0,
        )
        produktivitaet_mit = produktivitaet_df.pivot_table(
            index=['Tag', 'Runde'],
            columns='Job',
            values='Output_gut',
            aggfunc='sum',
            fill_value=0.0,
        )

        if not produktivitaet_ohne.empty or not produktivitaet_mit.empty:
            alle_index = produktivitaet_ohne.index.union(produktivitaet_mit.index).sort_values()
            produktivitaet_ohne = produktivitaet_ohne.reindex(alle_index, fill_value=0.0)
            produktivitaet_mit = produktivitaet_mit.reindex(alle_index, fill_value=0.0)

            job_spalten = list(produktivitaet_ohne.columns.union(produktivitaet_mit.columns))
            job_order = [job for job in Eingabe.jobs if job in job_spalten]
            job_order += [job for job in job_spalten if job not in job_order]
            produktivitaet_ohne = produktivitaet_ohne.reindex(columns=job_order, fill_value=0.0)
            produktivitaet_mit = produktivitaet_mit.reindex(columns=job_order, fill_value=0.0)

            labels = [f"Tag {tag} - Runde {runde}" for tag, runde in produktivitaet_ohne.index]
            x_positionen = np.arange(len(labels), dtype=float)
            bar_breite = 0.38

            fig, ax = plt.subplots(figsize=(12, 6))
            job_farben = generiere_farben(len(job_order))
            if not job_farben:
                job_farben = ["#4C4C4C"] * len(job_order)

            unterkanten_ohne = np.zeros(len(labels))
            unterkanten_mit = np.zeros(len(labels))

            for idx, job in enumerate(job_order):
                farbe = job_farben[idx % len(job_farben)]
                werte_ohne = produktivitaet_ohne[job].to_numpy(dtype=float, copy=False)
                werte_mit = produktivitaet_mit[job].to_numpy(dtype=float, copy=False)

                ax.bar(
                    x_positionen - bar_breite / 2,
                    werte_ohne,
                    width=bar_breite,
                    bottom=unterkanten_ohne,
                    color=farbe,
                    edgecolor='black',
                    linewidth=0.6,
                )
                ax.bar(
                    x_positionen + bar_breite / 2,
                    werte_mit,
                    width=bar_breite,
                    bottom=unterkanten_mit,
                    color=farbe,
                    edgecolor='black',
                    linewidth=0.6,
                    hatch='//',
                )

                unterkanten_ohne += werte_ohne
                unterkanten_mit += werte_mit

            ax.set_title(f"Produktivität pro Runde und Arbeitstag ({mitarbeiter_id})")
            ax.set_xlabel('Schichten (Tag - Runde)')
            ax.set_ylabel('Output (Stück)')
            ax.set_xticks(x_positionen)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            max_werte: List[float] = []
            if unterkanten_ohne.size:
                max_werte.append(float(np.nanmax(unterkanten_ohne)))
            if unterkanten_mit.size:
                max_werte.append(float(np.nanmax(unterkanten_mit)))
            if max_werte:
                max_hoehe = max(max_werte)
                if np.isfinite(max_hoehe):
                    if max_hoehe <= 0:
                        ax.set_ylim(0, 1)
                    else:
                        ax.set_ylim(0, max_hoehe * 1.05)

            job_handles = [
                Patch(facecolor=job_farben[idx % len(job_farben)], label=job)
                for idx, job in enumerate(job_order)
            ]
            if job_handles:
                leg_jobs = ax.legend(handles=job_handles, title='Job', loc='upper left')
                ax.add_artist(leg_jobs)

            muster_handles = [
                Patch(facecolor='#B2B2B2', edgecolor='black', label='Produktivität ohne Fehlerquote'),
                Patch(
                    facecolor='#B2B2B2',
                    edgecolor='black',
                    hatch='//',
                    label='Produktivität mit Fehlerquote',
                ),
            ]
            ax.legend(handles=muster_handles, loc='upper right')

            fig.tight_layout()
            fig.savefig(f"Produktivitaet_{mitarbeiter_id}.svg", dpi=300)
            plt.close(fig)

            produktivitaet_pro_tag_ohne = produktivitaet_df.pivot_table(
                index='Tag',
                columns='Runde',
                values='Output_gut_ohne_Fehler',
                aggfunc='sum',
                fill_value=0.0,
            )
            produktivitaet_pro_tag_mit = produktivitaet_df.pivot_table(
                index='Tag',
                columns='Runde',
                values='Output_gut',
                aggfunc='sum',
                fill_value=0.0,
            )

            if not produktivitaet_pro_tag_ohne.empty or not produktivitaet_pro_tag_mit.empty:
                tage_index = produktivitaet_pro_tag_ohne.index.union(
                    produktivitaet_pro_tag_mit.index
                ).sort_values()
                produktivitaet_pro_tag_ohne = produktivitaet_pro_tag_ohne.reindex(
                    tage_index, fill_value=0.0
                )
                produktivitaet_pro_tag_mit = produktivitaet_pro_tag_mit.reindex(
                    tage_index, fill_value=0.0
                )

                runden_spalten = list(
                    produktivitaet_pro_tag_ohne.columns.union(
                        produktivitaet_pro_tag_mit.columns
                    )
                )
                runden_sortiert = sorted(runden_spalten)
                produktivitaet_pro_tag_ohne = produktivitaet_pro_tag_ohne.reindex(
                    columns=runden_sortiert, fill_value=0.0
                )
                produktivitaet_pro_tag_mit = produktivitaet_pro_tag_mit.reindex(
                    columns=runden_sortiert, fill_value=0.0
                )

                tage = produktivitaet_pro_tag_ohne.index.to_list()
                x_positionen_tage = np.arange(len(tage), dtype=float)
                runden_farben = generiere_farben(len(runden_sortiert))
                if not runden_farben:
                    runden_farben = ["#4C4C4C"] * len(runden_sortiert)

                unterkanten_tag_ohne = np.zeros(len(tage))
                unterkanten_tag_mit = np.zeros(len(tage))

                fig, ax = plt.subplots(figsize=(12, 6))

                for idx, runde in enumerate(runden_sortiert):
                    farbe = runden_farben[idx % len(runden_farben)]
                    werte_ohne = produktivitaet_pro_tag_ohne[runde].to_numpy(
                        dtype=float, copy=False
                    )
                    werte_mit = produktivitaet_pro_tag_mit[runde].to_numpy(
                        dtype=float, copy=False
                    )

                    ax.bar(
                        x_positionen_tage - bar_breite / 2,
                        werte_ohne,
                        width=bar_breite,
                        bottom=unterkanten_tag_ohne,
                        color=farbe,
                        edgecolor='black',
                        linewidth=0.6,
                    )
                    ax.bar(
                        x_positionen_tage + bar_breite / 2,
                        werte_mit,
                        width=bar_breite,
                        bottom=unterkanten_tag_mit,
                        color=farbe,
                        edgecolor='black',
                        linewidth=0.6,
                        hatch='//',
                    )

                    unterkanten_tag_ohne += werte_ohne
                    unterkanten_tag_mit += werte_mit

                ax.set_title(
                    f"Produktivität pro Arbeitstag mit und ohne Fehlerquote ({mitarbeiter_id})"
                )
                ax.set_xlabel('Arbeitstag')
                ax.set_ylabel('Output (Stück)')
                ax.set_xticks(x_positionen_tage)
                ax.set_xticklabels([f"Tag {tag}" for tag in tage])
                ax.grid(True, axis='y', linestyle='--', alpha=0.3)

                max_werte_tag: List[float] = []
                if unterkanten_tag_ohne.size:
                    max_werte_tag.append(float(np.nanmax(unterkanten_tag_ohne)))
                if unterkanten_tag_mit.size:
                    max_werte_tag.append(float(np.nanmax(unterkanten_tag_mit)))
                if max_werte_tag:
                    max_hoehe_tag = max(max_werte_tag)
                    if np.isfinite(max_hoehe_tag):
                        if max_hoehe_tag <= 0:
                            ax.set_ylim(0, 1)
                        else:
                            ax.set_ylim(0, max_hoehe_tag * 1.05)

                runden_handles = [
                    Patch(facecolor=runden_farben[idx % len(runden_farben)], label=f"Runde {runde}")
                    for idx, runde in enumerate(runden_sortiert)
                ]
                if runden_handles:
                    leg_runden = ax.legend(handles=runden_handles, title='Runde', loc='upper left')
                    ax.add_artist(leg_runden)

                ax.legend(handles=muster_handles, loc='upper right')

                fig.tight_layout()
                fig.savefig(f"Produktivitaet_Tag_{mitarbeiter_id}.svg", dpi=300)
                plt.close(fig)

# Gesamtproduktivität über alle betrachteten Mitarbeitenden visualisieren
if gesamt_produktivitaet_rows:
    gesamt_produktivitaet_df = pd.concat(gesamt_produktivitaet_rows, ignore_index=True)

    gesamt_produktivitaet_df['Tag'] = pd.to_numeric(
        gesamt_produktivitaet_df['Tag'], errors='coerce'
    )
    gesamt_produktivitaet_df['Runde'] = pd.to_numeric(
        gesamt_produktivitaet_df['Runde'], errors='coerce'
    )
    gesamt_produktivitaet_df = gesamt_produktivitaet_df.dropna(subset=['Tag', 'Runde'])
    gesamt_produktivitaet_df['Tag'] = gesamt_produktivitaet_df['Tag'].astype(int)
    gesamt_produktivitaet_df['Runde'] = gesamt_produktivitaet_df['Runde'].astype(int)

    if JOB_FILTER != 'alle':
        gesamt_produktivitaet_df = gesamt_produktivitaet_df[
            gesamt_produktivitaet_df['Job'] == JOB_FILTER
        ]

    for spalte in ['Output_gut', 'Output_gut_ohne_Fehler']:
        gesamt_produktivitaet_df[spalte] = pd.to_numeric(
            gesamt_produktivitaet_df.get(spalte), errors='coerce'
        ).fillna(0.0)

    produktivitaet_pro_tag_ohne = gesamt_produktivitaet_df.pivot_table(
        index='Tag',
        columns='Runde',
        values='Output_gut_ohne_Fehler',
        aggfunc='sum',
        fill_value=0.0,
    )
    produktivitaet_pro_tag_mit = gesamt_produktivitaet_df.pivot_table(
        index='Tag',
        columns='Runde',
        values='Output_gut',
        aggfunc='sum',
        fill_value=0.0,
    )

    if not produktivitaet_pro_tag_ohne.empty or not produktivitaet_pro_tag_mit.empty:
        tage_index = produktivitaet_pro_tag_ohne.index.union(
            produktivitaet_pro_tag_mit.index
        ).sort_values()
        produktivitaet_pro_tag_ohne = produktivitaet_pro_tag_ohne.reindex(
            tage_index, fill_value=0.0
        )
        produktivitaet_pro_tag_mit = produktivitaet_pro_tag_mit.reindex(
            tage_index, fill_value=0.0
        )

        runden_spalten = list(
            produktivitaet_pro_tag_ohne.columns.union(
                produktivitaet_pro_tag_mit.columns
            )
        )
        runden_sortiert = sorted(runden_spalten)
        produktivitaet_pro_tag_ohne = produktivitaet_pro_tag_ohne.reindex(
            columns=runden_sortiert, fill_value=0.0
        )
        produktivitaet_pro_tag_mit = produktivitaet_pro_tag_mit.reindex(
            columns=runden_sortiert, fill_value=0.0
        )

        tage = produktivitaet_pro_tag_ohne.index.to_list()
        x_positionen_tage = np.arange(len(tage), dtype=float)
        runden_farben = generiere_farben(
            len(runden_sortiert), schema=FARBSCHEMA_GESAMT
        )
        if not runden_farben:
            runden_farben = ["#4C4C4C"] * len(runden_sortiert)

        unterkanten_tag_ohne = np.zeros(len(tage))
        unterkanten_tag_mit = np.zeros(len(tage))

        bar_breite = 0.38
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, runde in enumerate(runden_sortiert):
            farbe = runden_farben[idx % len(runden_farben)]
            werte_ohne = produktivitaet_pro_tag_ohne[runde].to_numpy(
                dtype=float, copy=False
            )
            werte_mit = produktivitaet_pro_tag_mit[runde].to_numpy(
                dtype=float, copy=False
            )

            ax.bar(
                x_positionen_tage - bar_breite / 2,
                werte_ohne,
                width=bar_breite,
                bottom=unterkanten_tag_ohne,
                color=farbe,
                edgecolor='black',
                linewidth=0.6,
            )
            ax.bar(
                x_positionen_tage + bar_breite / 2,
                werte_mit,
                width=bar_breite,
                bottom=unterkanten_tag_mit,
                color=farbe,
                edgecolor='black',
                linewidth=0.6,
                hatch='//',
            )

            unterkanten_tag_ohne += werte_ohne
            unterkanten_tag_mit += werte_mit

        ax.set_title(
            'Gesamtproduktivität pro Arbeitstag mit und ohne Fehlerquote'
        )
        ax.set_xlabel('Arbeitstag')
        ax.set_ylabel('Output (Stück)')
        ax.set_xticks(x_positionen_tage)
        ax.set_xticklabels([f"Tag {tag}" for tag in tage])
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        max_werte_gesamt: List[float] = []
        if unterkanten_tag_ohne.size:
            max_werte_gesamt.append(float(np.nanmax(unterkanten_tag_ohne)))
        if unterkanten_tag_mit.size:
            max_werte_gesamt.append(float(np.nanmax(unterkanten_tag_mit)))
        if max_werte_gesamt:
            max_hoehe_gesamt = max(max_werte_gesamt)
            if np.isfinite(max_hoehe_gesamt):
                if max_hoehe_gesamt <= 0:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(0, max_hoehe_gesamt * 1.05)

        runden_handles = [
            Patch(facecolor=runden_farben[idx % len(runden_farben)], label=f"Runde {runde}")
            for idx, runde in enumerate(runden_sortiert)
        ]
        if runden_handles:
            leg_runden = ax.legend(handles=runden_handles, title='Runde', loc='upper left')
            ax.add_artist(leg_runden)

        muster_handles = [
            Patch(facecolor='#B2B2B2', edgecolor='black', label='Produktivität ohne Fehlerquote'),
            Patch(
                facecolor='#B2B2B2',
                edgecolor='black',
                hatch='//',
                label='Produktivität mit Fehlerquote',
            ),
        ]
        ax.legend(handles=muster_handles, loc='upper right')

        fig.tight_layout()
        fig.savefig('Produktivitaet_Tag_Gesamt.svg', dpi=300)
        plt.close(fig)