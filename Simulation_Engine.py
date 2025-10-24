"""Simulations-Engine mit Protokollierungs- und Ereignisfunktionen."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from statistics import mean, pvariance
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set

import pandas as pd

from Simulation_Lernkurve import Lernkurve
from Simulation_Vergessenskurve import Vergessenskurve


class SimulationLogger(Protocol):
    """ Von der Simulations-Engine verwendetes Protokoll mit minimaler Protokollierung """

    def info(self, message: str) -> None:
        """ Informationsmeldung protokollieren """


class PrintLogger:
    """ Rückfall-Logger, der Nachrichten an :func:`print` weiterleitet """

    def info(self, message: str) -> None:  
        print(message)


def berechne_outputsumme_fuer_tag(
    job_history: Dict[str, Iterable[Dict[str, Any]]], tag: int
) -> Optional[float]:
    """ Summiere den *Output_gut* aller Tätigkeiten für einen Arbeitstag """

    gesamt = 0.0
    werte_gefunden = False
    for historie in job_history.values():
        for eintrag in historie:
            try:
                eintrag_tag = int(float(eintrag.get("Tag")))
                int(float(eintrag.get("Runde")))
            except (TypeError, ValueError):
                continue
            if eintrag_tag != tag:
                continue
            wert = eintrag.get("Output_gut")
            try:
                if wert is not None:
                    gesamt += float(wert)
                    werte_gefunden = True
            except (TypeError, ValueError):
                continue
    if not werte_gefunden:
        return None
    return gesamt


@dataclass
class Simulationsergebnis:
    """ Container für die rohen Simulationsergebnisse """

    output_data_all: Dict[str, Dict[str, pd.DataFrame]]
    job_history: Dict[str, List[Dict[str, Any]]]
    kompetenz_protokoll: Dict[str, List[Dict[str, Any]]]
    ziel_status: Optional[Dict[str, Any]] = None
    letzter_simulationstag: Optional[int] = None
    uebungsfaktor_protokoll: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )



BeforeRoundHandler = Callable[["SimulationRunner", int, Dict[str, Any]], None]
AfterDayHandler = Callable[["SimulationRunner", int], None]


@dataclass
class MitarbeiterProfil:
    """ Aggregierter Parametersatz für den aktiven Mitarbeitenden """

    name: str
    t_initiale_AFZ: Dict[str, float]
    vor_AFA: Dict[str, float]
    gw: Dict[str, float]
    lf: Dict[str, float]
    vf: Dict[str, float]
    lernkurve: Lernkurve
    vergessenskurve: Vergessenskurve


@dataclass
class MitarbeiterStateSnapshot:
    """ Speichern der Kompetenz- und Lernzustände für eine Person, falls sie ausfällt """

    AFA_pre: Dict[str, float]
    AFA_post: Dict[str, float]
    AFZ: Dict[str, float]
    AFZ_post: Dict[str, float]
    AFZ_last: Dict[str, float]
    vergessensdauer: Dict[str, float]
    ausgefuehrt: Dict[str, int]
    letzte_iteration: Optional[int]
    m_abbau: Dict[str, float]
    m_aufbau: Dict[str, float]
    m_break_stunden: float
    m_tag_index: Optional[int]
    m_tag_label: Optional[str]
    m_gesamt_wiederholungen: Dict[str, int]


@dataclass(order=True)
class Personalereignis:
    """Planbarer Personalvorfall (Fluktuation oder Ausfall)"""

    start_tag: int
    typ: str = field(compare=False)
    mitarbeiter: str = field(compare=False)
    dauer: Optional[int] = field(default=None, compare=False)


@dataclass
class PersonalStatus:
    """ Verwaltet die Personalplanung für eine Rolle """

    original: str
    ereignisse: List[Personalereignis] = field(default_factory=list)
    ersatz_index: int = 0
    ersatz_aktiv: bool = False
    aktives_ereignis: Optional[Personalereignis] = None
    verbleibende_ausfalltage: int = 0
    rueckkehr_tag: Optional[int] = None
    abwesenheit_sekunden: float = 0.0
    original_snapshot: Optional[MitarbeiterStateSnapshot] = None
    letzter_gepruefter_tag: int = 0
    ersatz_label: Optional[str] = None


@dataclass
class StoerungsEreignis:
    """ Geplante Produktionsunterbrechung für Jobs und Runden """

    start_tag: int
    end_tag: int
    runden: Optional[Set[int]]
    jobs: Optional[Set[str]]
    beschreibung: Optional[str] = None

''' Aufbau der Grundkonfiguration '''
class SimulationRunner:
    """ Klasse für Ausführung der Kompetenz- und Durchsatzsimulation, enthält alle notwendigen Funktionen """

    FEHLERARM_FEHLERQUOTE = 0.02
    FEHLERARM_MIN_REDUKTION = 0.85

    def __init__( # Initialisierung der Simulation
        self,
        eingabe_module: Any,        # Daten aus Eingabe-Modul lesen und verarbeiten
        *,
        logger: Optional[SimulationLogger] = None,
        event_handlers: Optional[Dict[str, Iterable[Callable[..., None]]]] = None,
        personalrisiken: Optional[Dict[str, Iterable[Dict[str, Any]]]] = None,
        produktionsstoerungen: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        self.eingabe = eingabe_module
        self.logger = logger or PrintLogger()
        self.aktive_mitarbeitende = list(getattr(self.eingabe, "mitarbeitende", []))
        if not self.aktive_mitarbeitende:                                               # zunächst wird Eingabekonsistenz geprüft
            raise ValueError("Es müssen aktive Mitarbeitende definiert sein.")
        self.arbeitsrunden = list(getattr(self.eingabe, "arbeitsrunden", []))
        if not self.arbeitsrunden:
            raise ValueError("Es muss mindestens eine Arbeitsrunde definiert sein.")
        jobs_config_raw = getattr(self.eingabe, "jobs_zuordnung", {})
        jobs_config_norm: Dict[str, List[str]] = {}
        for job, tasks in jobs_config_raw.items():
            job_name = str(job)
            if isinstance(tasks, str):
                task_iterable = [tasks]
            else:
                try:
                    task_iterable = list(tasks)
                except TypeError:
                    task_iterable = [tasks]
            jobs_config_norm[job_name] = [str(t) for t in task_iterable if t is not None]

        alle_jobs = getattr(self.eingabe, "jobs", None)
        if alle_jobs is None or not alle_jobs:
            alle_jobs = list(jobs_config_norm.keys())
        else:
            alle_jobs = [str(job) for job in alle_jobs]
        for job in jobs_config_norm:
            if job not in alle_jobs:
                alle_jobs.append(job)
        self.alle_jobs = alle_jobs

        self.taetigkeiten = [
            str(t) for t in list(getattr(self.eingabe, "taetigkeiten_liste", []))
        ]
        if not self.taetigkeiten:
            raise ValueError("Es müssen Tätigkeiten für die Simulation definiert sein.")

        self.jobs_zuordnung_basis: Dict[str, List[str]] = {}
        self.jobs_zuordnung: Dict[str, List[str]] = {}
        for job in self.alle_jobs:
            tasks_raw = list(jobs_config_norm.get(job, []))
            filtered_tasks: List[str] = []
            for taetigkeit in tasks_raw:
                if taetigkeit not in self.taetigkeiten:
                    self.logger.info(
                        f"Tätigkeit {taetigkeit} ist nicht in der Tätigkeitenliste enthalten und wird für Job {job} ignoriert."
                    )
                    continue
                filtered_tasks.append(taetigkeit)
            self.jobs_zuordnung_basis[job] = list(filtered_tasks)
            self.jobs_zuordnung[job] = list(filtered_tasks)

        self.t_initiale_AFZ_map: Dict[str, Dict[str, float]] = {}
        self.vor_AFA_map: Dict[str, Dict[str, float]] = {}
        self.gw_map: Dict[str, Dict[str, float]] = {}
        self.lf_map: Dict[str, Dict[str, float]] = {}
        self.vf_map: Dict[str, Dict[str, float]] = {}
        self.lernkurven: Dict[str, Lernkurve] = {}
        self.vergessenskurven: Dict[str, Vergessenskurve] = {}

        lf_quelle = getattr(self.eingabe, "LF_MA", {})
        vf_quelle = getattr(self.eingabe, "VF_MA", {})
        gw_quelle = getattr(self.eingabe, "GW_MA", {})
        for ma in self.aktive_mitarbeitende:
            self.t_initiale_AFZ_map[ma] = dict(getattr(self.eingabe, "t_initiale_AFZ", {}).get(ma, {}))
            self.vor_AFA_map[ma] = dict(getattr(self.eingabe, "vor_AFA", {}).get(ma, {}))
            self.gw_map[ma] = dict(gw_quelle.get(ma, {}))
            self.lf_map[ma] = dict(lf_quelle.get(ma, {}))
            self.vf_map[ma] = dict(vf_quelle.get(ma, {}))
            self.lernkurven[ma] = getattr(self.eingabe, "lernkurve_mitarbeiter", {}).get(ma)
            self.vergessenskurven[ma] = getattr(self.eingabe, "vergessenskurve_mitarbeiter", {}).get(ma)
            if self.lernkurven[ma] is None or self.vergessenskurven[ma] is None:
                raise ValueError(f"Für Mitarbeiter {ma} fehlen Lern- oder Vergessenskurven.")

        komplexitaet_roh = getattr(self.eingabe, "taetigkeit_komplexitaet", {})         # ab hier werden Komplexitätswerte, der Normwert für die Ausfühurungsanzahl  
        self.taetigkeit_komplexitaet: Dict[str, float] = {}                             # und Übungsfaktor-Parameter aus der Eingabe gelesen und ggf. auf sinnvolle Grenzen gekappt
        for taetigkeit in self.taetigkeiten:
            try:
                wert = float(komplexitaet_roh.get(taetigkeit, 3.0))
            except (TypeError, ValueError):
                wert = 3.0
            if wert < 1.0:
                wert = 1.0
            if wert > 5.0:
                wert = 5.0
            self.taetigkeit_komplexitaet[taetigkeit] = wert

        m_cfg = getattr(self.eingabe, "uebungsfaktor_parameter", {})
        self.m_min = float(m_cfg.get("m_min", 1.0))
        self.m_max = float(m_cfg.get("m_max", 5.0))
        if self.m_max < self.m_min:
            self.m_max = self.m_min
        try:
            self.m_alpha = float(m_cfg.get("alpha", 1.0))
        except (TypeError, ValueError):
            self.m_alpha = 1.0
        if self.m_alpha < 0:
            self.m_alpha = 0.0
        try:
            self.m_n_norm = float(m_cfg.get("n_norm", 500.0))
        except (TypeError, ValueError):
            self.m_n_norm = 500.0
        if self.m_n_norm <= 0:
            self.m_n_norm = 500.0
        try:
            self.m_q_min = float(m_cfg.get("q_min_stunden", 1.0))
        except (TypeError, ValueError):
            self.m_q_min = 1.0
        if self.m_q_min <= 0:
            self.m_q_min = 1.0

        ziel_roh = getattr(self.eingabe, "simulationsziele", None)
        self._ziele_config = self._parse_simulationsziele(ziel_roh)
        self._ziele_aktiv = any(
            wert is not None
            for wert in (
                self._ziele_config.get("fehlerquote"),
                self._ziele_config.get("produktivitaet"),
                self._ziele_config.get("kompetenz_durchschnitt"),
                self._ziele_config.get("kompetenz_varianz"),
            )
        )
        betrachtungszeitraum = getattr(self.eingabe, "Betrachtungszeitraum", 0)
        try:
            betrachtungszeitraum_tage = float(betrachtungszeitraum)
        except (TypeError, ValueError):
            betrachtungszeitraum_tage = 0.0
        if betrachtungszeitraum_tage <= 0:
            betrachtungszeitraum_tage = 1.0

        ziel_max_tage = self._ziele_config.get("max_tage")
        if self._ziele_aktiv:
            if ziel_max_tage is not None and ziel_max_tage > 0:
                self.max_tage = ziel_max_tage
            else:
                self.max_tage = betrachtungszeitraum_tage
                self.logger.info(
                    "Simulationsziele gesetzt, aber keine gültige Konfiguration für 'max_tage' gefunden – "
                    "es wird der Betrachtungszeitraum als Fallback genutzt."
                )
        else:
            self.max_tage = betrachtungszeitraum_tage
            if ziel_max_tage is not None:
                self.logger.info(
                    "Konfiguration 'max_tage' wird ignoriert, da keine Simulationsziele gesetzt sind."
                )
                self._ziele_config["max_tage"] = None

        self.simulationszeit_limit = self.max_tage * 24 * 3600
        self._simulation_stop_tag: Optional[int] = None
        self._ziel_status: Optional[Dict[str, Any]] = None
        self._tageskennzahlen: Dict[int, Dict[str, float]] = {}
        self._letzter_simulationstag: Optional[int] = None
        self._ziele_pruefen_deaktiviert = False
        self._externes_tag_limit: Optional[int] = None

        self.logger.info(
            f"SimulationRunner initialisiert mit {len(self.aktive_mitarbeitende)} Mitarbeitenden, "
            f"{len(self.arbeitsrunden)} Runden und {self.max_tage:g} Tagen."
        )

        self._before_round_handlers: List[BeforeRoundHandler] = []
        self._after_day_handlers: List[AfterDayHandler] = []
        if event_handlers:
            for key, handlers in event_handlers.items():
                if key == "before_round":
                    for handler in handlers:
                        self._before_round_handlers.append(handler)  # type: ignore[arg-type]
                elif key == "after_day":
                    for handler in handlers:
                        self._after_day_handlers.append(handler)  # type: ignore[arg-type]

        self.original_profile: Dict[str, MitarbeiterProfil] = {}
        for ma in self.aktive_mitarbeitende:
            self.original_profile[ma] = self._erstelle_profil(
                ma,
                self.t_initiale_AFZ_map[ma],
                self.vor_AFA_map[ma],
                self.gw_map[ma],
                self.lf_map[ma],
                self.vf_map[ma],
                self.lernkurven[ma],
                self.vergessenskurven[ma],
            )

        self.profil_aktuell: Dict[str, MitarbeiterProfil] = {
            ma: self.original_profile[ma] for ma in self.aktive_mitarbeitende
        }

        self.kompetenz_parameter: Dict[str, Dict[str, Dict[str, float]]] = {}
        for ma in self.aktive_mitarbeitende:
            self._set_aktives_profil(ma, self.profil_aktuell[ma])

        self._ersatz_profil_vorlage = self._berechne_standardprofilvorlage()

        self._personalstatus: Dict[str, PersonalStatus] = {}
        if personalrisiken is None:
            personalrisiken = {}
        for ma in self.aktive_mitarbeitende:
            ereignisliste = []
            for key in ("ausfaelle", "ausfälle", "ausfall", "fluktuation", "fluktuationen"):
                for eintrag in personalrisiken.get(key, []):
                    if str(eintrag.get("mitarbeiter")) != ma:
                        continue
                    raw_start = eintrag.get("start_tag", eintrag.get("tag", 0))
                    try:
                        start_tag = int(raw_start)
                    except (TypeError, ValueError):
                        continue
                    if start_tag <= 0:
                        continue
                    typ = "fluktuation" if "fluktu" in key.lower() else "ausfall"
                    dauer = eintrag.get("dauer")
                    if dauer is not None:
                        try:
                            dauer = int(dauer)
                        except (TypeError, ValueError):
                            dauer = None
                    ereignisliste.append(
                        Personalereignis(
                            start_tag=start_tag,
                            typ=typ,
                            mitarbeiter=ma,
                            dauer=dauer,
                        )
                    )
            ereignisliste.sort()
            self._personalstatus[ma] = PersonalStatus(original=ma, ereignisse=ereignisliste)

        if produktionsstoerungen is None:      
            produktionsstoerungen = getattr(self.eingabe, "produktionsstoerungen", None)
        self._stoerungsplan: Dict[int, List[StoerungsEreignis]] = {}
        self._initialisiere_stoerungsplan(produktionsstoerungen)

        self._initialisiere_jobprofil_aenderungen()

        self.gesamt_arbeitszeit_pro_tag = float(
            getattr(
                self.eingabe,
                "arbeitszeit_pro_tag",
                sum(runde["arbeitszeit"] for runde in self.arbeitsrunden),
            )
        )
        self.gesamt_pausenzeit_pro_tag = float(
            getattr(
                self.eingabe,
                "gesamt_pausenzeit",
                sum(runde.get("pause", 0.0) for runde in self.arbeitsrunden),
            )
        )
        self.schichtdauer_pro_tag = (
            self.gesamt_arbeitszeit_pro_tag + self.gesamt_pausenzeit_pro_tag
        )

        self.zustand_arbeitszeit: Dict[str, float] = {}
        self.z1: Dict[str, int] = {}
        self.z2: Dict[str, int] = {}
        self.simulationszeit: Dict[str, float] = {}
        self.runde_index: Dict[str, int] = {}
        self.runde_restzeit: Dict[str, float] = {}
        self.job_task_index: Dict[str, Dict[str, int]] = {}
        self.letzte_iteration: Dict[str, Optional[int]] = {}
        self.arbeitstag_zaehler: Dict[str, int] = {}
        self.job_durchsatz: Dict[str, Dict[str, Optional[float]]] = {}
        self.AFA_pre: Dict[str, Dict[str, float]] = {}
        self.AFA_post: Dict[str, Dict[str, float]] = {}
        self.AFZ: Dict[str, Dict[str, float]] = {}
        self.AFZ_post: Dict[str, Dict[str, float]] = {}
        self.AFZ_last: Dict[str, Dict[str, float]] = {}
        self.vergessensdauer: Dict[str, Dict[str, float]] = {}
        self.ausgefuehrt: Dict[str, Dict[str, int]] = {}
        self.output_data_all: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.job_history: Dict[str, List[Dict[str, Any]]] = {}
        self.kompetenz_protokoll: Dict[str, List[Dict[str, Any]]] = {}
        self.durchlauf_index_person: Dict[str, int] = {}
        self.slot_label: Dict[str, str] = {}
        self._arbeitstag_beendet: Dict[str, bool] = {}
        self._fehlerfrei_flag = False
        self._runde_event_marker: Dict[str, set] = {}
        self._letzter_abgeschlossener_tag = 0
        self._output_spalten: List[str] = []
        self._ersatz_global_index = 0

        self._initialisiere_zustaende()

    def _initialisiere_stoerungsplan(           # überträgt Produktionsstörungen in einen kalenderbasierten Plan und speichert Ereignisse pro Tag
        self, stoerungen: Optional[Iterable[Dict[str, Any]]] 
    ) -> None:
        if not stoerungen:
            return
        for eintrag in stoerungen:
            if not isinstance(eintrag, dict):
                continue
            raw_start = eintrag.get("start_tag", eintrag.get("tag"))
            try:
                start_tag = int(raw_start)
            except (TypeError, ValueError):
                continue
            if start_tag <= 0:
                continue
            end_tag = None
            if "end_tag" in eintrag:
                try:
                    end_tag = int(eintrag.get("end_tag"))
                except (TypeError, ValueError):
                    end_tag = None
            if end_tag is None:
                raw_dauer = eintrag.get("dauer") or eintrag.get("dauer_tage") or 1
                try:
                    dauer_tage = int(raw_dauer)
                except (TypeError, ValueError):
                    dauer_tage = 1
                dauer_tage = max(dauer_tage, 1)
                end_tag = start_tag + dauer_tage - 1
            else:
                end_tag = max(end_tag, start_tag)

            runden_roh = (
                eintrag.get("runden")
                or eintrag.get("runde")
                or eintrag.get("rounds")
                or eintrag.get("round")
            )
            runden = self._normalisiere_runden(runden_roh)
            jobs_roh = (
                eintrag.get("jobs")
                or eintrag.get("job")
                or eintrag.get("betroffene_jobs")
            )
            jobs = self._normalisiere_jobs(jobs_roh)
            beschreibung = (
                eintrag.get("beschreibung")
                or eintrag.get("grund")
                or eintrag.get("name")
            )

            stoerung = StoerungsEreignis(
                start_tag=start_tag,
                end_tag=end_tag,
                runden=runden,
                jobs=jobs,
                beschreibung=str(beschreibung) if beschreibung else None,
            )
            for tag in range(stoerung.start_tag, stoerung.end_tag + 1):
                self._stoerungsplan.setdefault(tag, []).append(stoerung)

    def _initialisiere_jobprofil_aenderungen(self) -> None:     # liest geplante Jobprofiländerungen und 
        self._jobprofil_aenderungen: List[Dict[str, Any]] = []
        roh = getattr(self.eingabe, "jobprofil_aenderungen", None)
        if not roh:
            for job in self.alle_jobs:
                self.jobs_zuordnung.setdefault(
                    job, list(self.jobs_zuordnung_basis.get(job, []))
                )
            return

        for eintrag in roh:
            if not isinstance(eintrag, dict):
                continue
            raw_start = (
                eintrag.get("start_tag")
                or eintrag.get("tag")
                or eintrag.get("ab_tag")
                or eintrag.get("start")
            )
            try:
                start_tag = int(raw_start)
            except (TypeError, ValueError):
                continue
            if start_tag <= 0:
                continue
            jobs_roh = (
                eintrag.get("jobs")
                or eintrag.get("jobprofile")
                or eintrag.get("profile")
                or eintrag.get("zuordnungen")
            )
            if not isinstance(jobs_roh, dict):      
                continue
            jobs_map: Dict[str, List[str]] = {}
            for job, tasks in jobs_roh.items():
                job_name = str(job)
                if tasks is None:
                    task_iterable: List[Any] = []
                elif isinstance(tasks, str):
                    task_iterable = [tasks]
                else:
                    try:
                        task_iterable = list(tasks)
                    except TypeError:
                        task_iterable = [tasks]
                neue_tasks: List[str] = []
                for wert in task_iterable:
                    if wert is None:
                        continue
                    taetigkeit = str(wert)
                    if taetigkeit not in self.taetigkeiten: # für den Fall einer fehlerhaften Jobzuordnung
                        self.logger.info(
                            f"Jobprofil-Änderung Tag {start_tag}: Tätigkeit {taetigkeit} ist nicht in der Tätigkeitenliste enthalten und wird ignoriert." 
                        )
                        continue
                    neue_tasks.append(taetigkeit)
                jobs_map[job_name] = neue_tasks
                if job_name not in self.alle_jobs:
                    self.alle_jobs.append(job_name)
            if not jobs_map:
                continue
            beschreibung = (
                eintrag.get("beschreibung")
                or eintrag.get("name")
                or eintrag.get("grund")
            )
            self._jobprofil_aenderungen.append(
                {
                    "start_tag": start_tag,
                    "jobs": jobs_map,
                    "beschreibung": str(beschreibung) if beschreibung else None,
                    "_applied": False,
                }
            )

        for job in self.alle_jobs:
            self.jobs_zuordnung.setdefault(
                job, list(self.jobs_zuordnung_basis.get(job, []))
            )
            self.jobs_zuordnung_basis.setdefault(
                job, list(self.jobs_zuordnung.get(job, []))
            )

        self._jobprofil_aenderungen.sort(key=lambda eintrag: eintrag["start_tag"])

    def _parse_simulationsziele(self, konfig: Any) -> Dict[str, Optional[float]]:   # aus Eingabe-Modul werden Zielkonfiguration extrahiert
        result: Dict[str, Optional[float]] = {                                      # gilt nur, falls Schwellenwerte für Fehlerquote, Produktivität und Kompetenzen gesetzt sind
            "fehlerquote": None,
            "produktivitaet": None,
            "kompetenz_durchschnitt": None,
            "kompetenz_varianz": None,
            "max_tage": None,
        }
        if not konfig or not isinstance(konfig, dict):
            return result

        def parse_float(value: Any) -> Optional[float]: # heterogene Eingaben der Zieldefinitionen werden in Gleitkommazahlen überführt
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        fehler_entry = konfig.get("fehlerquote")
        if isinstance(fehler_entry, dict):
            for key in ("schwellwert", "max", "maximal", "ziel", "wert"):
                wert = parse_float(fehler_entry.get(key))
                if wert is not None:
                    result["fehlerquote"] = wert
                    break
        elif fehler_entry is not None:
            wert = parse_float(fehler_entry)
            if wert is not None:
                result["fehlerquote"] = wert
        if result["fehlerquote"] is None:
            for key in (
                "fehlerquote",
                "fehlerquote_durchschnitt",
                "max_fehlerquote",
                "ziel_fehlerquote",
            ):
                wert = parse_float(konfig.get(key))
                if wert is not None:
                    result["fehlerquote"] = wert
                    break

        produktiv_keys = (
            "produktivitaet",
            "produktivität",
            "produktivitaet_pro_tag",
            "produktivität_pro_tag",
            "produktivität_tag",
            "produktivitätsschwelle",
            "produktivitaetsschwelle",
            "output_pro_tag",
            "leistung",
        )
        produktiv_entry: Any = None
        for key in produktiv_keys:
            if key in konfig:
                produktiv_entry = konfig.get(key)
                break
        if isinstance(produktiv_entry, dict):
            for key in ("schwellwert", "min", "minimum", "ziel", "wert"):
                wert = parse_float(produktiv_entry.get(key))
                if wert is not None:
                    result["produktivitaet"] = wert
                    break
        else:
            wert = parse_float(produktiv_entry)
            if wert is not None:
                result["produktivitaet"] = wert

        kompetenz_keys = (
            "kompetenzziel",
            "kompetenz",
            "kompetenzniveau",
            "kompetenz_schwelle",
        )
        kompetenz_entry: Any = None
        for key in kompetenz_keys:
            if key in konfig:
                kompetenz_entry = konfig.get(key)
                break
        if isinstance(kompetenz_entry, dict):
            durchschnitt = None
            varianz = None
            for key in ("durchschnitt", "mittelwert", "ziel", "mean"):
                wert = parse_float(kompetenz_entry.get(key))
                if wert is not None:
                    durchschnitt = wert
                    break
            for key in ("varianz", "var", "varianz_max", "max_varianz", "streuung"):
                wert = parse_float(kompetenz_entry.get(key))
                if wert is not None:
                    varianz = wert
                    break
            result["kompetenz_durchschnitt"] = durchschnitt
            result["kompetenz_varianz"] = varianz
        elif kompetenz_entry is not None:
            durchschnitt = parse_float(kompetenz_entry)
            if durchschnitt is not None:
                result["kompetenz_durchschnitt"] = durchschnitt

        max_keys = (
            "max_tage",
            "maximale_tage",
            "maximale_betrachtungsdauer",
            "maximaler_betrachtungszeitraum",
            "max_betrachtungszeitraum",
            "maximaler_zeitraum",
        )
        max_entry: Any = None
        for key in max_keys:
            if key in konfig:
                max_entry = konfig.get(key)
                break
        wert = parse_float(max_entry)
        if wert is not None and wert > 0:
            result["max_tage"] = float(int(wert))

        return result

    def _hole_job_tasks(self, job: str) -> List[str]: # gibt die aktuelle Liste der Tätigkeiten zurück, die einem Job zugeordnet sind
        return list(self.jobs_zuordnung.get(job, []))

    def _pruefe_jobprofil_aenderungen(self, tag: int) -> None: # prüft am Tagesbeginn, ob ab diesem Tag Jobprofiländerungen gelten
        if tag <= 0:
            return
        for eintrag in self._jobprofil_aenderungen:
            if eintrag.get("_applied"):
                continue
            if tag >= eintrag["start_tag"]:
                self._wende_jobprofil_aenderung_an(tag, eintrag)
                eintrag["_applied"] = True

    def _wende_jobprofil_aenderung_an( # setzt neue Tätigkeitslisten für betroffene Jobs
        self, tag: int, eintrag: Dict[str, Any]
    ) -> None:
        beschreibung = eintrag.get("beschreibung")
        prefix = f"Jobprofil-Änderung ab Tag {eintrag['start_tag']}"
        if beschreibung:
            prefix += f" ({beschreibung})"
        for job, neue_tasks in eintrag.get("jobs", {}).items():
            if job not in self.jobs_zuordnung:
                self.jobs_zuordnung[job] = []
            if job not in self.jobs_zuordnung_basis:
                self.jobs_zuordnung_basis[job] = []
            alte_tasks = list(self.jobs_zuordnung.get(job, []))
            self.jobs_zuordnung[job] = list(neue_tasks)
            for ma in self.aktive_mitarbeitende:
                self.job_task_index.setdefault(ma, {})
                self.job_durchsatz.setdefault(ma, {})
                self.job_task_index[ma][job] = 0
                self.job_durchsatz[ma][job] = None
            self.logger.info(
                f"{prefix}: {job} – {', '.join(alte_tasks) if alte_tasks else '-'} -> {', '.join(neue_tasks) if neue_tasks else '-'}"
            )
            self._protokolliere_jobprofilwechsel(
                tag, job, alte_tasks, list(neue_tasks), beschreibung
            )

    def _protokolliere_jobprofilwechsel( # erzeugt einen Ereigniseintrag „Jobprofil“ für die Ausgabe im Terminal mit detailliertem Text zur Änderung
        self,
        tag: int,
        job: str,
        alte_tasks: List[str],
        neue_tasks: List[str],
        beschreibung: Optional[str],
    ) -> None:
        alt_text = ", ".join(alte_tasks) if alte_tasks else "-"
        neu_text = ", ".join(neue_tasks) if neue_tasks else "-"
        text = f"Job {job}: {alt_text} -> {neu_text}"
        if beschreibung:
            text = f"{beschreibung} – {text}"
        for label in set(self.slot_label.values()):
            self.kompetenz_protokoll.setdefault(label, []).append(
                {
                    "Tag": tag,
                    "Tätigkeit": "Jobprofil",
                    "AFZ": None,
                    "Kompetenzstufe": None,
                    "Reduktion_%": None,
                    "Fehlerquote": None,
                    "Ereignis": text,
                }
            )

    
    def _normalisiere_runden(self, runden_roh: Any) -> Optional[Set[int]]: # organisiert Runden, die von Störungen betroffen sind
        if runden_roh is None:
            return None
        if isinstance(runden_roh, str):
            if runden_roh.strip().lower() in {"alle", "all", "*"}: # akzeptierte Schreibweisen für "alle Runden", bei Produktionsstörungen
                return None
            werte: Iterable[Any] = [runden_roh]
        elif isinstance(runden_roh, (int, float)):
            werte = [runden_roh]
        else:
            try:
                werte = list(runden_roh)
            except TypeError:
                werte = []

        result: Set[int] = set()
        for wert in werte:
            index: Optional[int] = None
            if isinstance(wert, str):
                if wert.strip().lower() in {"alle", "all", "*"}:
                    return None
                try:
                    index = int(float(wert)) - 1
                except (TypeError, ValueError):
                    wert_normalisiert = wert.strip().lower()
                    for idx, runde in enumerate(self.arbeitsrunden):
                        name = str(runde.get("name", "")).strip().lower()
                        if name and name == wert_normalisiert:
                            index = idx
                            break
            elif isinstance(wert, (int, float)):
                index = int(wert) - 1
            if index is None:
                continue
            if 0 <= index < len(self.arbeitsrunden):
                result.add(index)
        return result or None

    def _normalisiere_jobs(self, jobs_roh: Any) -> Optional[Set[str]]: # organisiert Jobs, die von Störungen betroffen sind
        if jobs_roh is None:
            return None
        if isinstance(jobs_roh, str):
            if jobs_roh.strip().lower() in {"alle", "all", "*"}:
                return None
            werte: Iterable[Any] = [jobs_roh]
        else:
            try:
                werte = list(jobs_roh)
            except TypeError:
                werte = [jobs_roh]

        result: Set[str] = set()
        bekannte_jobs = {str(job) for job in self.alle_jobs}
        job_lookup = {job.lower(): job for job in bekannte_jobs}
        for wert in werte:
            if wert is None:
                continue
            job = str(wert).strip()
            if not job:
                continue
            if job.lower() in {"alle", "all", "*"}:
                return None
            if job in bekannte_jobs:
                result.add(job)
                continue
            job_normalisiert = job.lower()
            if job_normalisiert in job_lookup:
                result.add(job_lookup[job_normalisiert])
        return result or None

    def _initialisiere_zustaende(self) -> None:     # Setzt sämtliche Simulationstabellen und -zählwerte auf ihre Startwerte
        spalten = [                                 # registriert alle Mitarbeitenden/Slots und startet für jede Person den ersten Arbeitstag            
            "DurchlaufNr",
            "AFZ",
            "AFA_pre",
            "AFZ_post",
            "AFA_post",
            "VG_Dauer",
            "Ausgefuehrt",
            "Sim_zeit",
            "Output_input",
            "Output_geplant",
            "Output_gut",
            "Ausschuss",
            "Output_geplant_kumuliert",
            "Ausschuss_kumuliert",
            "Fehlerquote",
        ]
        self.zustand_arbeitszeit = {
            ma: float(self.gesamt_arbeitszeit_pro_tag) for ma in self.aktive_mitarbeitende
        }
        self.z1 = {ma: 4 for ma in self.aktive_mitarbeitende}
        self.z2 = {ma: 1 for ma in self.aktive_mitarbeitende}
        self.simulationszeit = {ma: 0.0 for ma in self.aktive_mitarbeitende}
        self.runde_index = {ma: 0 for ma in self.aktive_mitarbeitende}
        self.runde_restzeit = {
            ma: float(self.arbeitsrunden[0]["arbeitszeit"]) if self.arbeitsrunden else 0.0
            for ma in self.aktive_mitarbeitende
        }
        self.job_task_index = {
            ma: {job: 0 for job in self.alle_jobs}
            for ma in self.aktive_mitarbeitende
        }
        self.letzte_iteration = {ma: None for ma in self.aktive_mitarbeitende}
        self.arbeitstag_zaehler = {ma: 0 for ma in self.aktive_mitarbeitende}
        self.job_durchsatz = {
            ma: {job: None for job in self.alle_jobs}
            for ma in self.aktive_mitarbeitende
        }
        self.AFA_pre = {
            ma: {t: self.vor_AFA_map[ma][t] for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.AFA_post = {
            ma: {t: self.vor_AFA_map[ma][t] for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.AFZ = {
            ma: {t: self.t_initiale_AFZ_map[ma][t] for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.AFZ_post = {
            ma: {t: self.t_initiale_AFZ_map[ma][t] for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.AFZ_last = {
            ma: {t: self.t_initiale_AFZ_map[ma][t] for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.vergessensdauer = {
            ma: {t: 0.0 for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.ausgefuehrt = {
            ma: {t: 0 for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.m_break_stunden = {ma: 0.0 for ma in self.aktive_mitarbeitende}
        self.m_abbau_tag = {
            ma: {t: self.m_min for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.m_aufbau_tag = {
            ma: {t: self.m_min for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self.m_tag_index: Dict[str, Optional[int]] = {
            ma: None for ma in self.aktive_mitarbeitende
        }
        self.m_tag_label: Dict[str, Optional[str]] = {
            ma: ma for ma in self.aktive_mitarbeitende
        }
        self.m_gesamt_wiederholungen = {
            ma: {t: 0 for t in self.taetigkeiten}
            for ma in self.aktive_mitarbeitende
        }
        self._output_spalten = spalten
        self.output_data_all.clear()
        self.job_history.clear()
        self.kompetenz_protokoll.clear()
        self.durchlauf_index_person.clear()
        self.slot_label = {ma: ma for ma in self.aktive_mitarbeitende}
        self.uebungsfaktor_protokoll = {}
        for label in self.slot_label.values():
            self._registriere_person(label)
        self._arbeitstag_beendet = {ma: False for ma in self.aktive_mitarbeitende}
        self._runde_event_marker = {ma: set() for ma in self.aktive_mitarbeitende}

        for ma in self.aktive_mitarbeitende:
            self._beginne_neuen_tag(ma)

    def before_round(self, tag: int, runde_index: int, runde: Dict[str, Any]) -> None: # ruft Ereignisse vor der Runde auf
        for handler in self._before_round_handlers:
            handler(self, tag, runde)

    def after_day(self, tag: int) -> None: # ruft Ereignisse nach dem Tag auf
        for handler in self._after_day_handlers:
            handler(self, tag)

    def _registriere_person(self, label: str) -> None: # legt zu Beginn für eine Person alle Output-DataFrames, Job-Historie und Kompetenzprotokoll an
        if label not in self.uebungsfaktor_protokoll:
            self.uebungsfaktor_protokoll[label] = []
        if label in self.output_data_all:
            return
        self.output_data_all[label] = {
            t: pd.DataFrame(columns=self._output_spalten) for t in self.taetigkeiten
        }
        self.job_history[label] = []
        self.kompetenz_protokoll[label] = []
        self.durchlauf_index_person[label] = 0

    def _aktuelles_label(self, ma: str) -> str: # Liefert das aktuell auf den Slot gemappte Label (Unterscheidung zwischen Original oder Ersatzkraft)
        return self.slot_label.get(ma, ma)

    def _erstelle_profil( # erzeugt ein Mitarbeiterprofil
        self,
        name: str,
        t_initiale: Dict[str, float],
        vor_afa: Dict[str, float],
        gw: Dict[str, float],
        lf: Dict[str, float],
        vf: Dict[str, float],
        lernkurve: Lernkurve,
        vergessenskurve: Vergessenskurve,
    ) -> MitarbeiterProfil:
        return MitarbeiterProfil(
            name=name,
            t_initiale_AFZ=dict(t_initiale),
            vor_AFA=dict(vor_afa),
            gw=dict(gw),
            lf=dict(lf),
            vf=dict(vf),
            lernkurve=lernkurve,
            vergessenskurve=vergessenskurve,
        )

    def _set_aktives_profil(self, ma: str, profil: MitarbeiterProfil) -> None:  # Aktualisiert alle Parameter und Kurvenreferenzen für eine Person und 
        self.t_initiale_AFZ_map[ma] = dict(profil.t_initiale_AFZ)               # berechnet die Kompetenzparameter (Initialwert, Grenzwert, Differenz) je Tätigkeit   
        self.vor_AFA_map[ma] = dict(profil.vor_AFA)
        self.gw_map[ma] = dict(profil.gw)
        self.lf_map[ma] = dict(profil.lf)
        self.vf_map[ma] = dict(profil.vf)
        self.lernkurven[ma] = profil.lernkurve
        self.vergessenskurven[ma] = profil.vergessenskurve
        self.kompetenz_parameter[ma] = {}
        for t in self.taetigkeiten:
            initial = float(self.t_initiale_AFZ_map[ma].get(t, 0.0))
            gw = float(self.gw_map[ma].get(t, 1.0))
            grenzwert = initial * gw
            differenz = max(initial - grenzwert, 0.0)
            self.kompetenz_parameter[ma][t] = {
                "initial": initial,
                "grenzwert": grenzwert,
                "differenz": differenz,
            }

    def _klone_profil(self, vorlage: MitarbeiterProfil, name: str) -> MitarbeiterProfil:    # kommt zum Einsatz, sobald eine Ersatzkraft aktiviert werden muss;
        t_initiale = dict(vorlage.t_initiale_AFZ)                                           # lässt sich beliebig oft eine standardisierte Ersatzperson erzeugen, 
        vor_afa = dict(vorlage.vor_AFA)                                                     # während die Stammkraft über einen Snapshot eingefroren bleib;   
        gw = dict(vorlage.gw)                                                               # bei Rückkehr der Stammkraft werden die gespeicherten Originalwerte wiederhergestellt
        lf = dict(vorlage.lf)
        vf = dict(vorlage.vf)
        lernkurve = Lernkurve(
            t_initiale_AFZ=t_initiale,
            M=gw,
            k=lf,
            prozesstaetigkeiten=self.taetigkeiten,
            m_min=self.m_min,
            m_max=self.m_max,
        )
        vergessenskurve = Vergessenskurve(
            t_initiale_AFZ=t_initiale,
            c=vf,
            prozesstaetigkeiten=self.taetigkeiten,
        )
        return MitarbeiterProfil(
            name=name,
            t_initiale_AFZ=t_initiale,
            vor_AFA=vor_afa,
            gw=gw,
            lf=lf,
            vf=vf,
            lernkurve=lernkurve,
            vergessenskurve=vergessenskurve,
        )

    def _berechne_standardprofilvorlage(self) -> MitarbeiterProfil: # erzeugt ein Standard-Ersatzprofil inkl. Kurvenobjekten
        if not self.aktive_mitarbeitende:
            raise ValueError("Es kann kein Ersatzprofil ohne Mitarbeitende erstellt werden.")
        referenz = self.aktive_mitarbeitende[0]
        default_initial = 120.0
        default_vor = 1.0
        default_gw = 0.3
        default_lf = -0.25
        default_vf = default_lf / 10

        ersatz_config = getattr(self.eingabe, "ersatz_standardparameter", {})
        cfg_t_initiale = ersatz_config.get("t_initiale_AFZ", {})
        cfg_vor = ersatz_config.get("vor_AFA", {})
        cfg_gw = ersatz_config.get("GW", ersatz_config.get("gw", {}))
        cfg_lf = ersatz_config.get("LF", ersatz_config.get("lf", {}))
        cfg_vf = ersatz_config.get("VF", ersatz_config.get("vf", {}))

        t_initiale: Dict[str, float] = {}
        vor_afa: Dict[str, float] = {}
        gw: Dict[str, float] = {}
        lf: Dict[str, float] = {}
        vf: Dict[str, float] = {}

        for t in self.taetigkeiten:
            initialwerte = [
                self.t_initiale_AFZ_map[ma].get(t)
                for ma in self.aktive_mitarbeitende
                if t in self.t_initiale_AFZ_map[ma]
            ]
            fallback_initial = (
                float(mean(initialwerte))
                if initialwerte
                else float(self.t_initiale_AFZ_map.get(referenz, {}).get(t, default_initial))
            )
            t_initiale[t] = float(cfg_t_initiale.get(t, fallback_initial))

            vorwerte = [
                self.vor_AFA_map[ma].get(t)
                for ma in self.aktive_mitarbeitende
                if t in self.vor_AFA_map[ma]
            ]
            fallback_vor = (
                float(mean(vorwerte))
                if vorwerte
                else float(self.vor_AFA_map.get(referenz, {}).get(t, default_vor))
            )
            vor_afa[t] = float(cfg_vor.get(t, fallback_vor))

            gwerte = [
                self.gw_map[ma].get(t)
                for ma in self.aktive_mitarbeitende
                if t in self.gw_map[ma]
            ]
            fallback_gw = (
                float(mean(gwerte))
                if gwerte
                else float(self.gw_map.get(referenz, {}).get(t, default_gw))
            )
            gw[t] = float(cfg_gw.get(t, fallback_gw))

            lwerte = [
                self.lf_map[ma].get(t)
                for ma in self.aktive_mitarbeitende
                if t in self.lf_map[ma]
            ]
            fallback_lf = (
                float(mean(lwerte))
                if lwerte
                else float(self.lf_map.get(referenz, {}).get(t, default_lf))
            )
            lf[t] = float(cfg_lf.get(t, fallback_lf))

            vwerte = [
                self.vf_map[ma].get(t)
                for ma in self.aktive_mitarbeitende
                if t in self.vf_map[ma]
            ]
            fallback_vf = (
                float(mean(vwerte))
                if vwerte
                else float(self.vf_map.get(referenz, {}).get(t, lf[t] / 10 if lf[t] else default_vf))
            )
            vf[t] = float(cfg_vf.get(t, fallback_vf))

        lernkurve = Lernkurve(
            t_initiale_AFZ=t_initiale,
            M=gw,
            k=lf,
            prozesstaetigkeiten=self.taetigkeiten,
            m_min=self.m_min,
            m_max=self.m_max,
        )
        vergessenskurve = Vergessenskurve(
            t_initiale_AFZ=t_initiale,
            c=vf,
            prozesstaetigkeiten=self.taetigkeiten,
        )
        return MitarbeiterProfil(
            name="ERSATZ_VORLAGE",
            t_initiale_AFZ=t_initiale,
            vor_AFA=vor_afa,
            gw=gw,
            lf=lf,
            vf=vf,
            lernkurve=lernkurve,
            vergessenskurve=vergessenskurve,
        )

    def _add_simulationszeit(self, ma: str, delta: float) -> None:  # Addiert positive Zeitsprünge auf die individuelle Simulationszeit
        if delta <= 0:                                              # und registriert sie ggf. als Abwesenheit (für personalbezogene Risiken)
            return
        self.simulationszeit[ma] += delta
        self._registriere_abwesenheitszeit(ma, delta)

    def _registriere_abwesenheitszeit(self, ma: str, delta: float) -> None: # Vermerkt zusätzliche Abwesenheitssekunden für aktive Ersatzereignisse (Ausfall/Fluktuation)
        if delta <= 0:
            return
        status = self._personalstatus.get(ma)
        if not status or not status.ersatz_aktiv:
            return
        if status.aktives_ereignis and status.aktives_ereignis.typ in {"ausfall", "fluktuation"}:
            status.abwesenheit_sekunden += delta

    def _registriere_m_unterbrechung(self, ma: str, dauer: float) -> None: # Konvertiert Abwesenheitsdauer in Stunden und akkumuliert sie zur späteren Übungsfaktor-Anpassung
        if dauer <= 0:
            return
        stunden = dauer / 3600.0
        if stunden <= 0:
            return
        self.m_break_stunden[ma] = self.m_break_stunden.get(ma, 0.0) + stunden

    def _erzeuge_snapshot(self, ma: str) -> MitarbeiterStateSnapshot:   # Sichert sämtliche Kompetenz-/Übungszustände einer Person in einem MitarbeiterStateSnapshot, 
        return MitarbeiterStateSnapshot(                                # um sie später wiederherstellen zu können
            AFA_pre=deepcopy(self.AFA_pre[ma]),
            AFA_post=deepcopy(self.AFA_post[ma]),
            AFZ=deepcopy(self.AFZ[ma]),
            AFZ_post=deepcopy(self.AFZ_post[ma]),
            AFZ_last=deepcopy(self.AFZ_last[ma]),
            vergessensdauer=deepcopy(self.vergessensdauer[ma]),
            ausgefuehrt=deepcopy(self.ausgefuehrt[ma]),
            letzte_iteration=self.letzte_iteration[ma],
            m_abbau=deepcopy(self.m_abbau_tag[ma]),
            m_aufbau=deepcopy(self.m_aufbau_tag[ma]),
            m_break_stunden=float(self.m_break_stunden.get(ma, 0.0)),
            m_tag_index=self.m_tag_index.get(ma),
            m_tag_label=self.m_tag_label.get(ma),
            m_gesamt_wiederholungen=deepcopy(self.m_gesamt_wiederholungen[ma]),
        )

    def _setze_ersatz_startzustand(self, ma: str) -> None:  # initialisiert die Ersatzkräfte
        profil = self.profil_aktuell[ma]
        self.AFA_pre[ma] = {t: profil.vor_AFA.get(t, 0.0) for t in self.taetigkeiten}
        self.AFA_post[ma] = {t: profil.vor_AFA.get(t, 0.0) for t in self.taetigkeiten}
        self.AFZ[ma] = {t: profil.t_initiale_AFZ.get(t, 0.0) for t in self.taetigkeiten}
        self.AFZ_post[ma] = {t: profil.t_initiale_AFZ.get(t, 0.0) for t in self.taetigkeiten}
        self.AFZ_last[ma] = {t: profil.t_initiale_AFZ.get(t, 0.0) for t in self.taetigkeiten}
        self.vergessensdauer[ma] = {t: 0.0 for t in self.taetigkeiten}
        self.ausgefuehrt[ma] = {t: 0 for t in self.taetigkeiten}
        self.letzte_iteration[ma] = None
        self.m_break_stunden[ma] = 0.0
        self.m_tag_index[ma] = None
        self.m_tag_label[ma] = self._aktuelles_label(ma)
        self.m_gesamt_wiederholungen[ma] = {t: 0 for t in self.taetigkeiten}
        self.m_abbau_tag[ma] = {t: self.m_min for t in self.taetigkeiten}
        self.m_aufbau_tag[ma] = {t: self.m_min for t in self.taetigkeiten}
        for taetigkeit in self.taetigkeiten:
            self.lernkurven[ma].set_m_faktor(taetigkeit, self.m_min)

    def _protokolliere_personalereignis(self, label: str, tag: int, text: str) -> None:     # protokolliert Personalbewegungen, z.B. Aktivieren einer Ersatzkraft oder Rückkehr
        self.kompetenz_protokoll[label].append(
            {
                "Tag": tag,
                "Tätigkeit": "Personal",
                "AFZ": None,
                "Kompetenzstufe": None,
                "Reduktion_%": None,
                "Fehlerquote": None,
                "Ereignis": text,
            }
        )

    def _aktiviere_ersatzkraft(self, ma: str, ereignis: Personalereignis, tag: int) -> None:    # sichert den aktuellen Zustand der ausfallenden Arbeitskraft, 
        status = self._personalstatus[ma]                                                       # erzeugt ein Ersatzprofil und protokolliert den Einsatz  
        if status.ersatz_aktiv:
            return
        status.original_snapshot = self._erzeuge_snapshot(ma)
        status.aktives_ereignis = ereignis
        status.verbleibende_ausfalltage = max(int(ereignis.dauer or 0), 0)
        if ereignis.typ == "ausfall" and status.verbleibende_ausfalltage <= 0:
            status.verbleibende_ausfalltage = 1
        status.rueckkehr_tag = None
        status.abwesenheit_sekunden = 0.0
        status.ersatz_index += 1
        self._ersatz_global_index += 1
        ersatz_label = f"EMA{self._ersatz_global_index}"
        ersatz_profil = self._klone_profil(
            self._ersatz_profil_vorlage,
            ersatz_label,
        )
        self.profil_aktuell[ma] = ersatz_profil
        self._set_aktives_profil(ma, ersatz_profil)
        self._setze_ersatz_startzustand(ma)
        status.ersatz_label = ersatz_label
        self.slot_label[ma] = ersatz_label
        self._registriere_person(ersatz_label)
        status.ersatz_aktiv = True
        self.logger.info(
            f"Ersatzkraft für {ma} aktiv ab Tag {tag} ({ereignis.typ})."
        )
        self._protokolliere_personalereignis(
            status.original,
            tag,
            f"Ersatzkraft startet ({ereignis.typ})",
        )
        self._protokolliere_personalereignis(
            ersatz_label,
            tag,
            f"Einsatz beginnt für {status.original} ({ereignis.typ})",
        )

    def _deaktiviere_ersatzkraft(self, ma: str, tag: int) -> None:      # beim Ende eines Ersatz-Einsatzes wird gespeicherte Originalzustand wieder hergestellt   
        status = self._personalstatus[ma]                               # Rückkehr wird protokolliert und Folgen der Abwesenheit verarbeitet
        if not status.ersatz_aktiv:
            return
        snapshot = status.original_snapshot
        if snapshot is None:
            return
        ersatz_label = status.ersatz_label
        self.profil_aktuell[ma] = self.original_profile[ma]
        self._set_aktives_profil(ma, self.profil_aktuell[ma])
        self.AFA_pre[ma] = deepcopy(snapshot.AFA_pre)
        self.AFA_post[ma] = deepcopy(snapshot.AFA_post)
        self.AFZ[ma] = deepcopy(snapshot.AFZ)
        self.AFA_post[ma] = deepcopy(snapshot.AFA_post)
        self.AFZ[ma] = deepcopy(snapshot.AFZ)
        self.AFZ_post[ma] = deepcopy(snapshot.AFZ_post)
        self.AFZ_last[ma] = deepcopy(snapshot.AFZ_last)
        self.vergessensdauer[ma] = deepcopy(snapshot.vergessensdauer)
        self.ausgefuehrt[ma] = deepcopy(snapshot.ausgefuehrt)
        self.slot_label[ma] = status.original
        self.letzte_iteration[ma] = snapshot.letzte_iteration
        self.m_abbau_tag[ma] = deepcopy(snapshot.m_abbau)
        self.m_aufbau_tag[ma] = deepcopy(snapshot.m_aufbau)
        self.m_break_stunden[ma] = float(snapshot.m_break_stunden)
        self.m_tag_index[ma] = snapshot.m_tag_index
        self.m_tag_label[ma] = snapshot.m_tag_label or status.original
        self.m_gesamt_wiederholungen[ma] = deepcopy(snapshot.m_gesamt_wiederholungen)
        for taetigkeit in self.taetigkeiten:
            self.lernkurven[ma].set_m_faktor(
                taetigkeit, self.m_abbau_tag[ma].get(taetigkeit, self.m_min)
            )
        delta = status.abwesenheit_sekunden
        status.ersatz_aktiv = False
        status.aktives_ereignis = None
        status.original_snapshot = None
        status.verbleibende_ausfalltage = 0
        status.rueckkehr_tag = None
        status.abwesenheit_sekunden = 0.0
        status.ersatz_label = None
        self.logger.info(f"Rückkehr von {ma} nach Ausfall an Tag {tag}.")
        self._protokolliere_personalereignis(status.original, tag, "Rückkehr nach Ausfall")
        if ersatz_label:
            self._protokolliere_personalereignis(
                ersatz_label,
                tag,
                f"Einsatz beendet für {status.original}",
            )
        index = self.letzte_iteration.get(ma)
        if delta > 0 and index is not None:
            self._verarbeite_abwesenheit(ma, delta, index, m_relevant=True)

    def _aktualisiere_personalstatus_nach_tag(self, ma: str) -> None: # zählt verbleibende Ausfalltage herunter und merkt den geplanten Rückkehrtag
        status = self._personalstatus.get(ma)
        if not status or not status.ersatz_aktiv or not status.aktives_ereignis:
            return
        if status.aktives_ereignis.typ == "ausfall" and status.verbleibende_ausfalltage > 0:
            status.verbleibende_ausfalltage -= 1
            if status.verbleibende_ausfalltage <= 0:
                status.rueckkehr_tag = self.arbeitstag_zaehler[ma] + 1

    def _pruefe_personalereignisse(self, ma: str, tag: int) -> None:    # verhindert Mehrfachprüfung pro Tag, beendet abgelaufene Ausfälle und
        status = self._personalstatus.get(ma)                           # aktiviert neue Personalereignisse (Ausfall/Fluktuation) in chronologischer Reihenfolge    
        if not status:
            return
        if status.letzter_gepruefter_tag == tag:
            return
        status.letzter_gepruefter_tag = tag

        if (
            status.ersatz_aktiv
            and status.aktives_ereignis
            and status.aktives_ereignis.typ == "ausfall"
            and status.verbleibende_ausfalltage <= 0
            and status.rueckkehr_tag == tag
        ):
            self._deaktiviere_ersatzkraft(ma, tag)

        while status.ereignisse and status.ereignisse[0].start_tag <= tag:
            ereignis = status.ereignisse.pop(0)
            if status.ersatz_aktiv:
                if ereignis.typ == "fluktuation":
                    status.aktives_ereignis = ereignis
                    status.verbleibende_ausfalltage = 0
                    status.rueckkehr_tag = None
                    self.logger.info(
                        f"Fluktuation für {ma} ab Tag {tag}: Ersatzkraft bleibt dauerhaft."
                    )
                    self._protokolliere_personalereignis(
                        status.original, tag, "Fluktuation – Ersatz bleibt"
                    )
                    aktuelles_label = status.ersatz_label or self._aktuelles_label(ma)
                    if aktuelles_label:
                        self._protokolliere_personalereignis(
                            aktuelles_label,
                            tag,
                            "Fluktuation – dauerhafte Übernahme",
                        )
                elif ereignis.typ == "ausfall":
                    status.aktives_ereignis = ereignis
                    status.verbleibende_ausfalltage = max(
                        status.verbleibende_ausfalltage, int(ereignis.dauer or 0)
                    )
                continue
            self._aktiviere_ersatzkraft(ma, ereignis, tag)
            if ereignis.typ == "fluktuation":
                status.verbleibende_ausfalltage = 0
                status.rueckkehr_tag = None
            break

    def _ermittle_stoerung(                                         # sucht im Störungsplan nach einem Ereignis, das Tag, Runde und Job betrifft
        self, tag: int, runden_index: int, job: str                 # liefert das passende Objekt bzw. Ereignis
    ) -> Optional[StoerungsEreignis]:
        ereignisse = self._stoerungsplan.get(tag, [])
        for stoerung in ereignisse:
            if stoerung.runden is not None and runden_index not in stoerung.runden:
                continue
            if stoerung.jobs is not None and job not in stoerung.jobs:
                continue
            return stoerung
        return None

    def _bearbeite_stoerung(                                        # verarbeitet Störungen, veranlasst Änderungen wegen Vergessens und Übungsfaktor-Updates,
        self,                                                       # aktualisiert Restzeiten und protokolliert die Störung
        ma: str,
        tag: int,
        runden_index: int,
        runde: Dict[str, Any],
        job: str,
        stoerung: StoerungsEreignis,
    ) -> None:
        downtime = self.runde_restzeit.get(ma, 0.0)
        if downtime <= 0:
            downtime = float(runde.get("arbeitszeit", 0.0))
        if downtime <= 0:
            return
        label = self._aktuelles_label(ma)
        beschreibung = stoerung.beschreibung or "Produktionsstörung"
        self.logger.info(
            f"Störung – {beschreibung} | Tag {tag}, Runde {runden_index + 1}, Job {job}: "
            f"{downtime/60:.1f} Minuten Stillstand für {label}"
        )
        self._add_simulationszeit(ma, downtime)
        index = self.letzte_iteration.get(ma)
        if index is not None:
            self._verarbeite_abwesenheit(
                ma,
                downtime,
                index,
                m_relevant=downtime >= 24 * 3600,
            )
        self.zustand_arbeitszeit[ma] = max(0.0, self.zustand_arbeitszeit[ma] - downtime)
        self.runde_restzeit[ma] = max(0.0, self.runde_restzeit.get(ma, 0.0) - downtime)
        self._protokolliere_stoerung(label, tag, runden_index, job, stoerung, downtime)

    def _bearbeite_job_ohne_taetigkeiten(                       # für den Fall, dass ein Job keine Tätigkeiten mehr hat, aber zugeschrieben wurde, z.B. J3 = []           
        self,                                                   # gesamte restliche Rundenzeit wird als Leerlauf behandelt (also als Pause)   
        ma: str,
        tag: int,
        runden_index: int,
        runde: Dict[str, Any],
        job: str,
    ) -> None:
        restzeit = self.runde_restzeit.get(ma, 0.0)
        if restzeit <= 0:
            restzeit = float(runde.get("arbeitszeit", 0.0))
        if restzeit <= 0:
            return
        label = self._aktuelles_label(ma)
        self.logger.info(
            f"Leerlauf – Job {job} ohne aktive Tätigkeiten | Tag {tag}, Runde {runden_index + 1}: "
            f"{restzeit/60:.1f} Minuten für {label}"
        )
        self._add_simulationszeit(ma, restzeit)
        index = self.letzte_iteration.get(ma)
        if index is not None:
            self._verarbeite_abwesenheit(ma, restzeit, index, m_relevant=False)
        self.zustand_arbeitszeit[ma] = max(0.0, self.zustand_arbeitszeit[ma] - restzeit)
        self.runde_restzeit[ma] = max(0.0, self.runde_restzeit.get(ma, 0.0) - restzeit)
        self.job_durchsatz.setdefault(ma, {})
        self.job_durchsatz[ma][job] = None
        self.job_task_index.setdefault(ma, {})
        self.job_task_index[ma][job] = 0


    def _protokolliere_stoerung( # schreibt den Störungseintrag ins Kompetenzprotokoll
        self,
        label: str,
        tag: int,
        runden_index: int,
        job: str,
        stoerung: StoerungsEreignis,
        dauer: float,
    ) -> None:
        if label not in self.kompetenz_protokoll:
            return
        dauer_minuten = dauer / 60 if dauer else 0
        beschreibung = stoerung.beschreibung or "Produktionsstörung"
        self.kompetenz_protokoll[label].append(
            {
                "Tag": tag,
                "Tätigkeit": "Störung",
                "AFZ": None,
                "Kompetenzstufe": None,
                "Reduktion_%": None,
                "Fehlerquote": None,
                "Ereignis": (
                    f"{beschreibung} – Runde {runden_index + 1}, Job {job}, "
                    f"Stillstand {dauer_minuten:.1f} min"
                ),
            }
        )

    def _lernen(self, ma: str, taetigkeit: str, wdh: float) -> float: # Berechnung der aktuellen Ausführungszeit (AFZ) anhand der Wiederholungszahl
        return self.lernkurven[ma].berechne_ausfuehrungszeit(taetigkeit, wdh)

    def _berechne_AFA(self, ma: str, taetigkeit: str, afz_post: float) -> float: # aus einer Ausführungszeit die bisherige Wiederholungsanzahl bestimmen
        return self.lernkurven[ma].berechne_ausfuehrungsanzahl(taetigkeit, afz_post)

    # def _vergessen(self, ma: str, taetigkeit: str, dauer: float, tpost: float, zi: float) -> float: 
    #     return self.vergessenskurven[ma].berechne_AFZ_nach_Vergessen(taetigkeit, dauer, tpost, zi)

    def _vergessen(self, ma: str, taetigkeit: str, dauer: float, tpost: float, zi: float) -> float: # Berechnung der AFZ nach einer Pause mit Übungsfaktor
        m_aufbau = self.m_aufbau_tag.get(ma, {}).get(taetigkeit)
        if m_aufbau is None:
            m_aufbau = self.m_min
        return self.vergessenskurven[ma].berechne_AFZ_nach_Vergessen(
            taetigkeit,
            dauer,
            tpost,
            zi,
            m_aufbau,
        )


    def _aktualisiere_uebungsfaktor_aufbau(                     # Berechnet für jede Tätigkeit einen Übungsfaktor basierend auf bisherigem Übungsstand, Komplexität 
        self, ma: str, tag: int, label: str                     # und Normalisierung, speichert ihn und merkt ihn sich für spätere Protokolle
    ) -> None:
        for taetigkeit in self.taetigkeiten:
            m_vorher = max(self.m_abbau_tag[ma].get(taetigkeit, self.m_min), self.m_min)
            komplexitaet = max(self.taetigkeit_komplexitaet.get(taetigkeit, 1.0), 1.0)
            basis_wiederholungen = self.m_gesamt_wiederholungen[ma].get(taetigkeit, 0)
            if self.m_n_norm > 0:
                basis_faktor = max(basis_wiederholungen / self.m_n_norm, 1.0)
            else:
                basis_faktor = 1.0
            exponent = 1.0 / (komplexitaet ** self.m_alpha) if komplexitaet > 0 else 1.0
            differenz = self.m_max - m_vorher
            if differenz <= 0:
                m_aufbau = min(self.m_max, max(self.m_min, m_vorher))
            else:
                m_aufbau = self.m_max - differenz / (basis_faktor ** exponent)
            m_aufbau = min(self.m_max, max(self.m_min, m_aufbau))
            self.m_aufbau_tag[ma][taetigkeit] = m_aufbau
        self.m_tag_index[ma] = tag
        self.m_tag_label[ma] = label

    def _verarbeite_uebungsfaktor_fuer_neuen_tag(self, ma: str) -> None:        # wandelt angesammelte Pausenstunden in neue m_abbau-Werte um
        tag_vorher = self.m_tag_index.get(ma)
        label_vorher = self.m_tag_label.get(ma, self._aktuelles_label(ma))
        q_stunden = max(self.m_break_stunden.get(ma, 0.0), 0.0)
        if tag_vorher is not None and label_vorher is not None:
            basis = max(q_stunden, self.m_q_min)
            for taetigkeit in self.taetigkeiten:
                m_abbau_alt = max(
                    self.m_abbau_tag[ma].get(taetigkeit, self.m_min), self.m_min
                )
                m_aufbau = max(
                    self.m_aufbau_tag[ma].get(taetigkeit, m_abbau_alt), self.m_min
                )
                komplexitaet = max(
                    self.taetigkeit_komplexitaet.get(taetigkeit, 1.0), 1.0
                )
                exponent = (1.0 / (komplexitaet ** self.m_alpha)) - 1.0
                if basis > 0:
                    multiplikator = basis ** exponent
                else:
                    multiplikator = 1.0
                m_abbau_neu = self.m_min + (m_aufbau - self.m_min) * multiplikator
                m_abbau_neu = min(self.m_max, max(self.m_min, m_abbau_neu))
                self.m_abbau_tag[ma][taetigkeit] = m_abbau_neu
                self.lernkurven[ma].set_m_faktor(taetigkeit, m_abbau_neu)
                protokoll_eintrag = {
                    "Tag": tag_vorher,
                    "Mitarbeiter": label_vorher,
                    "Tätigkeit": taetigkeit,
                    "m_abbau": float(m_abbau_alt),
                    "m_aufbau": float(m_aufbau),
                    "q_stunden": float(q_stunden),
                    "n_ges": int(self.m_gesamt_wiederholungen[ma].get(taetigkeit, 0)),
                    "Komplexität": float(
                        self.taetigkeit_komplexitaet.get(taetigkeit, 1.0)
                    ),
                }
                self.uebungsfaktor_protokoll.setdefault(label_vorher, []).append(
                    protokoll_eintrag
                )
            self.m_break_stunden[ma] = 0.0
            self.m_tag_index[ma] = None
        else:
            for taetigkeit in self.taetigkeiten:
                aktuelles_m = self.m_abbau_tag[ma].get(taetigkeit, self.m_min)
                self.lernkurven[ma].set_m_faktor(taetigkeit, aktuelles_m)
            self.m_break_stunden[ma] = 0.0
        self.m_tag_label[ma] = self._aktuelles_label(ma)

    def _ermittle_kompetenzstufe(self, ma: str, taetigkeit: str, aktuelle_afz: float) -> int: # berechnet aus relativer AFZ-Reduktion diskrete Kompetenzstufen
        if getattr(self, "_fehlerfrei_flag", False):
            return 5
        parameter = self.kompetenz_parameter[ma][taetigkeit]
        initial = parameter["initial"]
        differenz = parameter["differenz"]
        if differenz <= 0:
            return 5 if aktuelle_afz <= initial else 1
        reduktion = (initial - aktuelle_afz) / differenz
        reduktion = max(0, min(reduktion, 1))
        if reduktion < 0.5:         # Wichtig: Simulation_Berechnungen.berechne_kompetenzstufe ist nur ein Hilfswerkzeug für die Auswertung nach der Simulation
            return 1                # SimulationRunner._ermittle_kompetenzstufe ist die Laufzeitvariante im Simulator selbst, also Rechner während der Simulation
        if reduktion < 0.6:
            return 2
        if reduktion < 0.7:
            return 3
        if reduktion < 0.8:
            return 4
        return 5
    
    def _wende_fehlerarme_ausfuehrungszeit_an(      # Kürzt in fehlerarmer Simulationen  (Fehlerquote = 0.02) die AFZ auf einen Mindestwert 
        self,                                       # zwischen Grenzwert und definierter Reduktionsschwelle
        ma: str,
        taetigkeit: str,
        aktuelle_afz: float,
    ) -> float:
        if not getattr(self, "_fehlerfrei_flag", False):
            return aktuelle_afz
        parameter = self.kompetenz_parameter[ma][taetigkeit]
        differenz = parameter["differenz"]
        if differenz <= 0:
            return aktuelle_afz
        ziel_afz = parameter["initial"] - differenz * self.FEHLERARM_MIN_REDUKTION
        ziel_afz = max(parameter["grenzwert"], ziel_afz)
        return min(aktuelle_afz, ziel_afz)
    
    def _ermittle_fehlerquote(                      # leitet eine Fehlerquote aus stufenspezifischen Parametern und Standardwerten ab
        self,
        ma: str,
        taetigkeit: str,
        kompetenzstufe: int,
        reduktion_rel: Optional[float] = None,
    ) -> float:
        parameter = getattr(self.eingabe, "fehlerquote_parameter", {})
        standard = getattr(self.eingabe, "standard_fehlerquote", {})
        stufenwerte = parameter.get(ma, {}).get(taetigkeit, {})
        if kompetenzstufe in stufenwerte:
            return max(0.0, min(stufenwerte[kompetenzstufe], 1.0))
        if stufenwerte:
            naechste = max(stufenwerte.keys())
            return max(0.0, min(stufenwerte[naechste], 1.0))
        if kompetenzstufe in standard:
            return max(0.0, min(standard[kompetenzstufe], 1.0))
        if standard:
            return max(0.0, min(standard[max(standard.keys())], 1.0))
        if reduktion_rel is not None:
            return max(0.0, min(1.0, 1 - reduktion_rel))
        return 0.0

    def _ermittle_outputmenge(self, ma: str, taetigkeit: str) -> float:     # falls eine andere Outputmenge pro Ausführung (z.B. 2 Stücke pro Ausführung) 
        konfig = getattr(self.eingabe, "output_pro_ausfuehrung", {})        # im Eingabe-Modul hinterlegt wird, wird sie genutzt; ansosnten immer 1
        if isinstance(konfig, dict):
            if taetigkeit in konfig and not isinstance(konfig[taetigkeit], dict):
                return float(konfig[taetigkeit])
            if ma in konfig and isinstance(konfig[ma], dict):
                return float(konfig[ma].get(taetigkeit, 1))
            if taetigkeit in konfig and isinstance(konfig[taetigkeit], dict):
                werte = konfig[taetigkeit]
                if isinstance(werte, dict):
                    return float(werte.get(ma, werte.get("default", 1)))
        return 1.0

    def _simulation_abgeschlossen(self, ma: str) -> bool:     # prüft, ob ein externer Zieltag oder das maximale Simulationszeitlimit erreicht wurde
        if (
            self._simulation_stop_tag is not None
            and self.arbeitstag_zaehler.get(ma, 0) >= self._simulation_stop_tag
        ):
            return True
        return self.simulationszeit[ma] >= self.simulationszeit_limit

    def _beginne_neuen_tag(self, ma: str) -> None:     # Reset der Tagesstände (Arbeitszeit, Runden, etc.) und Aufruf des Übungsfaktor-Updates für neuen Arbeitstag
        self._verarbeite_uebungsfaktor_fuer_neuen_tag(ma)
        self.zustand_arbeitszeit[ma] = float(self.gesamt_arbeitszeit_pro_tag)
        self.runde_index[ma] = 0
        if self.arbeitsrunden:
            self.runde_restzeit[ma] = float(self.arbeitsrunden[0]["arbeitszeit"])
        else:
            self.runde_restzeit[ma] = 0.0
        self._arbeitstag_beendet[ma] = False
        self._runde_event_marker[ma] = set()

    def _runde_abschliessen(self, ma: str) -> None:     # erhöht den Rundenzähler und lädt die Restarbeitszeit der nächsten Runde oder beendet den Arbeitstag, 
        self.runde_index[ma] += 1                       # falls keine weiteren Runden existieren
        if self.runde_index[ma] < len(self.arbeitsrunden):
            self.runde_restzeit[ma] = float(
                self.arbeitsrunden[self.runde_index[ma]]["arbeitszeit"]
            )
        else:
            self.zustand_arbeitszeit[ma] = 0.0

    def _simuliere_bis_tagesende(self, ma: str, fehlerfrei: bool) -> None:      # Hauptschleife pro Mitarbeitendem: verarbeitet Tagesereignisse,führt Jobs aus,  
        while True:                                                             # beendet Tage und prüft Ziele, bis der Tag abgeschlossen ist oder die Simulation stoppt
            if self._simulation_abgeschlossen(ma):
                return

            aktueller_tag = self.arbeitstag_zaehler[ma] + 1
            self._pruefe_jobprofil_aenderungen(aktueller_tag)
            self._pruefe_personalereignisse(ma, aktueller_tag)

            if self.zustand_arbeitszeit[ma] <= 0 or self.runde_index[ma] >= len(
                self.arbeitsrunden
            ):
                if not self._beende_arbeitstag(ma, aktueller_tag):
                    return
                if self._simulation_abgeschlossen(ma):
                    return
                return

            runden_index = self.runde_index[ma]
            aktuelle_runde = self.arbeitsrunden[runden_index]
            marker_key = (aktueller_tag, runden_index)
            if marker_key not in self._runde_event_marker[ma]:
                self.before_round(aktueller_tag, runden_index, aktuelle_runde)
                self._runde_event_marker[ma].add(marker_key)

            zuordnung = aktuelle_runde.get("jobs", {})
            if ma not in zuordnung:
                self._fuehre_pause_durch(
                    ma, aktuelle_runde.get("pause", 0), self.letzte_iteration[ma]
                )
                self._runde_abschliessen(ma)
                continue

            job = zuordnung[ma]
            job_tasks = self._hole_job_tasks(job)
            if not job_tasks:
                self._bearbeite_job_ohne_taetigkeiten(
                    ma,
                    aktueller_tag,
                    runden_index,
                    aktuelle_runde,
                    job,
                )
                self._runde_abschliessen(ma)
                continue

            stoerung = self._ermittle_stoerung(aktueller_tag, runden_index, job)
            if stoerung is not None:
                self._bearbeite_stoerung(
                    ma,
                    aktueller_tag,
                    runden_index,
                    aktuelle_runde,
                    job,
                    stoerung,
                )
                self._runde_abschliessen(ma)
                continue

            zeit = self._arbeite_job(
                aktueller_tag,
                ma,
                job,
                runden_index,
                fehlerfrei,
                job_tasks,
            )
            if zeit < 0:
                if self._simulation_abgeschlossen(ma):
                    return
                return

            self.zustand_arbeitszeit[ma] = max(
                0.0, self.zustand_arbeitszeit[ma] - zeit
            )
            self.runde_restzeit[ma] = max(0.0, self.runde_restzeit[ma] - zeit)

            if self.runde_restzeit[ma] <= 0:
                self._fuehre_pause_durch(
                    ma, aktuelle_runde.get("pause", 0), self.letzte_iteration[ma]
                )
                self._runde_abschliessen(ma)

    def run(                                                    # führt die komplette Simulation aus, einmal mit und einmal ihne Fehler
        self,
        *,
        fehlerfrei: bool = False,
        ziel_tag_limit: Optional[int] = None,
        ignoriere_ziele: bool = False,
    ) -> Simulationsergebnis:
        self._fehlerfrei_flag = fehlerfrei
        self._ziele_pruefen_deaktiviert = bool(ignoriere_ziele)
        self._externes_tag_limit = (
            int(ziel_tag_limit)
            if ziel_tag_limit is not None and ziel_tag_limit > 0
            else None
        )
        self._simulation_stop_tag = None
        if self._ziele_pruefen_deaktiviert:
            self._ziel_status = None
        self._letzter_simulationstag = None
        for ma in self.aktive_mitarbeitende:
            self._beginne_neuen_tag(ma)

        abgeschlossene_ma: Set[str] = set()
        while True:
            alle_fertig = True
            for ma in self.aktive_mitarbeitende:
                if self._simulation_abgeschlossen(ma):
                    if ma not in abgeschlossene_ma:
                        self.logger.info(f"Simulation beendet (Mitarbeiter: {ma})")
                        abgeschlossene_ma.add(ma)
                    continue
                alle_fertig = False
                self._simuliere_bis_tagesende(ma, fehlerfrei)
            if alle_fertig:
                break

        self._fehlerfrei_flag = False
        self._ziele_pruefen_deaktiviert = False
        self._externes_tag_limit = None
        self._finalisiere_output()
        return Simulationsergebnis(             # Output der Simulation = alle Ergebnisse, Sheets, etc.
            self.output_data_all,
            self.job_history,
            self.kompetenz_protokoll,
            ziel_status=self._ziel_status,
            letzter_simulationstag=self._letzter_simulationstag,
            uebungsfaktor_protokoll={
                label: list(eintraege)
                for label, eintraege in self.uebungsfaktor_protokoll.items()
            },
        )

    def _arbeite_job(                                           # simuliert einen Arbeitsdurchlauf: bestimmt Tätigkeit, Lern-/Fehlerparameter,  
        self,                                                   # verarbeitet Output/Ausschuss, aktualisiert Tabellen, Vergessenswerte und 
        tag: int,                                               # Protokolle und gibt die benötigte Zeit zurück
        ma: str,
        job: str,
        runden_index: int,
        fehlerfrei: bool,
        job_tasks: List[str],
    ) -> float:
        if self._simulation_abgeschlossen(ma):
            return 0.0
        label = self._aktuelles_label(ma)
        self.job_task_index.setdefault(ma, {})
        self.job_durchsatz.setdefault(ma, {})
        if job not in self.job_task_index[ma]:
            self.job_task_index[ma][job] = 0
        if job not in self.job_durchsatz[ma]:
            self.job_durchsatz[ma][job] = None
        task_position = self.job_task_index[ma][job] % len(job_tasks)
        taetigkeit = job_tasks[task_position]
        aktuelle_afz = self._lernen(ma, taetigkeit, self.AFA_pre[ma][taetigkeit])
        aktuelle_afz = self._wende_fehlerarme_ausfuehrungszeit_an(ma, taetigkeit, aktuelle_afz)
        parameter = self.kompetenz_parameter[ma][taetigkeit]
        differenz = parameter["differenz"]
        if differenz > 0:
            reduktion_rel = max(
                0,
                min((parameter["initial"] - aktuelle_afz) / differenz, 1),
            )
        else:
            reduktion_rel = 1 if aktuelle_afz <= parameter["initial"] else 0
        if fehlerfrei:
            kompetenzstufe = 5
            fehlerquote = self.FEHLERARM_FEHLERQUOTE
        else:
            kompetenzstufe = self._ermittle_kompetenzstufe(ma, taetigkeit, aktuelle_afz)
            fehlerquote = self._ermittle_fehlerquote(
                ma, taetigkeit, kompetenzstufe, reduktion_rel
            )
        fehlerquote = max(0.0, min(fehlerquote, 1.0))
        output_geplant = max(0.0, self._ermittle_outputmenge(ma, taetigkeit))
        if aktuelle_afz > self.zustand_arbeitszeit[ma]:
            if not self._beende_arbeitstag(ma, tag):
                return -1.0
            return -1.0

        job_daten = self.job_durchsatz[ma]
        aktueller_job_input = job_daten.get(job)
        ist_letzte_taetigkeit = task_position == len(job_tasks) - 1
        ist_erster_schritt = task_position == 0 or aktueller_job_input is None
        if ist_erster_schritt:
            output_input = output_geplant
        else:
            output_input = max(0.0, aktueller_job_input)
        ausschuss = output_input * fehlerquote
        output_gut = output_input - ausschuss
        job_daten[job] = output_gut
        if ist_letzte_taetigkeit:
            job_daten[job] = None
        self.job_task_index[ma][job] = (task_position + 1) % len(job_tasks)

        i = self.durchlauf_index_person[label]
        df_map = self.output_data_all[label]
        for t in self.taetigkeiten:
            df_map[t].loc[i, "DurchlaufNr"] = i

        self.AFZ[ma][taetigkeit] = aktuelle_afz
        self._add_simulationszeit(ma, aktuelle_afz)
        self.ausgefuehrt[ma][taetigkeit] = 1
        self.m_gesamt_wiederholungen[ma][taetigkeit] = (
            self.m_gesamt_wiederholungen[ma].get(taetigkeit, 0) + 1
        )

        for t in self.taetigkeiten:
            df = df_map[t]
            ist_ausgefuehrt = 1 if t == taetigkeit else 0
            df.loc[i, "Ausgefuehrt"] = ist_ausgefuehrt
            df.loc[i, "Output_input"] = output_input if ist_ausgefuehrt else 0
            df.loc[i, "Output_geplant"] = output_geplant if ist_ausgefuehrt else 0
            df.loc[i, "Output_gut"] = output_gut if ist_ausgefuehrt else 0
            df.loc[i, "Ausschuss"] = ausschuss if ist_ausgefuehrt else 0
            vorheriger_ausschuss = 0
            if i > 0 and "Ausschuss_kumuliert" in df.columns:
                vorheriger_ausschuss = df.loc[i - 1, "Ausschuss_kumuliert"]
                if pd.isna(vorheriger_ausschuss):
                    vorheriger_ausschuss = 0
            df.loc[i, "Ausschuss_kumuliert"] = vorheriger_ausschuss + (
                ausschuss if ist_ausgefuehrt else 0
            )
            vorheriger_output = 0
            if i > 0 and "Output_geplant_kumuliert" in df.columns:
                vorheriger_output = df.loc[i - 1, "Output_geplant_kumuliert"]
                if pd.isna(vorheriger_output):
                    vorheriger_output = 0
            df.loc[i, "Output_geplant_kumuliert"] = vorheriger_output + (
                output_geplant if ist_ausgefuehrt else 0
            )
            df.loc[i, "Fehlerquote"] = fehlerquote if ist_ausgefuehrt else 0
            df.loc[i, "Sim_zeit"] = self.simulationszeit[ma]

        df_map[taetigkeit].loc[i, "AFZ"] = aktuelle_afz
        df_map[taetigkeit].loc[i, "AFA_pre"] = self.AFA_pre[ma][taetigkeit]
        self.AFA_post[ma][taetigkeit] = self.AFA_pre[ma][taetigkeit] + 1
        df_map[taetigkeit].loc[i, "AFA_post"] = self.AFA_post[ma][taetigkeit]
        df_map[taetigkeit].loc[i + 1, "AFA_pre"] = self.AFA_post[ma][taetigkeit]
        self.AFZ_post[ma][taetigkeit] = self._lernen(
            ma, taetigkeit, self.AFA_post[ma][taetigkeit]
        )
        self.AFZ_post[ma][taetigkeit] = self._wende_fehlerarme_ausfuehrungszeit_an(
            ma,
            taetigkeit,
            self.AFZ_post[ma][taetigkeit],
        )
        self.AFZ_last[ma][taetigkeit] = self.AFZ_post[ma][taetigkeit]
        df_map[taetigkeit].loc[i, "AFZ_post"] = self.AFZ_post[ma][taetigkeit]
        df_map[taetigkeit].loc[i + 1, "AFZ"] = self.AFZ_post[ma][taetigkeit]

        for t in self.taetigkeiten:
            if t != taetigkeit and self.ausgefuehrt[ma].get(t) == 1:
                self.vergessensdauer[ma][t] += aktuelle_afz
                self.AFZ_post[ma][t] = self._vergessen(
                    ma,
                    t,
                    self.vergessensdauer[ma][t],
                    self.AFZ_last[ma][t],
                    self.eingabe.ZI,
                )
                df_map[t].loc[i, "AFZ_post"] = self.AFZ_post[ma][t]
                df_map[t].loc[i + 1, "AFZ"] = self.AFZ_post[ma][t]
                self.AFA_post[ma][t] = self._berechne_AFA(ma, t, self.AFZ_post[ma][t])
                df_map[t].loc[i, "AFA_post"] = self.AFA_post[ma][t]
                df_map[t].loc[i + 1, "AFA_pre"] = self.AFA_post[ma][t]
            elif t == taetigkeit:
                self.vergessensdauer[ma][t] = 0.0
            df_map[t].loc[i, "VG_Dauer"] = self.vergessensdauer[ma][t]

        for t in self.taetigkeiten:
            self.AFA_pre[ma][t] = self.AFA_post[ma][t]
            self.AFZ[ma][t] = self.AFZ_post[ma][t]

        runde_nummer = runden_index + 1
        self.job_history[label].append(
            {
                "DurchlaufNr": i,
                "Runde": runde_nummer,
                "Job": job,
                "Tätigkeit": taetigkeit,
                "Bearbeitungszeit": aktuelle_afz,
                "Simulationszeit": self.simulationszeit[ma],
                "Tag": self.arbeitstag_zaehler[ma] + 1,
                "Kompetenzstufe": kompetenzstufe,
                "Reduktion_%": reduktion_rel * 100,
                "Fehlerquote": fehlerquote,
                "Output_input": output_input,
                "Output_geplant": output_geplant,
                "Output_gut": output_gut,
                "Ausschuss": ausschuss,
                "Letzte_Tätigkeit": ist_letzte_taetigkeit,
            }
        )
        self.durchlauf_index_person[label] += 1
        self.letzte_iteration[ma] = i
        return aktuelle_afz

    def _fuehre_pause_durch(self, ma: str, pause_dauer: float, index: Optional[int]) -> None:       # simuliert Pausen, trägt Vergessenseffekte in DataFrames ein und
        if pause_dauer <= 0:                                                                        # aktualisiert AFA-/AFZ-Werte, falls bereits mind. ein Durchlauf der Tätigkeit existiert
            return
        self.logger.info(f"Pause {pause_dauer/60:.0f} Minuten für {ma}")
        self._add_simulationszeit(ma, pause_dauer)
        if index is None:
            return
        label = self._aktuelles_label(ma)
        for t in self.taetigkeiten:
            if self.ausgefuehrt[ma].get(t) == 1:
                self.vergessensdauer[ma][t] += pause_dauer
                self.AFZ_post[ma][t] = self._vergessen(
                    ma,
                    t,
                    self.vergessensdauer[ma][t],
                    self.AFZ_last[ma][t],
                    self.eingabe.ZI,
                )
                df = self.output_data_all[label][t]
                df.loc[index, "AFZ_post"] = self.AFZ_post[ma][t]
                df.loc[index + 1, "AFZ"] = self.AFZ_post[ma][t]
                self.AFA_post[ma][t] = self._berechne_AFA(ma, t, self.AFZ_post[ma][t])
                df.loc[index, "AFA_post"] = self.AFA_post[ma][t]
                df.loc[index + 1, "AFA_pre"] = self.AFA_post[ma][t]
                self.AFA_pre[ma][t] = self.AFA_post[ma][t]
            self.output_data_all[label][t].loc[index, "VG_Dauer"] = self.vergessensdauer[ma][t]
            self.AFZ[ma][t] = self.AFZ_post[ma][t]

    def _beende_arbeitstag(self, ma: str, tag: int) -> bool:        # Arbeitstag wird als beendet markiert (Simulation wird beendet, falls Zähler für Betrachtungszeitraum erreicht wird), 
        index = self.letzte_iteration[ma]                           # Feierabend/Wochenend-Abwesenheiten und Kompetenzwerte werden verarbeitet
        if self._arbeitstag_beendet[ma]:                            # bei Simulation mit Ziel: aktualisiert Ziele und entscheidet, ob weitere Tage simuliert werden
            return True
        label = self._aktuelles_label(ma)
        self.logger.info(
            f"Arbeitstag {tag} beendet | Mitarbeiter: {ma} | Verbleibende Tage der Woche: {self.z1[ma]}"
        )
        wochenende_start = self.z1[ma] == 0
        feierabend_dauer = float(self.eingabe.Abwesenheit_1)
        self.m_break_stunden[ma] = 0.0
        if feierabend_dauer > 0:
            self._add_simulationszeit(ma, feierabend_dauer)
        self._verarbeite_abwesenheit(ma, feierabend_dauer, index, m_relevant=True)
        if not wochenende_start and self.z1[ma] > 0:
            self.z1[ma] -= 1
            self.z2[ma] = 1
        self.arbeitstag_zaehler[ma] += 1
        aktueller_tag = self.arbeitstag_zaehler[ma]
        for t in self.taetigkeiten:
            aktuelle_afz = self.AFZ_post[ma][t]
            kompetenzstufe = self._ermittle_kompetenzstufe(ma, t, aktuelle_afz)
            parameter = self.kompetenz_parameter[ma][t]
            differenz = parameter["differenz"]
            if differenz > 0:
                reduktion_rel = max(0, min((parameter["initial"] - aktuelle_afz) / differenz, 1))
            else:
                reduktion_rel = 1 if aktuelle_afz <= parameter["initial"] else 0
            fehlerquote_tag = 0.0
            if getattr(self, "_fehlerfrei_flag", False):
                fehlerquote_tag = self.FEHLERARM_FEHLERQUOTE
            else:
                fehlerquote_tag = self._ermittle_fehlerquote(ma, t, kompetenzstufe, reduktion_rel)
            self.kompetenz_protokoll[label].append(
                {
                    "Tag": aktueller_tag,
                    "Tätigkeit": t,
                    "AFZ": aktuelle_afz,
                    "Kompetenzstufe": kompetenzstufe,
                    "Reduktion_%": reduktion_rel * 100,
                    "Fehlerquote": fehlerquote_tag,
                    "Ereignis": "Feierabend",
                }
            )
            
        self._aktualisiere_uebungsfaktor_aufbau(ma, aktueller_tag, label)
        if wochenende_start:
            self.logger.info(f"Wochenende für {ma}")
            wochenende_differenz = max(
                float(self.eingabe.Abwesenheit_2)
                - feierabend_dauer
                - float(self.schichtdauer_pro_tag),
                0.0,
            )
            if wochenende_differenz > 0:
                self._add_simulationszeit(ma, wochenende_differenz)
            self._verarbeite_abwesenheit(
                ma, wochenende_differenz, index, m_relevant=True
            )
            for t in self.taetigkeiten:
                aktuelle_afz = self.AFZ_post[ma][t]
                kompetenzstufe = self._ermittle_kompetenzstufe(ma, t, aktuelle_afz)
                parameter = self.kompetenz_parameter[ma][t]
                differenz = parameter["differenz"]
                if differenz > 0:
                    reduktion_rel = max(0, min((parameter["initial"] - aktuelle_afz) / differenz, 1))
                else:
                    reduktion_rel = 1 if aktuelle_afz <= parameter["initial"] else 0
                fehlerquote_tag = 0.0
                if getattr(self, "_fehlerfrei_flag", False):
                    fehlerquote_tag = self.FEHLERARM_FEHLERQUOTE
                else:
                    fehlerquote_tag = self._ermittle_fehlerquote(ma, t, kompetenzstufe, reduktion_rel)
                self.kompetenz_protokoll[label].append(
                    {
                        "Tag": aktueller_tag,
                        "Tätigkeit": t,
                        "AFZ": aktuelle_afz,
                        "Kompetenzstufe": kompetenzstufe,
                        "Reduktion_%": reduktion_rel * 100,
                        "Fehlerquote": fehlerquote_tag,
                        "Ereignis": "Wochenende",
                    }
                )
            if self.z2[ma] > 0:
                self.z2[ma] -= 1

            self.z1[ma] = 4
        self._arbeitstag_beendet[ma] = True
        self._aktualisiere_personalstatus_nach_tag(ma)

        if self._pruefe_simulationsziele(aktueller_tag):
            if self.arbeitstag_zaehler[ma] > self._letzter_abgeschlossener_tag:
                self._letzter_abgeschlossener_tag = self.arbeitstag_zaehler[ma]
                self.after_day(self._letzter_abgeschlossener_tag)
            return False

        if self.simulationszeit[ma] >= self.simulationszeit_limit:
            if self._simulation_stop_tag is None:
                meldung = (
                    f"Maximale simulierte Dauer erreicht (Tag {aktueller_tag}, Limit {self.max_tage:g} Tage) – Simulation wird beendet."
                )
                self._set_simulationsziel_ergebnis(
                    aktueller_tag,
                    "Maximale Betrachtungsdauer",
                    float(aktueller_tag),
                    float(self.max_tage),
                    False,
                    details=self._tageskennzahlen.get(aktueller_tag),
                    meldung=meldung,
                )
            if self.arbeitstag_zaehler[ma] > self._letzter_abgeschlossener_tag:
                self._letzter_abgeschlossener_tag = self.arbeitstag_zaehler[ma]
                self.after_day(self._letzter_abgeschlossener_tag)
            return False

        self._beginne_neuen_tag(ma)
        if self.arbeitstag_zaehler[ma] > self._letzter_abgeschlossener_tag:
            self._letzter_abgeschlossener_tag = self.arbeitstag_zaehler[ma]
            self.after_day(self._letzter_abgeschlossener_tag)
        return True

        if self.arbeitstag_zaehler[ma] > self._letzter_abgeschlossener_tag:
            self._letzter_abgeschlossener_tag = self.arbeitstag_zaehler[ma]
            self.after_day(self._letzter_abgeschlossener_tag)
        return True

    def _set_simulationsziel_ergebnis(      # registriert den Tag, an dem ein Simulationsziel erstmals erreicht und verhindert, dass spätere Ereignisse diesen Status überschreiben
        self,                               # sobald ein gültiger Eintrag vorliegt, aktualisiert sie den internen Stopp-Tag 
        tag: int,
        ziel: str,
        wert: Optional[float],
        schwelle: Optional[float],
        erreicht: bool,
        *,
        details: Optional[Dict[str, float]] = None,
        meldung: Optional[str] = None,
    ) -> None:
        if self._simulation_stop_tag is not None:
            if tag > self._simulation_stop_tag:
                return
            if tag == self._simulation_stop_tag and self._ziel_status is not None:
                return
        self._simulation_stop_tag = tag
        status: Dict[str, Any] = {
            "tag": tag,
            "ziel": ziel,
            "wert": wert,
            "schwelle": schwelle,
            "erreicht": erreicht,
        }
        if details:
            status.update(details)
        self._ziel_status = status
        if meldung:
            self.logger.info(meldung)

    def _berechne_durchschnittliche_fehlerquote(self, tag: int) -> Optional[float]:     # aggregiert Feierabend-Fehlerquoten über alle Personen für einen Tag und liefert den Mittelwert 
        werte: List[float] = []
        for protokoll in self.kompetenz_protokoll.values():
            for eintrag in protokoll:
                if eintrag.get("Tag") != tag or eintrag.get("Ereignis") != "Feierabend":
                    continue
                wert = eintrag.get("Fehlerquote")
                try:
                    if wert is not None:
                        werte.append(float(wert))
                except (TypeError, ValueError):
                    continue
        if not werte:
            return None
        return float(mean(werte))

    def _berechne_tagesproduktivitaet(self, tag: int) -> Optional[float]:     # Gesamtoutput (ohne Defekt) eines Tages wird ermittelt
        return berechne_outputsumme_fuer_tag(self.job_history, tag)

    def _berechne_kompetenzmetriken(self, tag: int) -> Optional[Dict[str, float]]:     # berechnet Mittelwert und Varianz der Kompetenzstufen zum Feierabend eines Tages 
        werte: List[float] = []
        for protokoll in self.kompetenz_protokoll.values():
            for eintrag in protokoll:
                if eintrag.get("Tag") != tag or eintrag.get("Ereignis") != "Feierabend":
                    continue
                wert = eintrag.get("Kompetenzstufe")
                try:
                    if wert is not None:
                        werte.append(float(wert))
                except (TypeError, ValueError):
                    continue
        if not werte:
            return None
        durchschnitt = float(mean(werte))
        varianz = float(pvariance(werte)) if len(werte) > 1 else 0.0
        return {
            "kompetenz_durchschnitt": durchschnitt,
            "kompetenz_varianz": varianz,
        }

    def _pruefe_simulationsziele(self, aktueller_tag: int) -> bool:     # evaluierte alle aktiven Zielschwellen für den aktuellen Tag anhand seiner Kennzahlen
        extern_limit = self._externes_tag_limit                         # (Produktivität, Kompetenz, Qualität oder maximale Laufzeit) und entscheidet, ob die Simulation beendet wird
        if extern_limit is not None and aktueller_tag >= extern_limit:
            if self._simulation_stop_tag is None:
                self._simulation_stop_tag = extern_limit
            return True

        if self._simulation_stop_tag is not None:
            return False
        if aktueller_tag <= 0:
            return False
        if any(
            self.arbeitstag_zaehler.get(ma, 0) < aktueller_tag
            for ma in self.aktive_mitarbeitende
        ):
            return False

        if not self._ziele_aktiv or self._ziele_pruefen_deaktiviert:
            return False

        metriken: Dict[str, float] = {}
        fehler_mittel = self._berechne_durchschnittliche_fehlerquote(aktueller_tag)
        if fehler_mittel is not None:
            metriken["fehlerquote"] = fehler_mittel
        produktivitaet_tag = self._berechne_tagesproduktivitaet(aktueller_tag)
        if produktivitaet_tag is not None:
            metriken["produktivitaet"] = produktivitaet_tag
        kompetenz_metriken = self._berechne_kompetenzmetriken(aktueller_tag)
        if kompetenz_metriken:
            metriken.update(kompetenz_metriken)
        if metriken:
            self._tageskennzahlen[aktueller_tag] = dict(metriken)

        fehler_schwelle = self._ziele_config.get("fehlerquote")
        if (
            fehler_schwelle is not None
            and fehler_mittel is not None
            and fehler_mittel <= fehler_schwelle
        ):
            meldung = (
                f"Simulationsziel erreicht (Tag {aktueller_tag}): "
                f"Durchschnittliche Fehlerquote {fehler_mittel:.4f} <= {fehler_schwelle:.4f}."
            )
            self._set_simulationsziel_ergebnis(
                aktueller_tag,
                "Fehlerquote",
                fehler_mittel,
                fehler_schwelle,
                True,
                details=metriken,
                meldung=meldung,
            )
            return True

        produktiv_schwelle = self._ziele_config.get("produktivitaet")
        if (
            produktiv_schwelle is not None
            and produktivitaet_tag is not None
            and produktivitaet_tag >= produktiv_schwelle
        ):
            meldung = (
                f"Simulationsziel erreicht (Tag {aktueller_tag}): "
                f"Gesamtproduktivität {produktivitaet_tag:.2f} >= {produktiv_schwelle:.2f}."
            )
            self._set_simulationsziel_ergebnis(
                aktueller_tag,
                "Produktivität",
                produktivitaet_tag,
                produktiv_schwelle,
                True,
                details=metriken,
                meldung=meldung,
            )
            return True

        kompetenz_schwelle = self._ziele_config.get("kompetenz_durchschnitt")
        varianz_schwelle = self._ziele_config.get("kompetenz_varianz")
        if (
            kompetenz_schwelle is not None
            and varianz_schwelle is not None
            and kompetenz_metriken
        ):
            durchschnitt = kompetenz_metriken["kompetenz_durchschnitt"]
            varianz = kompetenz_metriken["kompetenz_varianz"]
            if durchschnitt >= kompetenz_schwelle and varianz <= varianz_schwelle:
                meldung = (
                    f"Simulationsziel erreicht (Tag {aktueller_tag}): "
                    f"Durchschnittliche Kompetenz {durchschnitt:.2f} >= {kompetenz_schwelle:.2f} "
                    f"und Varianz {varianz:.4f} <= {varianz_schwelle:.4f}."
                )
                self._set_simulationsziel_ergebnis(
                    aktueller_tag,
                    "Kompetenzniveau",
                    durchschnitt,
                    kompetenz_schwelle,
                    True,
                    details=metriken,
                    meldung=meldung,
                )
                return True

        max_tage_ziel = self._ziele_config.get("max_tage")
        if max_tage_ziel is not None and aktueller_tag >= max_tage_ziel:
            meldung = (
                f"Maximaler Betrachtungszeitraum erreicht (Tag {aktueller_tag} von {int(max_tage_ziel)}): Simulation wird beendet."
            )
            self._set_simulationsziel_ergebnis(
                aktueller_tag,
                "Maximaler Betrachtungszeitraum",
                float(aktueller_tag),
                max_tage_ziel,
                False,
                details=metriken,
                meldung=meldung,
            )
            return True

        return False

    def _verarbeite_abwesenheit(     # verarbeitet eine Abwesenheitsdauer für jeden Arbeiter und entscheidet, ob der Übungsfaktor verändert werden muss
        self,
        ma: str,
        dauer: float,
        index: Optional[int],
        *,
        m_relevant: bool = False,
    ) -> None:
        if m_relevant and dauer > 0:
            self._registriere_m_unterbrechung(ma, dauer)
        if dauer <= 0 or index is None:
            return
        label = self._aktuelles_label(ma)
        for t in self.taetigkeiten:

            if self.ausgefuehrt[ma].get(t) == 1:
                self.vergessensdauer[ma][t] += dauer
                self.AFZ_post[ma][t] = self._vergessen(
                    ma,
                    t,
                    self.vergessensdauer[ma][t],
                    self.AFZ_last[ma][t],
                    self.eingabe.ZI,
                )
                df = self.output_data_all[label][t]
                df.loc[index, "AFZ_post"] = self.AFZ_post[ma][t]
                df.loc[index + 1, "AFZ"] = self.AFZ_post[ma][t]
                self.AFA_post[ma][t] = self._berechne_AFA(ma, t, self.AFZ_post[ma][t])
                df.loc[index, "AFA_post"] = self.AFA_post[ma][t]
                df.loc[index + 1, "AFA_pre"] = self.AFA_post[ma][t]
                self.AFA_pre[ma][t] = self.AFA_post[ma][t]
            self.output_data_all[label][t].loc[index, "VG_Dauer"] = self.vergessensdauer[ma][t]
            self.AFZ[ma][t] = self.AFZ_post[ma][t]

    def _finalisiere_output(self) -> None:      # schließt die Übungsfaktorprotokolle ab, setzt den letzten Simulationstag 
        self._finalisiere_uebungsfaktor_log()   # und trimmt, falls nötig, die protokollierte Ergebnisse außerhalb des betrachteten Zeitraums
        cutoff_tag = self._simulation_stop_tag or self._letzter_abgeschlossener_tag
        if cutoff_tag:
            cutoff_int = int(cutoff_tag)
            self._letzter_simulationstag = cutoff_int
            self._trim_ergebnisse_bis_tag(cutoff_int)
        else:
            self._letzter_simulationstag = None

    def _finalisiere_uebungsfaktor_log(self) -> None:     # ausstehende Übungsfaktor-Einträge und schließt Übungsfaktorprotokollierung ab 
        for ma in self.aktive_mitarbeitende:
            tag = self.m_tag_index.get(ma)
            label = self.m_tag_label.get(ma, self._aktuelles_label(ma))
            if tag is None or label is None:
                continue
            q_stunden = max(self.m_break_stunden.get(ma, 0.0), 0.0)
            for taetigkeit in self.taetigkeiten:
                m_abbau_alt = max(
                    self.m_abbau_tag[ma].get(taetigkeit, self.m_min), self.m_min
                )
                m_aufbau = max(
                    self.m_aufbau_tag[ma].get(taetigkeit, m_abbau_alt), self.m_min
                )
                eintrag = {
                    "Tag": tag,
                    "Mitarbeiter": label,
                    "Tätigkeit": taetigkeit,
                    "m_abbau": float(m_abbau_alt),
                    "m_aufbau": float(m_aufbau),
                    "q_stunden": float(q_stunden),
                    "n_ges": int(self.m_gesamt_wiederholungen[ma].get(taetigkeit, 0)),
                    "Komplexität": float(
                        self.taetigkeit_komplexitaet.get(taetigkeit, 1.0)
                    ),
                }
                self.uebungsfaktor_protokoll.setdefault(label, []).append(eintrag)
            self.m_tag_index[ma] = None
        for label, taetigkeiten in self.output_data_all.items():
            for t in self.taetigkeiten:
                df = taetigkeiten[t]
                if df.empty:
                    continue

                df = df.copy()
                if "AFZ" in df.columns:
                    df["AFZ"] = pd.to_numeric(df["AFZ"], errors="coerce")
                if "AFZ_post" in df.columns:
                    df["AFZ_post"] = pd.to_numeric(df["AFZ_post"], errors="coerce")
                if "AFZ" in df.columns and "AFZ_post" in df.columns:
                    df["AFZ"] = df["AFZ"].combine_first(df["AFZ_post"])
                    df["AFZ"] = df["AFZ"].ffill()

                if not df.empty:
                    letzte_zeile = df.iloc[-1]
                    durchlauf_na = pd.isna(letzte_zeile.get("DurchlaufNr"))
                    if durchlauf_na:
                        relevante_werte = letzte_zeile.drop(
                            labels=[
                                "AFA_pre",
                                "AFZ",
                                "AFZ_post",
                                "VG_Dauer",
                            ],
                            errors="ignore",
                        ).dropna()
                        if relevante_werte.empty:
                            df = df.iloc[:-1].copy()
                if df.empty:
                    self.output_data_all[label][t] = df
                    continue

                df["Output_input"] = pd.to_numeric(df.get("Output_input"), errors="coerce").fillna(0.0)
                df["Ausschuss"] = pd.to_numeric(df.get("Ausschuss"), errors="coerce").fillna(0.0)
                df["Ausschuss_kumuliert"] = df["Ausschuss"].cumsum()
                df["Output_geplant"] = pd.to_numeric(df.get("Output_geplant"), errors="coerce").fillna(0.0)
                df["Output_geplant_kumuliert"] = df["Output_geplant"].cumsum()

                self.output_data_all[label][t] = df

    def _trim_ergebnisse_bis_tag(self, max_tag: int) -> None:       # trimmt Ergebnisse, die außerhalb des relevanten Zeitraums liegen,  
        if max_tag <= 0:                                            # z.B. falls eine Runde vom Folgetag nach dem Betrachtungszeitraum simuliert wird
            return

        def _tagwert(raw: Any) -> Optional[int]:     # Hilfsfunktion, die beliebige Eingaben (z. B. Strings, Floats oder bereits numerische Werte) robust in ganze Tagesnummern überführt
            if raw is None:
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return None

        for label, historie in self.job_history.items():
            gefiltert: List[Dict[str, Any]] = []
            for eintrag in historie:
                tag_wert = _tagwert(eintrag.get("Tag"))
                if tag_wert is None or tag_wert <= max_tag:
                    gefiltert.append(eintrag)
            self.job_history[label] = gefiltert

        for label, protokoll in self.kompetenz_protokoll.items():
            gefiltert_protokoll: List[Dict[str, Any]] = []
            for eintrag in protokoll:
                tag_wert = _tagwert(eintrag.get("Tag"))
                if tag_wert is None or tag_wert <= max_tag:
                    gefiltert_protokoll.append(eintrag)
            self.kompetenz_protokoll[label] = gefiltert_protokoll

        self._tageskennzahlen = {
            tag: werte for tag, werte in self._tageskennzahlen.items() if tag <= max_tag
        }