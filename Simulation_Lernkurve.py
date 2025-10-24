class Lernkurve:                # Klasse "Lernkurve" mit allen notwendigen Parametern
    def __init__(
        self,
        t_initiale_AFZ,
        M,
        k,
        prozesstaetigkeiten,
        *,
        m_faktoren=None,
        m_min=1.0,
        m_max=5.0,
    ):
        # Initialisiert die de-Jong-Lernkurve für einen Mitarbeiter

        self.t_initiale_AFZ = t_initiale_AFZ  # Initiale Bearbeitungsdauer
        self.M = M      # Dictionary mit Grenzwerten M für jede Tätigkeit
        self.k = k      # Dictionary mit Lernfaktoren für jede Tätigkeit
        self.prozesstaetigkeiten = prozesstaetigkeiten  # Liste der Tätigkeiten
        self.m_min = float(m_min)   # Berechnung des Übungsfaktors
        self.m_max = float(m_max) if m_max is not None else float(m_min)
        if self.m_max < self.m_min:
            self.m_max = self.m_min
        self.m_faktoren = {}
        default_m = self._clamp_m(self.m_min)
        if m_faktoren is None:
            m_faktoren = {}
        for taetigkeit in self.prozesstaetigkeiten:
            wert = m_faktoren.get(taetigkeit, default_m)
            self.m_faktoren[taetigkeit] = self._clamp_m(wert)


    def _clamp_m(self, wert):   # Übungsfaktor wird zwischen mindestens m_min und maximal m_max gehalten
        try:
            wert = float(wert)
        except (TypeError, ValueError):
            wert = self.m_min
        if wert < self.m_min:
            return self.m_min
        if wert > self.m_max:
            return self.m_max
        return wert

    def set_m_faktor(self, prozesstaetigkeit, wert): # speichert einen geclampeten Wert für eine Tätigkeit
        self.m_faktoren[prozesstaetigkeit] = self._clamp_m(wert)

    def get_m_faktor(self, prozesstaetigkeit): # liefert den hinterlegten Wert oder den Minimalfaktor
        return self.m_faktoren.get(prozesstaetigkeit, self.m_min)

    def get_m_faktoren(self): # gibt eine Kopie des internen Dictionaries zurück, um externe Mutationen zu vermeiden
        return dict(self.m_faktoren)

    def berechne_ausfuehrungszeit(self, prozesstaetigkeit, wiederholungen):
        # Berechnet die Ausführungszeit für eine bestimmte Prozesstätigkeit nach Dejong.

        t_initiale_AFZ = self.t_initiale_AFZ[prozesstaetigkeit]
        M = self.M[prozesstaetigkeit]
        k = self.k[prozesstaetigkeit]
        m = self.get_m_faktor(prozesstaetigkeit)
        s = max(float(wiederholungen), 0.0)

        # Dejong-Lernkurvenformel mit Erfahrungs-/ Übungsfaktor m
        t = t_initiale_AFZ * M + t_initiale_AFZ * (1 - M) * (s ** (m * k))
        return t

    def berechne_ausfuehrungsanzahl(self, prozesstaetigkeit, AFZ_nach_Vergessen):   # Umkehrfunktion ermittelt aus einer gegebenen Ausführungszeit nach Vergessen 
                                                                                    #  die effektive Anzahl absolvierter Wiederholungen
        t_initiale_AFZ = self.t_initiale_AFZ[prozesstaetigkeit]
        M = self.M[prozesstaetigkeit]
        k = self.k[prozesstaetigkeit]
        m = self.get_m_faktor(prozesstaetigkeit)

        nenner = t_initiale_AFZ * (1 - M)
        if nenner == 0:
            return 0.0
        basis = (AFZ_nach_Vergessen - t_initiale_AFZ * M) / nenner
        exponent = m * k
        if exponent == 0:
            return 0.0
        if basis <= 0:
            return 0.0
        Anzahl = basis ** (1 / exponent)  # aktuelle Anzahl der Wiederholungen nach dem Vergessen

        return Anzahl