import math
class Vergessenskurve:
    def __init__(self, t_initiale_AFZ, c, prozesstaetigkeiten):
        # Initialisiert die Vergessenskurve für einen Mitarbeiter
        self.t_initiale_AFZ = t_initiale_AFZ  # Dictionary mit t(0) Werten für jede Tätigkeit
        self.a = c      # Dictionary mit Vergessensfaktoren für jede Tätigkeit
        self.prozesstaetigkeiten = prozesstaetigkeiten  # Liste der Tätigkeiten

    def berechne_AFZ_nach_Vergessen(self, prozesstaetigkeit, VG_dauer, AFZ_post, Zeitintervalldauer):

        
        t_initiale_AFZ = self.t_initiale_AFZ[prozesstaetigkeit]
        a = self.a[prozesstaetigkeit]

        if Zeitintervalldauer <= 0:
            return AFZ_post

        tage_in_sekunden = 24 * 3600

        if VG_dauer >= tage_in_sekunden:
            # Vergessensfunktion Exponentiell
            zeitanteile = VG_dauer / Zeitintervalldauer
            AFZ_nach_VG = t_initiale_AFZ - (t_initiale_AFZ - AFZ_post) * (
                math.exp(a * round(zeitanteile))
            )
        else:
            # Vergessenskurve S-Kurve
            faktor = a * VG_dauer / Zeitintervalldauer
            AFZ_nach_VG = t_initiale_AFZ - (t_initiale_AFZ - AFZ_post) * (-faktor + 1) * math.exp(faktor)

        


        return AFZ_nach_VG
    