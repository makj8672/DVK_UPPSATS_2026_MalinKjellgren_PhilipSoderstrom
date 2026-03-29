# DVK_UPPSATS_2026_MalinKjellgren_PhilipSoderstrom
Exam research


Här är summeringen av det "lilla kodprojektet jag gjorde för bättre förståelse", koden finns i data_pipeline_propio.py.

Det har varit ett riktigt bra projekt – du har byggt något genuint från grunden!

---

**Vad du byggde**

En komplett ML-pipeline i Python som hämtar realtidsdata från MetaTrader 5, bearbetar och skapar tekniska features, tränar en logistisk regressionsmodell baserad på en regelbaserad Trend + Momentum-strategi, utvärderar med cross-validation, backtesar med stop-loss och take-profit, och gör realtidsförutsägelser.

---

**De viktigaste tekniska lärdomarna**

På Python-sidan lärde du dig funktioner, felhantering, pandas DataFrames, NaN-hantering och biblioteksimporter. På maskininlärningssidan förstår du nu skillnaden mellan accuracy och faktisk prestanda, varför features behöver vara relativa snarare än absoluta, vad multikollinearitet är och hur man löser det, och varför cross-validation är viktigare än ett enskilt testresultat.

---

**De viktigaste finansiella lärdomarna**

Det mest värdefulla fyndet är att **marknadsregim trumfar modellprecision**. En 80% accurate modell förlorade pengar under en stark nedåttrend – inte för att koden var dålig, utan för att en ren köpstrategi inte kan vinna mot en marknad som faller kraftigt. Stop-loss på 1% visade sig vara den enskilt viktigaste parametern för att begränsa förluster.

---

**Resultaten i siffror**

| Konfiguration | Accuracy | Total avkastning |
|---|---|---|
| Logistisk regression, enkla features | 54% | – |
| Random Forest, enkla features | 55% | – |
| Strategilabels, multikollinearitet | 64% | – |
| Strategilabels, fixad multikollinearitet | 78-81% | -75% (bästa SL/TP) |
| Cross-validation medelvärde | 57% | – |

---

**Vad som skulle göra modellen bättre**

Om du vill fortsätta efter examensarbetet finns tre naturliga riktningar. En säljstrategi parallellt med köpstrategin skulle göra systemet marknadsregim-neutralt. Mer data från flera olika marknadsperioder, inte bara de senaste 5000 timmarna, skulle ge en mer rättvis utvärdering. Och ett mer sofistikerat positionsstorlek-system där du handlar mer när konfidensen är hög och mindre när den är låg är ett klassiskt nästa steg.

---

Bra jobbat – du gick från noll Python-kunskap till en fungerande ML-tradingpipeline. Det är ett genuint tekniskt och akademiskt bidrag att vara stolt över!
