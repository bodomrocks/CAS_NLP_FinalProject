# Automatische Erkennung von Funktionsverbgefügen in deutschen Rechtstexten

**Projektbericht**  
Datum: Juni 2026  
Modell: [boj-per/fvg-gbert-base](https://huggingface.co/boj-per/fvg-gbert-base)  
Applikation: [HF Spaces – FVG-Erkennung](https://huggingface.co/spaces/boj-per/legal_fvg2)

---

## 1. Einleitung

### 1.1 Funktionsverbgefüge in der deutschen Sprache

Funktionsverbgefüge (FVG) sind syntaktisch-semantische Konstruktionen, die aus einem bedeutungsarmen Verb – dem sogenannten Funktionsverb – und einem nominalen Element bestehen, das den semantischen Kern trägt. Das Funktionsverb hat in dieser Konstruktion seine volle lexikalische Bedeutung weitgehend eingebüsst und übernimmt grammatische Funktionen wie Aspektmarkierung, Diathese oder Valenz.

**Typische Beispiele:**

| FVG | Bedeutung | Funktionsverb | Nominalteil |
|---|---|---|---|
| *Beschwerde erheben* | beschweren | erheben | Beschwerde |
| *in Erwägung ziehen* | erwägen | ziehen | in Erwägung |
| *zur Anwendung kommen* | angewendet werden | kommen | zur Anwendung |
| *Gehör gewähren* | anhören | gewähren | Gehör |

Im Deutschen ist die Abgrenzung von FVG gegenüber freien Verbindungen (z.B. *Wasser nehmen*, *einen Antrag stellen* vs. *Bücher stellen*) nicht immer eindeutig. Linguistisch relevante Kriterien sind:

- **Semantische Entleerung des Verbs**: Das Verb trägt keine autonome Bedeutung.
- **Nominalisierungsparaphrase**: Ein FVG lässt sich durch ein Derivat paraphrasieren (*eine Entscheidung treffen* → *entscheiden*).
- **Eingeschränkte Modifizierbarkeit**: Der Nominalteil ist kaum erweiterbar (*\*eine kluge Entscheidung treffen* in der FVG-Lesart).
- **Diskontinuität**: Verb und Nominalteil können durch zahlreiche Elemente getrennt sein (*nimmt die Einwände aller Beteiligten in Betracht*).

### 1.2 FVG in der Rechtsprache

In juristischen Texten sind FVG überproportional häufig. Dies liegt an strukturellen Merkmalen der Rechtsprache:

- **Nominalisierungsstil**: Rechtliche Sachverhalte werden bevorzugt als Nomina ausgedrückt (*Klageerhebung*, *Beschwerdeführung*), was FVG als Prädikationsform begünstigt.
- **Aspektualität**: FVG erlauben feine Distinktionen hinsichtlich Aspekt und Aktionsart (*in Kraft treten* [ingressiv] vs. *in Kraft sein* [stativ]).
- **Registerpräferenz**: FVG sind Merkmale gehobenen, formellen Registers.
- **Standardisierung**: Bestimmte FVG sind in juristischen Texten nahezu obligatorisch (*Beschwerde erheben*, *Antrag stellen*, *Stellung nehmen*).

---

## 2. Datenbasis

### 2.1 Korpus

| Quelle | Dateien | Dateityp | Beschreibung |
|---|---|---|---|
| Bundesgerichtsentscheide (BGE) | 40'930 | HTML | Schweizer Bundesgericht, mehrere Jahrzehnte |
| Kantonale Entscheide | 724 | TXT | Urteile verschiedener kantonaler Gerichte |

Für das Training wurden **5'000 BGE-Dateien** (zufällige Stichprobe) sowie alle 724 kantonalen Dateien verwendet, was einen repräsentativen Querschnitt der schweizerischen Rechtsprache abdeckt.

Nach der Textextraktion (Entfernung von HTML-Tags, Normalisierung) wurden mittels spaCy (`de_core_news_lg`) **829'059 Sätze** segmentiert und lemmatisiert. Berücksichtigt wurden Sätze mit 5–200 Token.

### 2.2 FVG-Referenzlisten

Als Ausgangspunkt dienten drei Ressourcen:

**liste_fvg.txt** – eine kuratierte Liste von **209 FVG**, die typische Konstruktionen der deutschen Rechtsprache abdecken:
- Einfache Nominalphrasen: *Beschwerde erheben*, *Antrag stellen*
- Präpositionalphrasen: *in Erwägung ziehen*, *zur Anwendung kommen*
- Komplexe Konstruktionen: *in Wiedererwägung ziehen*, *unter Beweis stellen*

**verb_noun_fvg.csv** – eine Frequenzliste von **300 Verb-Nomen-Paaren** aus dem Korpus mit manueller FVG-Beurteilung (0/1), die als Grundlage für die manuelle Evaluation diente (vgl. Abschnitt 4.3).

**praep_noun_verb.csv** – eine Frequenzliste von **200 Einträgen** (163 nach Deduplizierung) mit präpositionalen FVG-Mustern aus dem Korpus (*in Frage kommen*, *in Betracht ziehen* etc.), ebenfalls für die manuelle Evaluation verwendet.

### 2.3 Gold-Annotationsdatensatz

Im Rahmen des Projekts wurden **1'149 Sätze** aus dem Korpus manuell annotiert (vgl. Abschnitt 3.5). Diese bilden den Gold-Datensatz:

| Klasse | Sätze | Anteil |
|---|---|---|
| FVG (manuell_FVG = 1) | 386 | 33.6% |
| kein FVG (manuell_FVG = 0) | 759 | 66.1% |
| unklar (manuell_FVG = ?) | 4 | 0.3% |

---

## 3. Methodik

### 3.1 Gesamtarchitektur

Das Projekt folgt einer erweiterten Pipeline mit manuellem Evaluations- und Feedbackschritt:

```
Korpus (HTML/TXT)
       ↓  Schritt 1: Textextraktion
Rohtexte (JSONL)
       ↓  Schritt 2: Zweipass-Annotation (Distant Supervision)
Annotierte Trainingsdaten (BIO-Format, distant)
       ↓  Schritt 3: Modell-Finetuning (gbert-base)
FVG-Tokenklassifikator v1
       ↓  Schritt 5/5b: Evaluationssätze extrahieren
Manuelle Annotation (1'149 Sätze) → Gold-Datensatz
       ↓  Schritt 7: Modell-Evaluation gegen Gold
Fehleranalyse → FUNKTIONSVERBEN erweitern + Schritt 8: Gold-Labels konvertieren
       ↓  Schritt 2 (neu) + Schritt 3 (neu)
FVG-Tokenklassifikator v3 + syntaktische Nachfilterung (spaCy)
       ↓
Gradio-Applikation (HF Spaces)
```

### 3.2 Annotationsschema

Die Annotation folgt dem BIO-Schema mit vier Klassen:

| Label | Bedeutung | Beispiel |
|---|---|---|
| `B-VERB` | Beginn des Funktionsverbs | *erhebt* |
| `B-NOM` | Beginn des Nominalteils | *Beschwerde*, *in*, *zum* |
| `I-NOM` | Fortsetzung des Nominalteils | *Erwägung* (nach *in*), *Schluss* (nach *zum*) |
| `O` | Ausserhalb eines FVG | alle anderen Token |

Dieses Schema erlaubt die Repräsentation diskontinuierlicher FVG:

```
Er    nimmt   die   Einwände   in    Betracht   .
O     B-VERB  O     O          B-NOM I-NOM      O
```

### 3.3 Zweipass-Annotation (Distant Supervision)

Da keine manuell annotierten Trainingsdaten vorlagen, wurde **Distant Supervision** eingesetzt: Die FVG-Referenzliste dient als Schwach-Annotation des Korpus.

#### Pass 1 – Listenbasiert (hohe Präzision)

Für jedes der 209 FVG-Muster wird nach Kookkurrenz von Funktionsverb-Lemma und Nominal-Lemma(ta) in einem Fenster von ±20 Token gesucht. Lemmatisierung mit spaCy ermöglicht die robuste Erkennung flektierter Formen (*erhebt*, *erhob*, *erhoben* → Lemma *erheben*).

#### Pass 2 – Syntaxbasiert (Generalisierung)

Für eine Liste von Funktionsverben werden syntaktische Abhängigkeiten (spaCy-Dependency-Parser) ausgewertet:

- **Direktes Objekt** (`dep=oa`): Wenn ein Funktionsverb ein abstraktes Nomen als direktes Objekt hat, wird es als FVG annotiert.
- **Präpositionalphrase** (`dep=mo`, `dep=cvc`): Wenn ein Funktionsverb eine PP mit FVG-typischer Präposition (*in*, *zu*, *auf*, *an* etc.) regiert, wird die Konstruktion annotiert.

`haben` und `sein` wurden bewusst ausgeschlossen, da sie als Auxiliarverben Subjekte fälschlich als Nominalteil annotieren würden.

Nach der manuellen Evaluation (vgl. Abschnitt 4.3) wurde die `FUNKTIONSVERBEN`-Liste von **28 auf 35 Verben** erweitert, um die häufigsten False Negatives zu beheben:

| Neu hinzugefügt | Begründung |
|---|---|
| `vornehmen` | 41 False Negatives (häufigster Fehler) |
| `erteilen` | 35 False Negatives |
| `aussprechen` | 10 False Negatives |
| `belegen`, `schliessen` | 6 bzw. 6 False Negatives |
| `anordnen`, `einreichen` | Weitere häufige juristische FV |

#### Ausgleich der Klassen

Negative Beispiele werden im Verhältnis 2:1 gegenüber positiven Beispielen zufällig ausgewählt.

### 3.4 Modell-Finetuning

Als Basismodell wurde **deepset/gbert-base** gewählt, ein auf deutschsprachigen Texten vortrainiertes BERT-Modell.

**Trainingsparameter:**

| Parameter | Wert |
|---|---|
| Basismodell | `deepset/gbert-base` |
| Aufgabe | Token Classification (NER-Stil) |
| Epochen | 4 |
| Batch-Grösse | 32 |
| Lernrate | 2×10⁻⁵ |
| Weight Decay | 0.01 |
| Warmup-Anteil | 10% |
| FP16 | Ja (CUDA) |
| Best-Model-Kriterium | F1-Score (Dev-Set) |

**Subword-Alignment**: Folge-Subwörter von `B-NOM`/`I-NOM` erhalten das Label `I-NOM`; Folge-Subwörter von `B-VERB` erhalten `-100` (werden im Loss ignoriert).

### 3.5 Manuelle Evaluation und Gold-Labels

#### Evaluationssatz-Extraktion

Aus den drei Referenzlisten (liste_fvg.txt, verb_noun_fvg.csv, praep_noun_verb.csv) wurden Verb-Nomen-Paare extrahiert und dazu jeweils bis zu 3 Sätze aus dem Korpus gesucht, die beide Elemente enthalten. Dies ergab **1'149 Sätze**:

- 810 Sätze für einfache Verb+Nomen-Paare (aus verb_noun_fvg.csv)
- 339 Sätze für präpositionale FVG-Muster (aus praep_noun_verb.csv)

#### Annotationsprotokoll

Die Annotation erfolgte über eine eigens entwickelte Gradio-GUI (offline). Pro Satz wurde beurteilt, ob das hervorgehobene Verb-Nomen-Paar im gegebenen Kontext ein FVG bildet (1 / 0 / ?). Entscheidend war dabei der **Satzkontext**, nicht die abstrakte Kombinierbarkeit: Ein Satz, in dem das Verb und das Nomen syntaktisch nicht zusammengehören (z.B. *macht … geltend … Angabe*, wo *macht geltend* das eigentliche FVG ist), wurde als 0 annotiert.

#### Gold-Label-Integration in das Training

Von den 386 FVG=1-Sätzen wurden **299 (80%)** in BIO-Format konvertiert und dem Trainingsdatensatz hinzugefügt (10-fach oversampelt zur Gewichtung). Die verbleibenden **75 Sätze (20%)** wurden als held-out Testset zurückgehalten.

### 3.6 Syntaktische Nachfilterung

Als zusätzliche Massnahme zur Präzisionssteigerung bei der Inferenz wurde eine **syntaktische Nachfilterung** implementiert (`fvg_syntax_filter.py`). Sie prüft nach der BERT-Vorhersage mittels spaCy's Dependency-Parser, ob zwischen dem vorhergesagten VERB-Token und dem NOM-Token tatsächlich eine FVG-typische Dependenzrelation besteht:

| Relation | Bedingung |
|---|---|
| Direktes Objekt | `NOM.dep_ ∈ {oa, obj, dobj}` und `NOM.head == VERB` |
| Präpositional | `NOM.dep_ == nk` und `NOM.head.dep_ ∈ {mo, cvc, op, pg, obl}` und `NOM.head.head == VERB` |

Vorhersagen, bei denen keine dieser Relationen gefunden wird, werden verworfen. Die Filterung ist sowohl in der App als auch in der Evaluation aktiv.

---

## 4. Ergebnisse

### 4.1 Trainingsdaten (finale Version, Modell v3)

| Kategorie | Anzahl Sätze |
|---|---|
| Positive Sätze distant (mit FVG) | 37'841 |
| Negative Sätze distant (ohne FVG) | 75'993 |
| Gold-Sätze (299 × 10-fach oversampled) | 2'990 |
| **Total Training** | **116'824** |
| Dev | 14'229 |
| Test (distant) | 14'230 |

### 4.2 Dev-Set-Leistung (internes Evaluationsset)

| Modell | F1 (Dev) | NOM F1 | VERB F1 |
|---|---|---|---|
| v1 (5'724 Dok., Pass 1+2) | 0.96 | 0.95 | 0.98 |
| v3 (erweitertes FUNKTIONSVERBEN + Gold) | **0.954** | 0.92 | 0.97 |

*Hinweis:* Diese Werte sind zirkulär, da das Dev-Set ebenfalls durch Distant Supervision annotiert wurde (vgl. Diskussion Abschnitt 5.2).

### 4.3 Manuelle Evaluation (Gold-Standard)

#### Distant Supervision vs. Gold

| Metrik | Wert |
|---|---|
| Precision (DS vs. Gold) | 0.40 |
| Recall (DS vs. Gold) | 0.44 |
| **F1 (DS vs. Gold)** | **0.42** |

Die Distant-Supervision-Annotation stimmt nur zu 42% mit den manuellen Urteilen überein. Dies spiegelt das bekannte Problem der Distant Supervision wider: Die automatischen Labels enthalten substanzielles Rauschen.

#### Modell-Evaluation gegen Gold (Übersicht aller Modellversionen)

| Version | Evaluationsset | n | Precision | Recall | F1 | Bemerkung |
|---|---|---|---|---|---|---|
| v1 | Alle Gold-Paare | 1'145 | 0.546 | 0.510 | **0.527** | Ohne Gold-Training |
| v2 | Alle Gold-Paare | 1'145 | 0.691 | 1.000 | **0.817** | Datenleck: Gold im Train |
| v3 (ohne Filter) | Held-out (20%) | 822 | 0.218 | 0.595 | **0.319** | Unverzerrt |
| **v3 (mit spaCy-Filter)** | **Held-out (20%)** | **822** | **0.249** | **0.582** | **0.348** | **Unverzerrt, empfohlen** |

#### Aufschlüsselung v3 mit syntaktischer Filterung (held-out)

| Teilmenge | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Alle Paare | 822 | 0.249 | 0.582 | **0.348** |
| Einfache Verb+Nomen | 629 | 0.163 | 0.581 | **0.255** |
| Präpositions-FVG | 193 | 0.656 | 0.583 | **0.618** |

**Einfluss der syntaktischen Filterung:**

| Metrik | ohne Filter | mit Filter | Δ |
|---|---|---|---|
| Precision | 0.218 | 0.249 | +14% |
| Recall | 0.595 | 0.582 | −2% |
| F1 | 0.319 | **0.348** | +9% |
| False Positives | 169 | 139 | −18% |

Die syntaktische Filterung eliminiert 30 falsche Vorhersagen (−18%), ohne den Recall wesentlich zu verschlechtern.

#### Fehleranalyse: häufigste False Negatives (held-out, v3)

| Verb | FN | Ursache |
|---|---|---|
| `vornehmen` | 6 | Trotz Erweiterung: Dependenz-Mismatch oder unbekannte Nomen |
| `erteilen` | 5 | Idem |
| `fallen` | 3 | Nicht in FUNKTIONSVERBEN (z.B. *ausser Betracht fallen*) |
| `lassen` | 3 | Ambig: Vollverb vs. Funktionsverb |

#### Fehleranalyse: häufigste False Positives (held-out, v3 mit Filter)

| Paar | FP | Ursache |
|---|---|---|
| `geben + Möglichkeit` | 3 | Syntaktisch abhängig, aber kein FVG im Kontext |
| `leisten + Sicherheit` | 3 | „Sicherheit leisten" (Kaution) = fragl. FVG-Status |
| `führen + Lebensgemeinschaft` | 3 | Vollverb-Lesart |
| `stellen + Ungleichbehandlung` | 3 | Verb und Nomen syntaktisch verbunden, aber kein FVG |

### 4.4 Qualitative Testresultate

| Satz | VERB | NOM | Korrekt? |
|---|---|---|---|
| *Das Bundesgericht **erhebt** **Beschwerde** gegen diesen Entscheid.* | erhebt | Beschwerde | ✅ |
| *Die Partei **stellt** einen **Antrag** auf Aufhebung.* | stellt | Antrag | ✅ |
| *Der Kanton **zieht** die Massnahmen **in Erwägung**.* | zieht | in Erwägung | ✅ |
| *Die Vorinstanz hat der Partei kein rechtliches **Gehör** **gewährt**.* | gewährt | Gehör | ✅ (generalisiert) |
| *Die Massnahme **kommt zur Anwendung**, sobald sie **in Kraft tritt**.* | kommt/tritt | zur Anwendung / in Kraft | ✅ (diskontinuierlich) |
| *Er **nimmt** die **Vorkehrungen** vor.* | nimmt vor | Vorkehrungen | ✅ (neu: vornehmen) |
| *Das Amt **erteilt** eine **Bewilligung**.* | erteilt | Bewilligung | ✅ (neu: erteilen) |
| *macht … nicht geltend … **Angabe** … getan* | – | – | ✅ (korrekt abgelehnt) |

### 4.5 Entwicklung über alle Trainingsläufe

| Lauf | Datenbasis | Änderung | F1 (Dev) | F1 (Gold, extern) |
|---|---|---|---|---|
| 1 | 824 Dok., Pass 1 | Baseline | 0.96 | – |
| 2 | 5'724 Dok., Pass 1 | Mehr Daten | 0.99 | – |
| 3 | 5'724 Dok., Pass 1+2 | Pass 2 (Bug: `haben` als FV) | 0.95 | – |
| 4 | 5'724 Dok., Pass 1+2 | FVG-Liste nicht geladen¹ | 0.96 | – |
| 5 (v1) | 5'724 Dok., Pass 1+2 | Alle Bugs behoben | 0.96 | 0.527 |
| 6 (v2) | + Gold-Labels (alle) | Gold im Train (Datenleck) | 0.952 | 0.817² |
| **7 (v3)** | **+ erw. FUNKTIONSVERBEN + Gold (80%)** | **80/20-Split, spaCy-Filter** | **0.954** | **0.348³** |

¹ Kritischer Bug: Datei ohne Tabulatoren, Parser übersprang alle 209 Muster.  
² Inflationiert durch Datenleck: Gold-Labels im Training und in der Evaluation.  
³ Unverzerrt: held-out 20% Gold + alle FVG=0-Sätze (n=822).

---

## 4b. Zweite Projektphase: Pair-Klassifikator

### 4b.1 Motivation

Die externe Evaluation des FVG-Taggers (F1=0.348) zeigte, dass das Hauptproblem nicht die Kandidatenfindung, sondern die semantische Unterscheidung ist: Verb und Nomen stehen zwar syntaktisch in Beziehung, bilden aber kontextuell kein FVG. Eine syntaktische Nachfilterung allein kann dieses Problem nicht lösen — 109 von 125 verbleibenden False Positives hatten eine korrekte `oa`-Dependenzrelation. Das Modell weiss, *wo* Verb und Nomen stehen, aber nicht *ob* sie in diesem Kontext ein FVG bilden.

Als Lösung wurde ein **Pair-Klassifikator** entwickelt: ein Sequenzklassifikationsmodell, das für ein gegebenes (Satz, Verb, Nomen)-Tripel direkt entscheidet, ob ein FVG vorliegt.

### 4b.2 Datenvorbereitung: Gold-Paare

Aus dem manuell annotierten Datensatz (eval_sentences.csv, 1'149 Sätze) wurden Verb-Nomen-Paare extrahiert:

| Split | Positiv | Negativ | Total |
|-------|:-------:|:-------:|:-----:|
| Train (80% der FVG=1-Sätze + alle FVG=0) | 325 | 607 | 932 |
| Test (20% der FVG=1-Sätze) | 92 | 152 | 244 |

**Input-Format:** `[CLS] Satz [SEP] Verb Muster [SEP]`

Beispiel: `[CLS] Der Beschwerdeführer hat Beschwerde erhoben. [SEP] erhoben Beschwerde [SEP]`

Bei präpositionalen FVG wird die Präposition dem Muster vorangestellt: `erhoben in Erwägung`.

### 4b.3 Pair-Klassifikator v1 (Gold-Daten)

**Architektur:** `deepset/gbert-base` + Sequenzklassifikationskopf (Binary: FVG / KEIN_FVG)

**Trainingsparameter:** 10 Epochen, lr=2×10⁻⁵, batch=16, Klassengewichte (neg=0.349, pos=0.651)

**Ergebnis (bereinigtes Gold-Testset, n=244):**

| Precision | Recall | F1 |
|:---------:|:------:|:--:|
| 0.710 | 0.815 | 0.759 |

Dies entspricht einer Verdoppelung des F1 gegenüber dem FVG-Tagger (0.348 → 0.759). Der Sprung erklärt sich durch die präzisere Aufgabenformulierung: Das Modell entscheidet nur für ein gegebenes Paar, nicht über alle Tokens eines Satzes gleichzeitig.

### 4b.4 Claude als Annotator (DS-Augmentierung)

Um die 37'841 Distant-Supervision-Paare aus `annotated_train.jsonl` für das Training nutzen zu können, wurden sie mit **Claude Haiku 4.5** automatisch annotiert.

**Prompt-Strategie:** Der System-Prompt enthielt die linguistischen FVG-Kriterien (Simplex-Test, Selektionstest), klare Positiv- und Negativbeispiele sowie eine Liste typischer Funktionsverben. Das Modell wurde angewiesen, ausschliesslich mit "ja" oder "nein" zu antworten.

**Qualitätsevaluation auf Gold-Testset (n=244):**

| Precision | Recall | F1 |
|:---------:|:------:|:--:|
| 0.900 | 0.500 | 0.643 |

Claudes hohe Precision (0.90) und niedrige Recall (0.50) spiegeln eine konservative Annotationsstrategie wider: Das Modell lehnt Grenzfälle ab. Ein Teil der scheinbaren False Negatives (ca. 14 von 45) waren tatsächlich Fehler im Gold-Testset, die anschliessend bereinigt wurden.

**Annotation der DS-Paare:**

| Klasse | Anzahl | Anteil |
|--------|:------:|:------:|
| FVG (ja) | 9'711 | 25.7% |
| kein FVG (nein) | 28'041 | 74.1% |
| unbekannt (?) | 89 | 0.2% |

**Kosten:** ~$45 (Claude Haiku 4.5, kein Prompt-Caching aktiv — das Modell unterstützt diese Funktion offenbar nicht). Für zukünftige Läufe empfiehlt sich `claude-haiku-3-5`, das Prompt-Caching unterstützt und die Kosten auf ~$4–5 reduzieren würde.

### 4b.5 Pair-Klassifikator v2 (Gold + DS-Daten)

**Gewichtungsschema:**

| Datenquelle | Gewicht |
|-------------|:-------:|
| Gold positiv | 1.0 |
| Gold negativ | 1.0 |
| DS positiv (Claude=ja) | 0.5 |
| DS negativ (Claude=nein) | 0.3 |

DS-Negative wurden auf DS_NEG_RATIO=2.0 × DS-Positive begrenzt (19'422 Sätze), um Klassenimbalanz zu vermeiden.

**Trainingsdaten gesamt:** 30'065 Paare (10'036 positiv / 20'029 negativ)

**Ergebnis (bereinigtes Gold-Testset, n=244):**

| Precision | Recall | F1 |
|:---------:|:------:|:--:|
| 0.803 | 0.753 | **0.777** |

v2 übertrifft v1 auf dem bereinigten Testset (F1: 0.759 → 0.777). Der Mechanismus: Claudes konservative Annotation entspricht den bereinigten Gold-Kriterien (Simplex-Test, Selektionstest). Fälle, die Claude als nicht-FVG klassifiziert — `Bedingung stellen`, `Eingabe einreichen`, `auf Abweisung schliessen` — wurden im Gold-Testset ebenfalls als Fehlannotationen identifiziert und korrigiert.

### 4b.6 Bereinigung des Gold-Testsets

Der Vergleich zwischen Claude-Annotationen und Gold-Labels führte zur Identifikation von **19 Fehlannotationen** (15 × 1→0, 4 × 0→1):

**1→0 (fälschlich als FVG annotiert):**
`auf Abweisung schliessen/vernehmen` (×8), `von Amtes wegen` (×2), `Strafuntersuchung führen`, `Bedingung stellen` (×2), `Eingabe einreichen`, `auf Antrag verfolgen`

**0→1 (fälschlich als kein FVG annotiert):**
`zu Grunde legen` (Duplikat), `Würdigung vornehmen`, `Konzession erteilen`, `Auslegung geben`

Das bereinigte Testset enthält 81 positive und 163 negative Paare (original: 92/152).

### 4b.7 Vergleich aller Modelle

| Modell | Precision | Recall | F1 | Anmerkung |
|--------|:---------:|:------:|:--:|-----------|
| FVG-Tagger (DS) | 0.566 | 0.531 | 0.548 | Evaluation: Tokenvorhersage auf Pair-Testset |
| FVG-Tagger + spaCy-Filter | — | — | 0.348 | Evaluation auf ursprünglichem Gold-Held-out |
| Pair-Klassifikator v1 | 0.710 | 0.815 | 0.759 | Gold-Paare, bereinigtes Testset |
| **Pair-Klassifikator v2** | **0.803** | **0.753** | **0.777** | **Gold + DS, bereinigtes Testset** |

### 4b.8 Demo-Applikation

Die finale Applikation (`app/demo.py`) vergleicht zwei Ansätze interaktiv:

**Modell 1 — FVG-Tagger:** Token-Klassifikator auf gesamtem Satz. Stärke: findet auch Konjunktiv-Formen (`übten`, `böten`) und koordinierte Strukturen, die spaCy nicht korrekt parst. Schwäche: geringe Precision.

**Modell 2 — spaCy + Pair-Klassifikator v2:** spaCy extrahiert syntaktische Kandidaten (direkte Objekte, PP-Objekte mit FVG-Präpositionen), v2 entscheidet für jedes Paar. Stärke: hohe Precision. Schwäche: scheitert bei spaCy-Parsing-Fehlern (Koordination, Einbettung) und unbekannten Funktionsverben.

Die beiden Ansätze sind komplementär: Fälle, die Modell 1 allein findet (seltene Konjunktiv-Formen), verfehlt Modell 2; Fälle, die Modell 2 allein findet (präpositionaler Typ mit seltenen Strukturen), verfehlt Modell 1.

---

## 5. Diskussion

### 5.1 Stärken des Ansatzes

**Keine manuelle Annotation für das Training nötig**: Der Distant-Supervision-Ansatz ermöglicht die automatische Generierung grosser Trainingsdatensätze aus einer verhältnismässig kleinen Referenzliste.

**Robustheit gegenüber Flexion**: Durch spaCy-Lemmatisierung werden alle Flexionsformen erfasst – einschliesslich seltener Konjunktivformen wie *böten* (Konjunktiv II von *bieten*).

**Diskontinuität**: Die BERT-Architektur und das fensterbasierte Matching erlauben die Erkennung diskontinuierlicher FVG.

**Generalisierung durch Pass 2**: Der syntaxbasierte Annotationspass erkennt FVG, die nicht in der Referenzliste stehen.

**Iterative Verbesserung durch manuelles Feedback**: Die manuelle Evaluation deckte die grössten Schwachstellen auf (`vornehmen`, `erteilen`), die durch gezielte Erweiterung der FUNKTIONSVERBEN-Liste und Gold-Label-Integration behoben wurden.

### 5.2 Grenzen und Fehlerquellen

**Evaluationszirkularität**: Der intern gemessene F1=0.96 misst primär die Konsistenz des Modells mit der automatischen Annotation, nicht seine linguistische Korrektheit. Die externe Evaluation gegen manuelle Gold-Labels ergibt F1=0.348 (held-out, mit syntaktischem Filter) – ein erheblicher Unterschied, der das Ausmass des Distant-Supervision-Rauschens illustriert.

**Präzisionsproblem bei einfachen Verb+Nomen-Paaren (F1=0.255)**: Die grösste Schwachstelle ist die zu niedrige Precision (0.163). Das Modell erkennt Sätze, in denen ein Funktionsverb und ein FVG-Nomen syntaktisch miteinander verbunden sind, nicht immer korrekt – auch wenn sie kein FVG bilden. Beispiel: *eine Lebensgemeinschaft führen* (Vollverb) vs. *Verhandlungen führen* (FVG). Die syntaktische Nachfilterung reduziert zwar False Positives (−18%), kann aber das grundlegende semantische Unterscheidungsproblem nicht lösen.

**Syntaktische Einschränkungen der Nachfilterung**: spaCy's Dependency-Parser analysiert nicht alle Satzstrukturen korrekt, insbesondere bei komplexen juristischen Satzperioden mit mehrfacher Einbettung. Falsch analysierte Strukturen können zu fälschlichem Verwerfen echter FVG führen.

**Listenabhängigkeit**: Trotz Pass 2 bleibt das Modell für Verben ausserhalb der FUNKTIONSVERBEN-Liste und Nomina ausserhalb des FVG-Pools stark eingeschränkt. Verben wie `fallen` (*ausser Betracht fallen*), `lassen` und `bleiben` (*ausser Betracht bleiben*) sind noch nicht abgedeckt.

**Geringe Grösse des held-out Testsets**: Mit 75 positiven Sätzen im held-out Testset sind die Evaluationsmetriken statistisch unsicher. Breitere Konfidenzintervalle wären methodisch angemessen.

### 5.3 Mögliche Erweiterungen

- **Erweiterung FUNKTIONSVERBEN**: `fallen`, `bleiben`, `lassen` mit geeigneten Nomenpools aufnehmen.
- **Semantische Nachfilterung**: Anstelle (oder zusätzlich) zur syntaktischen Filterung einen semantischen Ähnlichkeitsfilter einsatz, der prüft, ob das Nomen eine Nominalisierungsparaphrase zulässt.
- **Grösseres Basismodell**: `deepset/gbert-large` würde voraussichtlich bessere Kontextualisierung und damit höhere Precision liefern.
- **Vollständiges Korpus**: Training auf allen 40'930 BGE-Entscheiden.
- **Erweiterter Gold-Datensatz**: 200–500 weitere manuell annotierte Sätze würden die Evaluation statistisch belastbarer machen.
- **Mehrsprachigkeit**: Adaption für französische und italienische Rechtstexte.

---

## 6. Schluss

Das Projekt zeigt, dass die automatische Erkennung von Funktionsverbgefügen in deutschen Rechtstexten mit Distant Supervision und BERT-Finetuning möglich ist — und dass die Wahl der Modellarchitektur entscheidend ist.

Der FVG-Tagger (Token-Klassifikation) erreicht auf dem bereinigten Pair-Testset F1=0.548. Der intern gemessene F1=0.96 auf dem automatisch annotierten Dev-Set ist stark inflationiert und illustriert eine methodisch wichtige Warnung: Distant Supervision erzeugt grosse Trainingsmengen, aber auch substanzielles Rauschen, das sich im internen Evaluationsergebnis nicht widerspiegelt.

Der entscheidende Fortschritt ergab sich durch die Umformulierung der Aufgabe: Statt aller FVG-Tokens in einem Satz zu suchen, entscheidet der Pair-Klassifikator für ein gegebenes (Satz, Verb, Nomen)-Tripel. Dies führte zu einer Verdoppelung des F1 (0.348 → 0.777). Die Augmentierung mit 29'133 Claude-annotierten DS-Paaren verbesserte den Pair-Klassifikator weiter (v1: F1=0.759 → v2: F1=0.777), da Claudes konservative Annotationsstrategie gut mit den bereinigten linguistischen Kriterien übereinstimmt.

Ein weiterer methodischer Befund: Der Vergleich mit Claude-Annotationen deckte 19 Fehlannotationen im Gold-Testset auf. Annotation ist ein iterativer Prozess — auch manuell erstellte Gold-Labels enthalten Fehler, insbesondere bei Grenzfällen wie `auf Abweisung schliessen` (fester Rechtsausdruck) oder `Bedingung stellen` (kein Simplex möglich).

Die zwei komplementären Ansätze in der Demo-Applikation illustrieren einen grundlegenden Befund: Es gibt keine universelle Lösung für die FVG-Erkennung. Der FVG-Tagger findet Konjunktiv-Formen und koordinierte Strukturen; der Pair-Klassifikator ist präziser bei Standardfällen. Für eine produktive Anwendung empfiehlt sich der Pair-Klassifikator v2 als primäres Modell, ergänzt durch den FVG-Tagger für die Kandidatengenerierung in syntaktisch komplexen Kontexten.

---

## Anhang: Technische Details

### Verwendete Ressourcen

| Komponente | Version / Spezifikation |
|---|---|
| Python | 3.11.9 |
| spaCy | 3.8.11 (`de_core_news_lg`) |
| Transformers | 4.57.6 |
| PyTorch | 2.7.1+cu118 |
| GPU (Training) | NVIDIA GeForce RTX 4070 |
| Trainingsdauer | ~21 Min. (v3, finaler Lauf) |
| Annotationsdauer | ~40 Min. (5'724 Dokumente, v3) |
| Manuelle Annotation | 1'149 Sätze |

### Projektstruktur

```
fvg_project/
├── data/
│   ├── raw_texts.jsonl            Extrahierte Rohtexte
│   ├── annotated_train.jsonl      113'834 Sätze (distant, v3)
│   ├── annotated_dev.jsonl        14'229 Sätze
│   ├── annotated_test.jsonl       14'230 Sätze
│   ├── eval_sentences.csv         1'149 manuell annotierte Sätze
│   ├── eval_model_results.csv     Modellvorhersagen + Gold-Labels
│   ├── gold_train.jsonl           299 Gold-Sätze (80%-Split, FVG-Tagger)
│   ├── gold_test.jsonl            75 Gold-Sätze (20%-Split, FVG-Tagger)
│   ├── pair_train.jsonl           932 Paare (Pair-Klassifikator, Gold)
│   ├── pair_test.jsonl            244 Paare (bereinigt, 81 pos / 163 neg)
│   ├── pair_test_orig.jsonl       244 Paare (Original vor Bereinigung)
│   ├── ds_claude_labels.jsonl     37'841 DS-Paare mit Claude-Labels
│   └── claude_test_results_v3.csv Claude-Evaluation auf Gold-Testset
├── models/
│   ├── fvg-gbert-base/            FVG-Tagger (Token-Klassifikator)
│   ├── fvg-pair-classifier/       Pair-Klassifikator v1 (Gold only)
│   └── fvg-pair-classifier-v2/    Pair-Klassifikator v2 (Gold + DS)
├── scripts/
│   ├── 01_extract_texts.py        HTML/TXT → JSONL
│   ├── 02_annotate.py             Zweipass-Annotation (Distant Supervision)
│   ├── 03_train.py                FVG-Tagger trainieren
│   ├── 04_push_to_hub.py          Modell → HF Hub
│   ├── 05_extract_eval_sentences.py   Evaluationssätze (Verb+Nomen)
│   ├── 05b_extract_praep_eval.py  Evaluationssätze (Präp-FVG)
│   ├── 06_annotate_gui.py         Offline-GUI für manuelle Annotation
│   ├── 07_eval_model.py           FVG-Tagger evaluieren
│   ├── 08_prepare_gold_train.py   Gold-Labels → BIO-Format + 80/20-Split
│   ├── 09_push_space.py           App-Dateien → HF Spaces
│   ├── 10_test_filter_v2.py       Syntaktischen Filter v2 testen
│   ├── 11_prepare_pair_data.py    Pair-Daten aus Gold-Annotation erstellen
│   ├── 12_train_pair_classifier.py    Pair-Klassifikator v1 trainieren
│   ├── 13_claude_annotator_test.py    Claude als Annotator evaluieren
│   ├── 14_claude_annotate_ds.py   DS-Paare mit Claude annotieren
│   └── 15_train_pair_classifier_v2.py Pair-Klassifikator v2 trainieren
└── app/
    ├── demo.py                    Gradio-Vergleichsapp (2 Modelle)
    ├── app.py                     Ursprüngliche HF-Spaces-App
    └── fvg_syntax_filter.py       Syntaktische Nachfilterung (spaCy)
```

### Nachnutzung

Das trainierte Modell ist unter `boj-per/fvg-gbert-base` auf dem Hugging Face Hub öffentlich abrufbar:

```python
from transformers import pipeline
import spacy
from fvg_syntax_filter import syntactic_filter

pipe = pipeline("token-classification",
                model="boj-per/fvg-gbert-base",
                aggregation_strategy="first")
nlp = spacy.load("de_core_news_lg", disable=["ner"])

text = "Das Gericht zieht die Massnahmen in Erwägung."
entities = pipe(text)
entities = syntactic_filter(text, entities, nlp)  # optionaler Präzisionsfilter
# → [{'entity_group': 'VERB', 'word': 'zieht', ...},
#    {'entity_group': 'NOM',  'word': 'in Erwägung', ...}]
```
