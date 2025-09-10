
# NeuralNetworkPipeline

Dieses Projekt dient zur vertiefenden Anwendung der Inhalte aus den abgeschlossenen Kaggle-Kursen, inbesondere den Kursen 
**Intermediate Machine Learning**, **Deep Learning** und **Feature Engineering**.
Aus diesem Grund wird hier ein künstliches neuronales Netzwerk als ML-Modell gewählt und dafür
eine Bibliothek zur Datenvorverarbeitung in Pipeline-Struktur entwickelt. 

Weitere Details zur Entwicklung, den Hintergründen und den getroffenen Entscheidungen 
finden sich in der begleitenden [Dokumentation](Projektdokumentation.pdf).

---

## Installation

1. Python 3.10+ empfohlen  
2. Virtuelle Umgebung erstellen (optional, aber empfohlen):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Benötigte Pakete installieren

```bash
pip install -r requirements.txt
```

---

## Datensätze

Die im Projekt verwendeten Kaggle-Datensaetze sind aus Lizenzgründen nicht im Repository gespeichert.
Die benötigten Datensätze **train.csv** und **test.csv** müssen mit einem eigenen Kaggle-Profil von 
[Kaggle heruntergeladen werden](https://www.kaggle.com/c/titanic/data?select) geladen werden 
(scrollen bis der Data Explorer, nach dem Ende der Dataset Description, auf der rechten Seite erscheint).
Diese beiden Datensätze in den Unterordner **Datensaetze** des Projektordners in dieser Art ablegen:

```kotlin
Datensaetze/
├── train.csv      
├── test.csv       
└── .gitkeep       
```

---

## Ausführung

Im Projektordner sind zwei Ausführungsskripte enthalten, welche jeweils eine unterschiedliche 
Datenvorverarbeitung ausführen und dafür auf jeweils unterschiedliche Versionen der Datenvorverarbeitungs-Bibliothek 
zugreifen. Diese und alle anderen eigens entwickelten Bibliotheken befinden sich in dem Unterordner **Bibliotheken**.

Beide Ausführungsskripte geben einen Plot mit dem Trainingsverlauf, Trainingsparameter und Modellarchitektur aus.
Das Training mit einer älteren Version der Datenvorverarbeitungs-Bibliothek lässt sich mit 

```bash
python Training_with_DataPreprocessingFunctions_old.py
```

ausführen. Hierbei ist die Datenvorverarbeitung in einer völlig automatisierten Pipeline ausgeführt.

Das Training mit der neuesten Version der Datenvorverarbeitungs-Bibliothek lässt sich mit

```bash
dotenv run -- python Training_with_DataPreprocessingFunctions.py
````

ausführen. Diese Bibliothek geht aus der Optimierung der älteren Version bezüglich Erweiterbarkeit, Übersichtlichkeit 
sowie der Realisierung von Feature Engineering Prozessen, zwischen den Datenvorverarbeitungsprozessen, hervor. Hierbei wird für das Feature Engineering die Mutual Information zwischen den Features und dem Target
berechnet, wofür in diesem Skript die Datenvorverarbeitungsprozesse einzeln ausgeführt sind. Für die berechnete 
Mutual Information wird ein zusätzlicher Plot ausgegeben.

---

## Reproduzierbarkeit

Für die Reproduzierbarkeit der Trainingsergebnisse sind die benötigten Environment-Variablen in der Datei **.env** hinterlegt, auf die beide Ausführungsskripte zugreifen.  
Zusätzlich enthalten beide Skripte mehrere Codezeilen zur Sicherstellung der Reproduzierbarkeit.

---

## Hinweis

In den Versionen der Datenvorverarbeitungs-Bibliotheken erfolgen die Vorverarbeitungsprozesse auf einem 
zusammengefassten Datensatz aus Trainings-, Validierungs- und Testdaten, wodurch Data Leakage entsteht, welcher die 
Vorhersageergebnisse (Validierungsfehler) verzerren kann.
In künftigen Versionen ist daher Data Leakage zu entfernen.

---

## Lizenz
Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).