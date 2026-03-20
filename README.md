# NeuralNetworkPipeline
Dieses Projekt realisiert eine generalisierte Pipeline für neuronale Netzwerke auf Basis des Frameworks Keras (TensorFlow).
Dabei soll mit dieser Pipeline das Training eines neuronalen Netzwerkes so kompakt und funktionell wie bei der 
Pipeline-Funktion für klassische Machine-Learning-Modelle aus der Bibliothek scikit-learn durchgeführt werden.
Die Pipeline übernimmt die Vorverarbeitung strukturierter Daten und trainiert damit das ausgewählte neuronale Netzwerk. 
Vorhersagen mit der Pipeline können dabei autark von den Trainingsdaten durchgeführt werden.
Die Pipeline ist objektorientiert in der Programmiersprache Python implementiert.


Weitere Details finden sich in der begleitenden [Dokumentation](./Projektdokumentation.pdf).

---

## Installation

1. Python 3.10.x empfohlen (getestet 3.10.9)
2. Repository von GitHub klonen:

```bash
git clone https://github.com/MDutsch/NeuralNetworkPipeline.git
cd NeuralNetworkPipeline
```

3. Virtuelle Umgebung erstellen:

```bash
python3.10 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\Activate.ps1  # Windows PowerShell
venv\Scripts\activate.bat  # Windows cmd.exe
```

4. Benötigte Pakete installieren:

```bash
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt
```

---

## Datensätze

Die im Projekt verwendeten Kaggle-Datensaetze sind aus Lizenzgründen nicht im Repository gespeichert.
Die benötigten Datensätze **train.csv** und **test.csv** müssen mit einem eigenen Kaggle-Profil von 
[Kaggle heruntergeladen werden](https://www.kaggle.com/c/titanic/data?select) geladen werden 
(scrollen bis der Data Explorer nach dem Ende der Dataset Description auf der rechten Seite erscheint).
Diese sind in den Unterordner **Datensaetze** des Projektordners wie folgt abzulegen:

```kotlin
Datensaetze/
├── train.csv      
├── test.csv       
└── .gitkeep       
```

---

## Ausführung
Zur Demonstration der Pipeline sind die zwei hier aufgelisteten Skripte nacheinander auszuführen.
#### 1. Training
Das erste Skript führt das Training über die Pipeline durch. Dabei wird das zu trainierende neuronale Netzwerk definiert 
und mit dem Datensatz **train.csv** trainiert. Abschließend speichert das Skript die Pipeline, welche sowohl das 
trainierte neuronale Netzwerk als auch die gelernten Transformation States enthält,in eine Joblib-Datei mit 
dem Namen **Pipeline.joblib**.
Das Skript lässt sich mit dem Befehl
```bash
python3.10 Training_with_Pipeline.py
```
ausführen. 
#### 2. Vorhersage
Die Vorhersage mit den Testdaten **test.csv** erfolgt im zweiten Skript. Dieses lädt die Pipeline aus der 
Datei **Pipeline.joblib** und führt damit die Vorhersage durch. 
Das Skript lässt sich mit dem Befehl
```bash
python3.10 Prediction_with_Pipeline.py
```
ausführen.

---
## Reproduzierbarkeit

Für die Reproduzierbarkeit der Ergebnisse sind die benötigten Umgebungsvariablen in der Datei **.env** 
hinterlegt, auf die beide Ausführungsskripte zugreifen. Zusätzlich enthalten beide Skripte Konfigurationen zur 
Sicherstellung der Reproduzierbarkeit.

---

## Lizenz
Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).