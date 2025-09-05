# NeuralNetworkPipeline

In diesem Projekt wird eine Datenvorverarbeitung für ein neuronales Netzwerk entwickelt, das 
auf dem TensorFlow (Keras) Framework basiert. Für das Training dieses Netzwerks wird der 
Titanic-Kaggle-Datensatz verwendet, der vorverarbeitet wird, um die Daten für das Modell 
optimal vorzubereiten.

Weitere Details zur Entwicklung, den Hintergründen und den getroffenen Entscheidungen 
finden sich in der begleitenden [Dokumentation](Projektdokumentation.pdf).  

---

## Installation

1. Python 3.10+ empfohlen  
2. Virtuelle Umgebung erstellen (optional, aber empfohlen):
````bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
````
3. Benötigte Pakete installieren
````bash
pip install -r requirements.txt
````

---

## Datensätze
Die im Projekt verwendeten Kaggle-Datensaetze sind aus Lizenzgründen nicht im Repository gespeichert.
Die benötigten Datensätze **train.csv** und **test.csv** müssen mit einem eigenen Kaggle-Profil von 
[Kaggle heruntergeladen werden](https://www.kaggle.com/c/titanic/data?select) geladen werden 
(scrollen bis der Data Explorer, nach dem Ende der Dataset Description, auf der rechten Seite erscheint).
Diese beiden Datensätze in den Unterordner Datensaetze des Projektordners in dieser Art ablegen:
````kotlin
Datensaetze/
├── train.csv      # wird lokal vom Nutzer abgelegt, NICHT ins Repo
├── test.csv       # wird lokal vom Nutzer abgelegt, NICHT ins Repo
└── .gitkeep       # leere Datei, damit der Ordner im Repo sichtbar bleibt
````

---

## Ausführung
Im Projektordner sind zwei Ausführungsskripte enthalten, welche jeweils eine unterschiedliche 
Datenvorverarbeitung ausführen und dafür auf jeweils zwei unterschiedliche Datenvorverarbeitungs-Bibliotheken 
zugreifen. Diese und alle anderen eigens entwickelten Bibliotheken befinden sich in dem Unterordner Bibliotheken.

Das Training mit einer älteren Version der Datenvorverarbeitungs-Bibliothek lässt sich mit 
````bash
python Training_with_DataPreprocessingFunctions_old.py
````
ausführen.

Das Training mit der neuesten Version der Datenvorverarbeitungs-Bibliothek lässt sich mit
````bash
dotenv run -- python Training_with_DataPreprocessingFunctions.py
````
ausführen.

---

## Reproduzierbarkeit
Für die Reproduzierbarkeit der Trainingsergebnisse sind die benötigten Environment-Variablen in der Datei **.env** hinterlegt, auf die beide Ausführungsskripte zugreifen.  
Zusätzlich enthalten beide Skripte mehrere Codezeilen zur Sicherstellung der Reproduzierbarkeit.

---

## Lizenz
Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).