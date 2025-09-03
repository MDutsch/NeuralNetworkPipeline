# Hilfsfunktion zur Darstellung der Modellarchitektur
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from textwrap import fill
def model_to_text_colored(model):
    lines = []
    colors = []
    for layer in model.layers:
        line = f"{layer.name}: {layer.__class__.__name__}"
        color = "black"
        if hasattr(layer, "units"):
            line += f", units={layer.units}"
            color = "black"
        if hasattr(layer, "activation"):
            line += f", activation={layer.activation.__name__}"
            if "relu" in layer.activation.__name__ or "sigmoid" in layer.activation.__name__:
                color = "red"
        if hasattr(layer, "rate"):  # Dropout
            line += f", rate={layer.rate}"
            color = "magenta"
        if "BatchNormalization" in layer.__class__.__name__:
            color = "green"
        lines.append(line)
        colors.append(color)
    return lines, colors


def plot_training(model,stopper,train_history,epochs, batchsize,plot_title):
    fig = plt.figure(figsize=(21,8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1])
    # Trainingsverlauf oben links platzieren
    ax1 = fig.add_subplot(gs[0, 0])  # oben links
    # Auslesen der Epochen und des besten Validierungsfehlers
    n_epochs_trained = len(train_history.history['loss'])
    best_epoch = np.argmin(train_history.history['val_loss'])
    best_val_loss = train_history.history['val_loss'][best_epoch]
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    # Darstellung des Diagramms
    ax1.plot(loss, label='Train Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.axvline(x=best_epoch, linestyle='--', color='red')
    ax1.text(0.95, 0.95, f'Finaler Val.-Fehler (Ep. {best_epoch}): {best_val_loss:.4f}',
                  transform=ax1.transAxes, fontsize=12, va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.set_title(plot_title+f'(Early Stopping bei Epoche {n_epochs_trained})')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #ax1.set_xticks([np.arange(0,301,25)])
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    #-------------------------------------------------------------------------------------------------------
    # Parametertabelle unten links platzieren
    ax2 = fig.add_subplot(gs[1, 0])  # unten links
    ax2.axis('off')
    # Auslesen der Parameter
    train_params = [
    ["Parameter","Epochs (max)", "Batchsize", "min_delta","Patience", "Optimizer", "Loss-Function", "Metric"],
    ["Werte", str(epochs), str(batchsize), str(abs(stopper.min_delta)), str(stopper.patience), str(model.optimizer.__class__.__name__),
     str(model.loss), str(list(train_history.history.keys())[0])]
    ]
    # Anpassen der Spaltengrößen nach längsten Eintrag
    tablevalues_matrix = np.array(train_params)
    columnwidths = []
    for colindex in range(0, tablevalues_matrix.shape[1]):
        colcellwidths = []
        for rowindex in range(0, tablevalues_matrix.shape[0]):
            colcellwidths.append(len(tablevalues_matrix[rowindex][colindex]))
        columnwidths.append(max(colcellwidths))
    # Erstellen der Tabelle
    tabelle=ax2.table(
        cellText=train_params,
        cellColours=[["grey","white", "lightgrey","white", "lightgrey"
                      ,"white", "lightgrey", "white"],
                     ["grey", "white", "lightgrey", "white", "lightgrey"
                         , "white", "lightgrey", "white"]],
        colWidths=[(1 / sum(columnwidths) * item) for item in columnwidths],
        bbox=[0, 0.25, 1, 0.5],
        loc="center"
    )
    tabelle.auto_set_font_size(False)
    tabelle.set_fontsize(9)
    #-------------------------------------------------------------------------------------------------------------
    # Modelarchitektur in der rechten Spalte unten darstellen
    ax3=fig.add_subplot(gs[:, 1])
    ax3.set_title("Modell-Architektur")
    # Maße für Darstellung festlegen
    ax3.axis("off")
    top = 1
    bottom = 0
    left = 0
    right = 1
    ax3.set_xlim(left, right)
    ax3.set_ylim(bottom, top)
    # Breite der Boxen ist immer gleich sein
    box_width = 1
    # Höhe der Boxen auf maximale Anzahl von Schichten zugeschnitten
    maxlayers=15
    box_height = 1 / (2 * maxlayers)
    # Farb-Festlegung
    color_dict = {
        'Dense': "lightgray",
        'BatchNormalization': "orange",
        'Dropout': "skyblue",
        'Activation': "lime"}
    # Startwert für ybox
    xbox = ((right - left) - box_width) / 2
    ybox = (top - bottom) - (box_height + box_height * 0.05)
    # Darstellung der Schichten über Auslesen aus dem Modell
    for layer in model.layers:
        typeoflayer = str(layer.__class__.__name__)
        color = color_dict.get(typeoflayer)
        cfg = layer.get_config()
        if typeoflayer == 'Dense':
            Text = ("Layer:"+typeoflayer +
                    "| Input=" + str(layer.input.shape[1]) +
                    "| Units=" + str(cfg["units"]) +
                    "| Activation=" + str(cfg["activation"]))
        elif typeoflayer == 'BatchNormalization':
            Text = ("Layer:"+typeoflayer + "| Input="
                    + str(layer.input.shape[1]))
        elif typeoflayer == 'Dropout':
            Text = ("Layer:"+typeoflayer + "| Input="
                    + str(layer.input.shape[1]) + "| Rate=" + str(layer.rate))
        elif typeoflayer == 'Activation':
            Text = ("Layer:"+typeoflayer + "| Input="
                    + str(layer.input.shape[1]) + "| Activation=" + str(cfg["activation"]))
        else:
            Text = ("Layer:"+typeoflayer + "|Layer-Typ in Darstellungsfunktion nicht definiert")
            color = "red"
        box = FancyBboxPatch((xbox, ybox), box_width, box_height, boxstyle="square, pad=0", facecolor=color)
        ax3.add_patch(box)
        ax3.text(xbox + (box_width / 2), ybox + box_height / 2, Text, ha="center", va="center",fontsize=10, wrap=True)
        if model.layers.index(layer) < len(model.layers) - 1:
            ax3.annotate("", xy=((right - left) / 2, ybox - box_height),
                        xytext=((right - left) / 2, ybox),
                        arrowprops=dict(arrowstyle="->", lw=1))
        ybox = ybox - 2 * box_height
    fig.subplots_adjust(left=0.05, right=1, top=0.95, bottom=0.0)
    fig.tight_layout()
    return fig