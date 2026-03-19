from Bibliotheken.DataPreprocessingFunctions import preprocess_train, preprocess_testval
from Bibliotheken.DisplayFunctions import plot_training
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib as jl

class NeuralNetworkPipeline:
    # Attribute
    _pipelinename=None
    _model=None
#-----------------------------------------------------------------------------------------------------------------------
    # Optionen für den Trainingsalgorithmus
    _optimizer=None
    _loss=None
    _metrics=None
#-----------------------------------------------------------------------------------------------------------------------
    # Optionen für die Datenvorverarbeitung
    _maxcardhiddencat=None
    _maxcardcat=None
    _maxcardcat=None
    _maxmisspercent=None
    _detection_excep=None
    _drop_excep=None
    _impute_excep=None
    _cols_for_manual_drop=None
    _cols_for_onehot=None
    _cols_for_embedding=None
    _cols_and_scalers=None
    _separate_inputs=None
#-----------------------------------------------------------------------------------------------------------------------
    # Informationen für die Datenverarbeitung
    _columns_to_drop=None
    _manual_columns_to_drop=None
    _impute_and_convert_info=None
    _one_hot_encode_info=None
    _embedding_info=None
    _input_dims=None
    _output_dims=None
    _columnscalers_info=None
    _input_info = None
#-----------------------------------------------------------------------------------------------------------------------
    # Trainingsergebnisse
    _train_chart=None

#-----------------------------------------------------------------------------------------------------------------------
    # Konstruktor
    def __init__(self, name):
        self._name=name


#-----------------------------------------------------------------------------------------------------------------------
    # Methoden
    # Ein- und Ausgabefunktionen
    def _set_name(self, name):
        self._name=name

    def _set_model(self, model):
        self._model=model

    def _show_required_input_shape(self, Data, Target, train_test_split_ratio):

        y = Data[Target]
        Data.drop(labels=Target, axis=1, inplace=True)
        TrainData, ValData, y_train, y_val = train_test_split(Data, y, test_size=train_test_split_ratio)
        (preproc_TrainData, columns_to_drop, manual_columns_to_drop, impute_and_convert_info, one_hot_encode_info,
         embedding_info, input_dims, output_dims, columnscalers_info) = preprocess_train(TrainData,
                                                                                         self._maxcardhiddencat,
                                                                                         self._maxcardcat,
                                                                                         self._maxmisspercent,
                                                                                         self._detection_excep,
                                                                                         self._drop_excep,
                                                                                         self._impute_excep,
                                                                                         self._cols_for_manual_drop,
                                                                                         self._cols_for_onehot,
                                                                                         self._cols_for_embedding,
                                                                                         self._cols_and_scalers,)

        return preproc_TrainData.shape


    def _show_trainchart(self):
        if self._train_chart is None:
            print("Keine Bild gespeichert!")
            return

        dummy_fig = plt.figure(figsize=(21, 8))
        new_manager = dummy_fig.canvas.manager

        new_manager.canvas.figure = self._train_chart
        self._train_chart.set_canvas(new_manager.canvas)
        plt.show()

    def _save_pipeline_in_joblib(self):
        jl.dump(self, self._name+'.joblib')

#-----------------------------------------------------------------------------------------------------------------------
    # Trainings-Voreinstellungen
    def _set_trainalgo(self, optimizer, loss, metrics):
        self._optimizer=optimizer
        self._loss=loss
        self._metrics=metrics

    def _set_separate_inputs(self, separate_inputs):
        self._separate_inputs=separate_inputs

    def _set_preprocessing_options(self, maxcardhiddencat, maxcardcat, maxmisspercent,
                                          detection_excep, drop_excep, impute_excep,
                                          cols_for_manual_drop, cols_for_onehot, cols_for_embedding,
                                          cols_and_scalers):
        self._maxcardhiddencat=maxcardhiddencat
        self._maxcardcat=maxcardcat
        self._maxmisspercent=maxmisspercent
        self._detection_excep=detection_excep
        self._drop_excep=drop_excep
        self._impute_excep=impute_excep
        self._cols_for_manual_drop=cols_for_manual_drop
        self._cols_for_onehot=cols_for_onehot
        self._cols_for_embedding=cols_for_embedding
        self._cols_and_scalers=cols_and_scalers

#-----------------------------------------------------------------------------------------------------------------------
    # Verarbeitungsfunktionen
    def _train(self, Data, Target, train_test_split_ratio, batchsize, epochs, stopper):
        # Datenvorverarbeitung
        y = Data[Target]
        Data.drop(labels=Target, axis=1,inplace=True)
        TrainData, ValData, y_train, y_val = train_test_split(Data, y, test_size=train_test_split_ratio)

        (preproc_TrainData, columns_to_drop, manual_columns_to_drop, impute_and_convert_info, one_hot_encode_info,
         embedding_info, input_dims, output_dims, columnscalers_info) = preprocess_train(TrainData,
                                                                                         self._maxcardhiddencat,
                                                                                         self._maxcardcat,
                                                                                         self._maxmisspercent,
                                                                                         self._detection_excep,
                                                                                         self._drop_excep,
                                                                                         self._impute_excep,
                                                                                         self._cols_for_manual_drop,
                                                                                         self._cols_for_onehot,
                                                                                         self._cols_for_embedding,
                                                                                         self._cols_and_scalers)

        self._columns_to_drop = columns_to_drop
        self._manual_columns_to_drop = manual_columns_to_drop
        self._impute_and_convert_info = impute_and_convert_info
        self._one_hot_encode_info = one_hot_encode_info
        self._embedding_info = embedding_info
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._columnscalers_info = columnscalers_info

        preproc_ValData = preprocess_testval(ValData, columns_to_drop, manual_columns_to_drop, impute_and_convert_info,
                                             one_hot_encode_info, embedding_info, columnscalers_info)

        # Eingabe-Einstellungen
        input_info = {}
        if self._separate_inputs == None or not self._separate_inputs:
            TrainDataForFit = preproc_TrainData
            ValDataForFit = preproc_ValData
            self._input_info = self._separate_inputs
        else:
            TrainDataDictForFit={}
            ValDataDictForFit={}
            TrainDataColumns=list(preproc_TrainData.columns)
            ValDataColumns=list(preproc_ValData.columns)
            for key in self._separate_inputs:
                TrainDataDictForFit[key]=preproc_TrainData[[self._separate_inputs[key]]]
                TrainDataColumns.remove(self._separate_inputs[key])
                ValDataDictForFit[key]=preproc_ValData[[self._separate_inputs[key]]]
                ValDataColumns.remove(self._separate_inputs[key])
                input_info[key] = self._separate_inputs[key]
            TrainDataDictForFit['NormalInputs']=preproc_TrainData[TrainDataColumns]
            ValDataDictForFit['NormalInputs']=preproc_ValData[ValDataColumns]
            TrainDataForFit=TrainDataDictForFit
            ValDataForFit =ValDataDictForFit
            self._input_info=input_info

        # Training
        self._model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
        train_history = self._model.fit(TrainDataForFit,
                                        y_train,
                                        validation_data=(ValDataForFit, y_val),
                                        callbacks=stopper,
                                        batch_size=batchsize,
                                        epochs=epochs)

        # Visualisierung des Trainingsverlaufs
        self._train_chart = plot_training(train_history, "Trainingsverlauf")



    def _predict(self, TestData):
        preproc_TestData = preprocess_testval(TestData, self._columns_to_drop, self._manual_columns_to_drop, self._impute_and_convert_info,
                                             self._one_hot_encode_info, self._embedding_info, self._columnscalers_info)


        if self._input_info == None or not self._input_info:
            predict_Data=preproc_TestData
        else:
            predictDataDict={}
            TestDataColumns = list(preproc_TestData.columns)
            for key in self._input_info:
                predictDataDict[key]=preproc_TestData[[self._input_info[key]]]
                TestDataColumns.remove(self._input_info[key])
            predictDataDict['NormalInputs']=preproc_TestData[TestDataColumns]
            predict_Data =predictDataDict

        prediction = self._model.predict(predict_Data)
        return prediction






