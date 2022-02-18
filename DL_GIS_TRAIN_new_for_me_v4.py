#!/usr/bin/env python
# coding: utf-8

# -- MAIN IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.models import model_from_json
import json, warnings
import os
import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
warnings.filterwarnings('ignore')


def removing_long_nan_pause(data, step, dept_label, interp_method='linear', nan_long_label=5):
    """
    -------------------------------------------------
    функция интерполяции и/или удаления пустых значений с датафрейма
    -------------------------------------------------
    args:
    `data` (pd.DataFrame): данные по скважинам в виде датафрейма
    `step` (float): значение шага глубины
    'dept_label' (str): обозначение глубины в датафрейме
    'interp_method' (str): метод интерполяции
    'nan_long_label' (int): допускаемое кол-во строк с пустыми значениями для интерполяции
    -------------------------------------------------
    return: очищенный от пустых значений датафрейм
    """
    created_df = pd.DataFrame([])  # df with short pauses
    dframes = []

    # creating delta column (delta of DEPT)
    local_df = data.copy()
    local_df = local_df[local_df.isna().any(axis=1)]  # taking only rows with nan
    local_df['delta'] = local_df[dept_label].diff(1)
    local_df['delta'].replace(np.nan, step, inplace=True)

    ind = local_df.last_valid_index()
    df_cr = local_df[local_df.delta > step+0.1]

    if len(local_df) <= nan_long_label:
        local_df2 = local_df  # if short pause, then left it
        # print('only one short pause with length less than or equal to 5 rows')
    elif df_cr.shape[0] == 0 and len(local_df) > nan_long_label:
        local_df2 = local_df.dropna() 
        # print('only one short pause with the length greater than 5 rows')
        # if there sre some long pauses
        # 1 long pause
    elif df_cr.shape[0] == 1:
        n = df_cr.index.item() 
        k = ind - n
        if n > nan_long_label and k > nan_long_label:
            local_df2 = local_df.dropna()
        elif n > nan_long_label and k <= nan_long_label:
            local_df2 = local_df.drop(local_df.index[:n])
        elif n <= nan_long_label and k > nan_long_label:
            local_df2 = local_df.drop(local_df.index[n:])
        else:
            local_df2 = local_df
    elif df_cr.shape[0] > 1:
        L = df_cr.index.to_list()
        ind = local_df.index.to_list()[len(local_df) - 1]
        L.append(ind + 1)
        local_df2 = pd.DataFrame([])
        last_check = 0
        dfs = []
        for i in L:
            local = local_df.loc[last_check:i - 1]
            if len(local) <= nan_long_label:
                dfs.append(local)
            last_check = i
        local_df2 = pd.concat(objs=dfs)
    else:
        # print('something get wrong')
        return None

    dframes.append(local_df2)

    created_df = pd.concat(objs=dframes)  # created df only with short pause (indexes are saved, as in original df)
    df_raw = data.dropna()
    df_new = pd.concat(objs=[df_raw, created_df])  # concatinating df without NaN with short pause
    if interp_method=='linear':
        df_new = df_new.interpolate(method='linear', axis=0)  # interpolating short pause
    elif interp_method=='pad':
        df_new = df_new.interpolate(method='pad', axis=0) 
    else:
        df_new = df_new.dropna()  
        # interpolating short pause
    df_new = df_new.drop(columns='delta')

    return df_new 


def split_to_pic_wout_scaling(df, facies_method="CURRENT_ROW", cut_rows=1, drop_dept=True):
    """
    -------------------------------------------------
    функция разбиения на картинки датафрейма с данными одной скважины
    -------------------------------------------------
    args:
    `df` (pd.DataFrame): датафрейм с данными одной скважины
    `facies_method` (str): метод выбора значения целевой переменной:
        "CURRENT_ROW" - серединное значение
        "WINDOW" - часто встречаемое значение
        "PROBABILITY" - список вероятностей
    'cut_rows' (int): кол-во строк в картинке
    -------------------------------------------------
    return: список датафреймов с признаками, список список со значениями целевой переменной
    """
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)
    labels = []
    dfs = []
    total_rows = df.shape[0]
    i = cut_rows
    while i < total_rows - cut_rows:
        start_row = i - cut_rows
        last_row = i + cut_rows
        test_df = df.loc[start_row:last_row]
        try:
            if facies_method == "WINDOW":
                facies = test_df['Facies_IPSOM_IPSOM'].value_counts().idxmax()
            if facies_method == "CURRENT_ROW":
                facies = test_df['Facies_IPSOM_IPSOM'][i]
            elif facies_method == "PROBABILITY":
                value_counts = test_df.Facies_IPSOM_IPSOM.value_counts()
                values = value_counts.index.to_list()
                counts = value_counts.to_list()
                total = sum(counts)
                prob = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0} 
                for k, j in zip(values, counts):
                    prob[k] = round(j/total, 2)
                    facies = list(prob.values())
        except Exception as ex:
            print(ex)
            facies = -1
        test_df = test_df.drop(['Facies_IPSOM_IPSOM', 'WELL', 'DEPT'], axis=1)
        labels.append(facies)
        dfs.append(test_df.to_numpy())
        i += 1
    if drop_dept:
        return dfs, labels
    else:
        return dfs, labels, df['DEPT'][1:-1].tolist()


def reshape_for_cnn(X, y, facies_method, num_classes=5):
    """
    -------------------------------------------------
    функция преобразования списка признаков и целевых переменных в массивы numpy - формат для входа в сверточную нейронную сеть 
    -------------------------------------------------
    args:
    `X` (list): список датафреймов с признаками 
    `y` (list): список со значениями целевой переменной
    'facies_method' (int): метод выбора значения целевой переменной
    'num_classes' (int): кол-во классов 
    -------------------------------------------------
    return: массив numpy признаков, массив numpy со значенями целевой переменной
    """
    X_flat = [val for sublist in X for val in sublist]
    X_flat = np.asarray(X_flat)
    X_flat= X_flat.reshape( X_flat.shape[0], X_flat.shape[1], X_flat.shape[2], 1)
    y_flat = [val for sublist in y for val in sublist]
    if facies_method=='PROBABILITY':
        y_flat=y_flat
    else:
        y_flat = [x - 1 for x in y_flat]
        y_flat = keras.utils.to_categorical(y_flat, num_classes)
    y_flat = np.asarray(y_flat)
    
    return X_flat, y_flat


def plot_metrics(hist):
    """
    -------------------------------------------------
    функция генерации графика loss и accuracy 
    -------------------------------------------------
    args:
    `history` (dict): словарь, где хранятся значения loss и accuracy на каждой эпохе
    -------------------------------------------------
    return: график
    """
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


class GISInterpDLPredict:
    """
    Класс для загрузки и чискти данных, генерации фреймов данных для обучения, 
    переобучения существующей модели и обучения новой модели,
    прогноза с подключенной обученной моделью и созданной новой моделью, 
    вызгрузки созданной модели
    """
    def __init__(self, title, model_info, model_name, verbose=True):
        """
        -------------------------------------------------
        args:
        `title` (str): название экземпляра
        `model_info` (dict): параметры модели
        `model_name` (str): название модели
        `verbose` (bool): Если True - выводит информацию на консоль. default value = True
        -------------------------------------------------
        attrs:
        `self.__fitted_model` - загруженная модель
        `self.new_model` - новая модель. default value = None
        `self.refitted_model` - переобученная модель
        `self.load_data_status` - статус загрузки данных. default value = None
        `self.split_data_status` - статус деление данных на тренировочные и тестовые. default value = None
        `self.fit_status` - статус модели. default value = None
        """
        self.title = title
        self.model_info = model_info
        self.model_name = model_name
        logging.basicConfig(filename='example.log', encoding='utf-8',
                            format='%(asctime)s - %(message)s', level=logging.INFO)
        if model_info[model_name][6]=='weights':
            with open(model_info[model_name][7]) as json_data:
                json_str = json.load(json_data)
            model = model_from_json(json_str)
            model.load_weights(model_info[model_name][0])
        else:
            model = keras.models.load_model(model_info[model_name][0])
        logging.info(f'Model successfully loaded!')
        self.__fitted_model = model
        self.new_model = None
        self.refitted_model = model
        self.verbose = verbose
        self.load_data_status = None
        self.split_data_status = None
        self.fit_status = None
    

    def load_data(self, raw_data, raw_data_info):
        """
        -------------------------------------------------
        метод загрузки набор данных для модели
        -------------------------------------------------
        args:
        `raw_data` (pd.DataFrame): данные по скважинам в виде датафреймов
        `raw_data_info` (dict): описание загруженных данных
        -------------------------------------------------
        attrs:
        `self.raw_data` - исходный неизменяемый набор данных
        `self.df_train` - обучающая выборка
        `self.df_test` - тестовая выборка
        `self.data` - окончательные наборы данных для модели
        -------------------------------------------------
        return: none
        """
        try:
            self.dict_for_date = { i: raw_data[raw_data.WELL==i].DATE.apply(str).values[0].split()[0] for i in raw_data.WELL.unique()}
            self.raw_data = raw_data.drop(columns=['DATE'])
            self.raw_data_info = raw_data_info
            self.target_exist = True if raw_data_info['target_name'] in raw_data.columns.tolist() else False
            colums = self.raw_data.columns.tolist()
            if not all(item in colums for item in self.model_info[self.model_name][1]):
                return None
            self.raw_data.replace(self.raw_data_info.get('NULL'), np.nan, inplace=True)
            logging.info(f'Data cleaned!')
            cols = self.model_info[self.model_name][1] + [self.raw_data_info['well_label'],
                    self.raw_data_info['dept_label'], self.raw_data_info['target_name']]
            
            df_train = self.raw_data[self.raw_data[self.raw_data_info['well_label']].isin(self.model_info[self.model_name][4])].copy()
            df_train = df_train.dropna(subset = [self.raw_data_info['target_name']], axis=0)
            df_list=[]
            for well in  df_train[self.raw_data_info['well_label']].unique().tolist():
                data_local = df_train[df_train[self.raw_data_info.get('well_label')]==well]
                if data_local.isna().values.any():
                    data_local = removing_long_nan_pause(data_local, self.raw_data_info.get('step'), dept_label=self.raw_data_info['dept_label'])
                df_list.append(data_local)
            df_train = pd.concat(objs=df_list)
            
            if self.target_exist:
                data_corr_col = self.raw_data[cols]
                data_corr_col = self.raw_data.dropna(subset =[ self.raw_data_info['target_name']], axis=0)
                df_list=[]
                for well in  data_corr_col[self.raw_data_info['well_label']].unique().tolist():
                    data_local = data_corr_col[data_corr_col[self.raw_data_info.get('well_label')]==well]
                    if data_local.isna().values.any():
                        data_local = removing_long_nan_pause(data_local, self.raw_data_info.get('step'), dept_label=self.raw_data_info['dept_label'])
                    df_list.append(data_local)
                data = pd.concat(objs=df_list)
                
                test_wells=[]
                for well in data[raw_data_info['well_label']].unique().tolist():
                    if well not in self.model_info[self.model_name][4]:
                        test_wells.append(well)
                df_test = data.loc[data[raw_data_info['well_label']].isin(test_wells)]

                if len(df_test)==0:
                    df_test = None
            else:
                data_corr_col = self.raw_data[cols[:-1]]
                df_list=[]
                for well in self.raw_data[self.raw_data_info['well_label']].unique().tolist():
                    data_local = data_corr_col[data_corr_col[self.raw_data_info.get('well_label')]==well]
                    data_local = data_local.dropna()
                    df_list.append(data_local)
                data = pd.concat(objs=df_list)
                df_test = data
            self.df_train = df_train 
            self.df_test = df_test
            logging.info('Train and test datas created!')
            logging.info('Data is loaded')
            self.load_data_status = 'Data is loaded'
            if df_test is None:
                self.data = df_train.copy()
            else:
                self.data = pd.concat(objs=[df_train, df_test])
        except Exception as ex:
            print(ex)
            logging.warning(f'Error in load function: {ex}')

    
    def __gen_pic__(self, scaler, train_data, valid_data, cut_rows, facies_method, new_data=False):
        """
        -------------------------------------------------
        метод генерации картинок
        -------------------------------------------------
        args:
        `scaler` (sklearn.object): метод масштабирования числовых данных
        `train_data` (pd.DataFrame): обучающая выборка
        `valid_data` (pd.DataFrame): тестовая выборка
        `cut_rows` (int): количество строк в одной картинке
        `facies_method` (str): метод извлечение целевых значения из картинок
        `new_data` (bool): Если True - масштабируют числовых данных на новом методе. default value = False
        -------------------------------------------------
        return: массивы numpy
        -------------------------------------------------
        methods: split_data()
        """
        try:
            if not new_data:
                train_data[self.model_info[self.model_name][1]] = scaler.transform(train_data[self.model_info[self.model_name][1]])
                valid_data[self.model_info[self.model_name][1]] = scaler.transform(valid_data[self.model_info[self.model_name][1]])
            else:
                scaler_new =MinMaxScaler(feature_range=(-1,1))
                train_data[self.model_info[self.model_name][1]] = scaler_new.fit_transform(train_data[self.model_info[self.model_name][1]])
                valid_data[self.model_info[self.model_name][1]] = scaler_new.transform(valid_data[self.model_info[self.model_name][1]])
                
            X_list_train = []
            y_list_train = []       
            for well in  train_data[self.raw_data_info['well_label']].unique().tolist():
                df = train_data[train_data[self.raw_data_info.get('well_label')]==well]
                X, y = split_to_pic_wout_scaling(df, facies_method, cut_rows)
                X_list_train.append(X)
                y_list_train.append(y)
                    
            X_list_valid = []
            y_list_valid = []
            for well in  valid_data[self.raw_data_info['well_label']].unique().tolist():
                df = valid_data[valid_data[self.raw_data_info.get('well_label')]==well]
                X, y = split_to_pic_wout_scaling(df, facies_method,cut_rows)
                X_list_valid.append(X)
                y_list_valid.append(y)
                
            X_train, y_train = reshape_for_cnn(X_list_train, y_list_train, facies_method)
            X_valid, y_valid = reshape_for_cnn(X_list_valid, y_list_valid, facies_method)    
            logging.info('Train and valid pictures generated!')
            return X_train, y_train, X_valid, y_valid
        except Exception as ex:
            print(ex)
            logging.warning(f'Error in gen_pic function {ex}')


    def __split_data__(self, valid_mode='valid_new', split_by_well=False, valid_wells=None, train_wells=None, test_portion=0.2, random_state=42):
        """
        -------------------------------------------------
        метод разделения фрейм данных на два подмножества : для обучающих данных и для тестовых данных. 
        -------------------------------------------------
        args:
        `valid_mode` (str): валидация модели на основе новых или рандомных данных. default value = 'valid_new'
        `split_by_well` (bool): Если True - разделения рандомна по скважинам. default value = False
        `valid_wells` (list): список скважин для валидация. default value = None
        `train_wells` (list): список скважин для обучения. default value = None
        `test_portion` (float): представлять долю набора данных, включаемую в тестовое разделение. default value = 0.2
        `random_state` (int): управляет перемешиванием данных перед применением разделения. default value = 42
        -------------------------------------------------
        return: True or None
        """
        try:
            if self.load_data_status != 'Data is loaded':
                logging.warning('Data is not loaded !!!!!!')
                return None
            if not self.target_exist:
                if self.verbose:
                    print('Cannot do a split with no target value')
                logging.warning('Cannot do a split with no target value')
                return None
            else:
                if valid_mode is None:
                    if valid_wells is None:
                        logging.warning('Valid mode and valid wells are NONE')
                        return None
                    if train_wells is None:               
                        train_data = self.df_train.copy()
                    else:
                        train_data = self.data.loc[self.data[self.raw_data_info['well_label']].isin(train_wells)]
                    valid_data = self.data.loc[self.data[self.raw_data_info['well_label']].isin(valid_wells)]
                else:
                    if self.df_test is None:
                        logging.warning('No new wells')
                        if self.verbose:
                            print('No new wells')
                        if not split_by_well:
                            train_data, valid_data = train_test_split(self.df_train, test_size=test_portion, shuffle=True, 
                                                                        random_state=random_state)                          
                        else:
                            wells_list = self.df_train[self.raw_data_info['well_label']].unique().tolist()
                            if random_state is not None:
                                np.random.seed(random_state)
                            valid_w = np.random.choice(wells_list, int(len(wells_list) * test_portion),replace=False)
                            train_w = list(set(wells_list) - set(valid_w))
                            train_data = self.df_train.loc[self.df_train[self.raw_data_info.get('well_label')].isin(train_w)]
                            valid_data =  self.df_train.loc[self.df_train[self.raw_data_info.get('well_label')].isin(valid_w)]
                    else:
                        data_to_split = pd.concat(objs=[self.df_train, self.df_test])
                        if valid_mode == 'valid_new':
                            train_data = data_to_split
                            valid_data = self.df_test
                        elif valid_mode == 'valid_random':
                            train_data = data_to_split
                            if not split_by_well:
                                valid_data = train_data.sample(frac = test_portion)
                            else:
                                wells_list = train_data[self.raw_data_info['well_label']].unique().tolist()
                                if random_state is not None:
                                    np.random.seed(random_state)
                                valid_w = np.random.choice(wells_list, int(len(wells_list) * test_portion),replace=False)
                                valid_data =  train_data.loc[train_data[self.raw_data_info.get('well_label')].isin(valid_w)] 
                        else:
                            train_data = self.df_train.copy()
                            valid_data = self.df_test.copy() 
                                                        
            if self.df_test is None:    
                X_train, y_train, X_valid, y_valid = self.__gen_pic__(self.model_info[self.model_name][3], train_data, valid_data, 
                                                                    self.model_info[self.model_name][2], 
                                                                    self.model_info[self.model_name][5], new_data=False)
            else:
                X_train, y_train, X_valid, y_valid = self.__gen_pic__(self.model_info[self.model_name][3], train_data, valid_data, 
                                                                    self.model_info[self.model_name][2], 
                                                                    self.model_info[self.model_name][5], new_data=True)
                
            self.X_train, self.X_valid = X_train, X_valid
            self.y_train, self.y_valid = y_train, y_valid
            self.split_data_status = 'Data is splitted'
            logging.info('Data is splitted !!!')
            self.valid_mode = valid_mode
            self.train_data, self.valid_data = train_data, valid_data
            
            if self.verbose:
                print(f'X_train shape is {self.X_train.shape}')
                print(f'X_valid shape is {self.X_valid.shape}')
                print(f'y_train shape is {self.y_train.shape}')
                print(f'y_valid shape is {self.y_valid.shape}')
                print(f'df shape is {len(self.df_train)}')
                print(f'number of wells is {len(self.df_train.WELL.unique().tolist())}')
                print(self.split_data_status)
            
            return True
        except Exception as ex:
            print(ex)
            logging.warning(f'Error in split function {ex}')
            return None
    
    
    def __fit_evalute__(self, model, X, y):
        """
        -------------------------------------------------
        метод генерация метрики
        -------------------------------------------------
        args:
        `model` (model): метод масштабирования числовых данных
        `X`, `y` (array): входные данные для проверки  
        -------------------------------------------------
        return: список метрик
        -------------------------------------------------
        methods: fit()
        """
        try:
            evaluate = model.evaluate(X, y, verbose=0)
            y_pred_ = model.predict(X)
            y_pred = [np.argmax(v) for v in y_pred_]
            y_true = [np.argmax(v) for v in y]
            f1_macro_valid = f1_score(y_true, y_pred, average='macro')
            logging.info("Metrics generated !!!")
            return evaluate, f1_macro_valid
        except Exception as ex:
            logging.warning(f"Erro in fit_evalute function !!!")
            return (0, 0), 0


    def fit(self, refit_exist_model=True, new_model_name=None, new_model_info=dict(), 
            callback=None, batch_size=256, epochs=150, plot_met=True, valid_mode='valid_new', 
            split_by_well=False, valid_wells=None, train_wells=None, 
            test_portion=0.2, random_state=42):
        """
        -------------------------------------------------
        метод обучение модели
        -------------------------------------------------
        args:
        `refit_exist_model` (bool): Если True - обучает существующую модель. default value = True
        `new_model_name` (str): название модели. default value = None
        `new_model_info` (вшсе): параметры модели. default value = {}
        `callback` (obj): обратный вызов при обучение
        `batch_size` (int):  количество выборок, обработанных перед обновлением модели
        `epochs` (int): количество полных проходов через обучающий набор данных
        `valid_mode` (str): валидация модели на основе новых или рандомных данных. default value = 'valid_new'
        `split_by_well` (bool): Если True - разделения рандомна по скважинам. default value = False
        `valid_wells` (list): список скважин для валидация. default value = None
        `train_wells` (list): список скважин для обучения. default value = None
        `test_portion` (float): представлять долю набора данных, включаемую в тестовое разделение. default value = 0.2
        `random_state` (int): управляет перемешиванием данных перед применением разделения. default value = 42
        -------------------------------------------------
        return: словарь метрик
        """
        try:
            split_data = self.__split_data__(valid_mode, split_by_well, valid_wells, train_wells, test_portion, random_state)
            if split_data is None:
                return None
            if refit_exist_model:
                model = self.refitted_model
                if self.model_info[self.model_name][6] != 'full':
                    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                
                if self.valid_mode is None:
                    logging.info('only predict')
                    if self.verbose:
                        print('only predict')
                elif self.df_test is None:
                    logging.info('Loaded data contains only wells that used in training model')
                    if self.verbose:
                        print('Loaded data contains only wells that used in training model')
                else:
                    hist = model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, 
                                    validation_data=(self.X_valid, self.y_valid), callbacks=callback)
                    if plot_met:
                        plot_metrics(hist)
                logging.info("Old model was refitted")
                self.fit_status = "Old model was refitted"
                self.refitted_model = model
            else:
                if new_model_info[new_model_name][6] == 'full':
                    model = keras.models.load_model(new_model_info[new_model_name][0])
                    hist = model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, 
                            validation_data=(self.X_valid, self.y_valid), callbacks=callback)
                    if plot_met:
                        plot_metrics(hist)
                else:
                    with open(new_model_info[new_model_name][7]) as json_data:
                        json_str = json.load(json_data)
                    model = model_from_json(json_str)
                    model.load_weights(new_model_info[new_model_name][0])
                    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
                    hist = model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, 
                                validation_data=(self.X_valid, self.y_valid), callbacks=callback)
                    if plot_met:
                        plot_metrics(hist)
                self.new_model = model
                self.new_model_info = new_model_info[new_model_name]
                logging.info("Generated new model")
                self.fit_status = "Generated new model"
            
            X = np.vstack((self.X_train, self.X_valid))
            y = np.vstack((self.y_train, self.y_valid))
                
            eval_valid, f1_macro_valid = self.__fit_evalute__(model, self.X_valid, self.y_valid)
            eval_train, f1_macro_train = self.__fit_evalute__(model, self.X_train, self.y_train)
            eval_all, f1_macro_all = self.__fit_evalute__(model, X, y)
            
            metrics_dict=dict()
            metrics_dict['valid_loss'], metrics_dict['valid_acc'] = eval_valid
            metrics_dict['f1_macro_valid'] = f1_macro_valid
            metrics_dict['train_loss'], metrics_dict['train_acc'] = eval_train
            metrics_dict['f1_macro_train'] = f1_macro_train
            metrics_dict['train_and_valid_loss'], metrics_dict['train_and_valid_ acc'] = eval_all
            metrics_dict['f1_macro_all'] = f1_macro_all
            self.fit_metrics_dict = metrics_dict
            
            return metrics_dict
        except Exception as ex:
            print(ex)
            logging.warning(f'Error in fit function {ex}')
            
            return None
    

    def predict(self, model='old', wells=None, add_train_data=False):
        """
        -------------------------------------------------
        метод прогноза
        -------------------------------------------------
        args:
        `model` (bool): модель для прогназа. default value = None
        `wells` (str): список скважин для прогноза. default value = None
        `add_train_data` (bool): Если True - делает прогноз с учетом обучающей выборки default value = false
        -------------------------------------------------
        return: словарь метрик
        """
        try:
            estimator_info = dict()
            if model == 'new':
                if self.fit_status == "Generated new model":
                    logging.info('Using new_model for predict')
                    estimator_info['model_path'] = self.new_model_info[0]
                    estimator_info['scaler'] = self.new_model_info[3]
                    estimator_info['wells_list'] = self.new_model_info[4]
                    estimator_info['facies_method'] = self.new_model_info[5]
                    estimator_info['saving_type'] = self.new_model_info[6]
                    estimator_info['arch_path'] = self.new_model_info[7]
                    model = self.new_model
                else:
                    return None
            elif model == 'refit':
                if self.fit_status == "Old model was refitted":
                    logging.info('Using refitted_model for predict')
                    model = self.refitted_model
                    estimator_info['model_path'] = self.model_info[self.model_name][0]
                    estimator_info['scaler'] = self.model_info[self.model_name][3]
                    estimator_info['wells_list'] = self.model_info[self.model_name][4]
                    estimator_info['facies_method'] = self.model_info[self.model_name][5]
                    estimator_info['saving_type'] = self.model_info[self.model_name][6]
                    estimator_info['arch_path'] = self.model_info[self.model_name][7]
                else:
                    return None
            elif model == 'old':
                model = self.__fitted_model
                logging.info('Using old_model for predict')
                estimator_info['model_path'] = self.model_info[self.model_name][0]
                estimator_info['scaler'] = self.model_info[self.model_name][3]
                estimator_info['wells_list'] = self.model_info[self.model_name][4]
                estimator_info['facies_method'] = self.model_info[self.model_name][5]
                estimator_info['saving_type'] = self.model_info[self.model_name][6]
                estimator_info['arch_path'] = self.model_info[self.model_name][7]
            else:
                if self.verbose:
                    print('Cooshe between: old, new, refit!')
                return None
            well_label = self.raw_data_info['well_label']
            train_wells = self.df_train[well_label].unique().tolist()
            test_wells = [] if self.df_test is None else self.df_test[well_label].unique().tolist()
            
            if wells is None:
                wells = train_wells
            else:
                if add_train_data:
                    wells = list(set(train_wells + wells))
                else:
                    wells = [i for i in wells if i in test_wells] # only test df ~train_df_well
            if wells == []:
                return None
            data_for_pred = self.data[self.data[well_label].isin(wells)].copy()
            wells = sorted(wells)
            self.predict_metric_dict= {'PREDICT_VALEUS': {}, 'TRUE_VALUES': {}, 'METRICS': {}}
            scaler = self.model_info[self.model_name][3]
            data_for_pred[self.model_info[self.model_name][1]] = scaler.fit_transform(data_for_pred[self.model_info[self.model_name][1]])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            logging.info("Model compiled! ")
            metric_dict = dict()
            x_list, y_list = [], []
            for well in wells:
                df = data_for_pred[data_for_pred[well_label]==well].copy()
                X, y, dept = split_to_pic_wout_scaling(df, self.model_info[self.model_name][5], self.model_info[self.model_name][2], drop_dept=False)
                x_list.append(X)
                y_list.append(y)
                X, y = reshape_for_cnn([X], [y], self.model_info[self.model_name][5])
                y_pred = model.predict(X)
                loss , accr = model.evaluate(X, y, verbose=0)
                y_pred = [np.argmax(v) for v in y_pred]
                y_true = [np.argmax(v) for v in y]
                logging.info(f"Predict for well---{well}")
                self.predict_metric_dict['PREDICT_VALEUS'][f"{well}_{self.dict_for_date.get(well, 'NaN')}"] = {v:y_pred[k] for k, v in enumerate(dept)}
                if y_true[0] == -1 and y_true[1]==-1:
                    self.predict_metric_dict['TRUE_VALUES'][f"{well}_{self.dict_for_date.get(well, 'NaN')}"] = {v:None for k, v in enumerate(dept)}
                else:
                    self.predict_metric_dict['TRUE_VALUES'][f"{well}_{self.dict_for_date.get(well, 'NaN')}"] = {v:y_true[k] for k, v in enumerate(dept)}
                    f1_macro = f1_score(y_true, y_pred, average='macro')
                    metric_dict[well] = (loss, f1_macro)
                    self.predict_metric_dict['METRICS'][f"{well}_{self.dict_for_date.get(well, 'NaN')}"] = {'f1_macro': f1_macro, 'loss': loss, 'accuracy': accr}
                if self.verbose:
                    print(f'Well {well}')
                    print(f'X shape is: {X.shape}')
                    print(f'y shape is: {y.shape}')
                    print(len(y_true))
                    print(len(y_pred))
                    print(y_true[:5])
                    print(y_pred[:5])
                    print(f'accuracy {accuracy_score(y_true, y_pred)}')
                    print(f'X shape for well {well} is: {X.shape}')
                    print(f'y shape for well {well} is: {len(y_true)}')
                    print(f'for well {well}: ', classification_report(y_true, y_pred))
            logging.info('Predicting for all well')
            X_all, y_all = reshape_for_cnn(x_list, y_list, self.model_info[self.model_name][5])
            y_all_pred = model.predict(X_all)
            loss_all, accuracy = model.evaluate(X_all, y_all, verbose=0)
            y_all_pred = [np.argmax(v) for v in y_all_pred]
            y_all_true = [np.argmax(v) for v in y_all]
            if y_all_true[0] != -1:
                f1_macro_all = f1_score(y_all_true, y_all_pred, average='macro')
                metric_dict['all_wells'] = (loss_all, f1_macro_all)
                self.predict_metric_dict['METRICS']['ALL WELLS'] = {'f1_macro': f1_macro_all, 'loss': loss_all, 'accuracy': accuracy}
            self.predict_metric_dict['ESTIMATOR'] = estimator_info
            
            return self.predict_metric_dict
        except Exception as ex:
            print(ex)
            logging.warning(f'Error in predict function {ex}')
            
            return None

    
    def export_model(self, name, path_to_save="", save_type='Full model'):
        """
        -------------------------------------------------
        выгрузка модели 
        -------------------------------------------------
        args:
        `name` (str): название для экспорта модели (без расширение)
        `path_to_save` (str): путь до файла, default value = ""
        `save_type` (str): тип экспорта. В виде целой модели или архитектуры. default value = "Full model"
        -------------------------------------------------
        return: словарь метрик
        """
        try:
            if self.fit_status is None:
                logging.info('Before export need fit model!')
                if self.verbose:
                    print('Fit model first')
                return None
            elif self.fit_status == "Old model was refitted":
                model = self.refitted_model
                name += "refitted_model."
            else:
                model = self.new_model
                name += "new_model."
            file_name =os.path.join(path_to_save, f'{name}')
            if save_type == "weights":
                model.save_weights(file_name + 'h5')
                arch = model.to_json()
                with open(file_name + 'json', 'w') as file:
                    json.dump(arch, file)
                logging.info('Model weights and architexture are saved!')
            else:
                model.save(file_name + 'hdf5')
                logging.info('Model saved!')
            
            return True
        except Exception as ex:
            print(ex)
            logging.warning(f"Error in export_model function : {ex}")
            return False
