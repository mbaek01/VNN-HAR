import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader

from utils import Normalizer,components_selection_one_signal

def get_data(dataset, batch_size, flag="train"):
    if flag == 'train':
        shuffle_flag = True
    else:
        shuffle_flag = False
  
    dataset = Dataset(dataset, flag)
    data_loader = DataLoader(dataset,
                             batch_size = batch_size,
                             shuffle = shuffle_flag,
                             num_workers = 0,
                             drop_last = False)
    return data_loader

class PAMAP2(object):
    def __init__(self, args):
        self.args          = args
        self.data_path     = args.data_path
        self.windowsize    = args.windowsize
        self.freq1         = args.freq1
        self.freq2         = args.freq2
        self.sampling_freq = args.sampling_freq
        self.exp_mode      = args.exp_mode
        self.datanorm_type = args.datanorm_type
        self.train_vali_quote = args.train_vali_quote
        self.LOCV_keys = [[1],[2],[3],[4],[5],[6],[7],[8]] # 9 omitted
        self.all_keys = [1,2,3,4,5,6,7,8]

        self.index_of_cv = 0

        # if self.exp_mode == "LOCV":
        #     self.num_of_cv = len(self.LOCV_keys)

        # else:
        #     self.num_of_cv = 1

        self.used_cols = [1,# this is "label"
                            # TODO check the settings of other paper 
                            # the second column is heart rate (bpm) --> ignore?
                            # each IMU sensory has 17 channals , 3-19,20-36,38-53
                            # the first temp ignores
                            # the last four channel according to the readme are invalide
                            # 4, 5, 6,   7, 8, 9,   10, 11, 12,   13, 14, 15,        # IMU Hand
                            # 21, 22, 23,   24, 25, 26,   27, 28, 29,   30, 31, 32,  # IMU Chest
                            # 38, 39, 40,    41, 42, 43,    44, 45, 46,    47, 48, 49   # IMU ankle
                            # +- 16g scale accelerometer 
                            4, 5, 6,      10, 11, 12,      # IMU Hand
                            21, 22, 23,   27, 28, 29,    # IMU Chest
                            38, 39, 40,   44, 45, 46,   # IMU ankle
                            ]

        self.col_names = ['activity_id',
                            'acc_x_hand', 'acc_y_hand', 'acc_z_hand',
                            'gyro_x_hand', 'gyro_y_hand', 'gyro_z_hand',
                            'acc_x_chest', 'acc_y_chest', 'acc_z_chest',
                            'gyro_x_chest', 'gyro_y_chest', 'gyro_z_chest',
                            'acc_x_ankle', 'acc_y_ankle', 'acc_z_ankle',
                            'gyro_x_ankle', 'gyro_y_ankle',  'gyro_z_ankle'
                            ]

        self.label_map = [ 
            (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'nordic walking'),

            (12, 'ascending stairs'),  # 8
            (13, 'descending stairs'), # 9
            (16, 'vacuum cleaning'),   # 10
            (17, 'ironing'),           # 11

            (24, 'rope jumping')       # 12
        ]

        # Select which sensor columns to use
        self.sensor_filter      = ["acc", "gyro"]
        # self.pos_filter         = ["hand", "chest", "ankle"]

        # self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.pos_select, self.pos_filter, self.col_names[1:], "position")
        self.selected_cols  = self.Sensor_filter_acoording_to_pos_and_type(args.sensor_select, self.sensor_filter, self.col_names[1:], "Sensor Type") # self.col_names[1:] to self.selected_cols

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}

        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [0]
        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        self.file_encoding = {'subject101.dat':1, 'subject102.dat':2, 'subject103.dat':3, 
                            'subject104.dat':4, 'subject105.dat':5, 'subject106.dat':6,
                            'subject107.dat':7, 'subject108.dat':8 } # 'subject109.dat':9 - not used

        # 'subject101.dat', 'subject102.dat', 'subject103.dat',  'subject104.dat', 
        # 'subject105.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat'

        # subjects except the test_sub
        # self.train_keys   = [i for i in range(self.all_keys[0], self.all_keys[-1]+1) if i != self.args.test_sub]

        # # 'subject106.dat'
        # self.test_keys    = [self.args.test_sub]

        self.sub_ids_of_each_sub = {}

        # dataset
        self.data_x, self.data_y = self.load_all_the_data(self.data_path)

        # noise, gravitational force filtering
        if self.args.filtering:
            self.data_x = self.Sensor_data_noise_grav_filtering(self.data_x.set_index('sub_id').copy())

        # sliding window indexing
        self.train_slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "train")
        self.test_slidingwindows  = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "test")



    def load_all_the_data(self, data_path):
        file_list = os.listdir(data_path)
        
        df_dict = {}
        for file in file_list:
            if file == 'subject109.dat': continue
            
            sub_data = pd.read_table(os.path.join(data_path,file), header=None, sep='\s+')
            sub_data = sub_data.iloc[:,self.used_cols]
            sub_data.columns = self.col_names

            # if missing values, imputation
            sub_data = sub_data.interpolate(method='linear', limit_direction='both')
            sub = int(self.file_encoding[file])
            sub_data['sub_id'] = sub
            sub_data["sub"] = sub

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub)
            df_dict[self.file_encoding[file]] = sub_data   

        df_all = pd.concat(df_dict)

        # Downsampling - 99hz to 33hz
        df_all.reset_index(drop=True,inplace=True)
        index_list = list(np.arange(0,df_all.shape[0],3))
        df_all = df_all.iloc[index_list]

        df_all = df_all.set_index('sub_id')

        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder
        if self.selected_cols:
            df_all = df_all[self.selected_cols+["sub"]+["activity_id"]]
        else:
            df_all = df_all[self.col_names[1:]+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()

        return data_x, data_y
    
    def update_train_val_test_keys(self):
        """
        It should be called at the begin of each iteration
        it will update:
        1. train_window_index
        2. vali_window_index
        3. test_window_index
        it will also:
        normalize the data , because each iteration uses different training data
        calculate the weights of each class
        """
        self.test_keys =  self.LOCV_keys[self.index_of_cv]
        self.train_keys = [key for key in self.all_keys if key not in self.test_keys]
        # update the index_of_cv for the next iteration
        self.index_of_cv = self.index_of_cv + 1

        # Normalization of the data
        if self.datanorm_type is not None:
            train_vali_x = pd.DataFrame()
            for sub in self.train_keys:
                temp = self.data_x[self.data_x["sub"]==sub]
                train_vali_x = pd.concat([train_vali_x,temp])

            test_x = pd.DataFrame()
            for sub in self.test_keys:
                temp = self.data_x[self.data_x["sub"]==sub]
                test_x = pd.concat([test_x,temp])

            train_vali_x, test_x = self.normalization(train_vali_x, test_x)

            self.normalized_data_x = pd.concat([train_vali_x,test_x])
            self.normalized_data_x.sort_index(inplace=True)
        else:
            self.normalized_data_x = self.data_x.copy()

        # window index
        all_test_keys = self.test_keys.copy()

        # -----------------test_window_index---------------------
        # test_file_name = os.path.join(self.window_save_path,
        #                                 "{}_droptrans_{}_windowsize_{}_{}_test_ID_{}.pickle".format(self.data_name, 
        #                                                                                             self.drop_transition,
        #                                                                                             self.exp_mode,
        #                                                                                             self.windowsize, 
        #                                                                                             self.index_of_cv-1))
        # if os.path.exists(test_file_name):
        #     with open(test_file_name, 'rb') as handle:
        #         self.test_window_index = pickle.load(handle)
        # else:
        self.test_window_index = []
        for index, window in enumerate(self.test_slidingwindows):
            sub_id = window[0]
            if sub_id in all_test_keys:
                self.test_window_index.append(index)
            # with open(test_file_name, 'wb') as handle:
            #     pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # -----------------train_vali_window_index---------------------

        # train_file_name = os.path.join(self.window_save_path,
        #                                 "{}_droptrans_{}_windowsize_{}_{}_train_ID_{}.pickle".format(self.data_name, 
        #                                                                                             self.drop_transition,
        #                                                                                             self.exp_mode,
        #                                                                                             self.windowsize, 
        #                                                                                             self.index_of_cv-1))
        # if os.path.exists(train_file_name):
        #     with open(train_file_name, 'rb') as handle:
        #         train_vali_window_index = pickle.load(handle)
        # else:
        train_vali_window_index = []
        for index, window in enumerate(self.train_slidingwindows):
            sub_id = window[0]
            if sub_id not in all_test_keys:
                train_vali_window_index.append(index)
            # with open(train_file_name, 'wb') as handle:
            #     pickle.dump(train_vali_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        random.shuffle(train_vali_window_index)
        # train valid split
        self.train_window_index = train_vali_window_index[:int(self.train_vali_quote*len(train_vali_window_index))]
        self.vali_window_index = train_vali_window_index[int(self.train_vali_quote*len(train_vali_window_index)):]

    def normalization(self, train_vali, test): # test=None
        train_vali_sensors = train_vali.iloc[:,1:-1]
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors)
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors)
        train_vali_sensors = pd.concat([train_vali.iloc[:,0],train_vali_sensors,train_vali.iloc[:,-1]], axis=1)
        # if test is None:
        #     return train_vali_sensors
        # else:
        test_sensors  = test.iloc[:,1:-1]
        test_sensors  = self.normalizer.normalize(test_sensors)
        test_sensors  =  pd.concat([test.iloc[:,0],test_sensors,test.iloc[:,-1]], axis=1)
        return train_vali_sensors, test_sensors

    def Sensor_data_noise_grav_filtering(self, df):
        """
        df : sensor_1, sensor_2, sub
            index=sub_id
        """
        all_columns = list(df.columns)[:-1]
        #rest_columns = list(set(all_columns) - set(self.col_names))

        filtered_data = []
        for sub_id in df.index.unique():
            temp = df.loc[sub_id,all_columns]
            filtered_temp = pd.DataFrame()

            for col in temp.columns:
                t_signal=np.array(temp[col]) # copie the signal values in 1D numpy array

                if 'acc' in col: 
                    # the 2nd output DC_component is the gravity_acc
                    # The 3rd one is the body_component which in this case the body_acc
                    grav_acc, body_acc=components_selection_one_signal(t_signal,
                                                                    self.freq1,
                                                                    self.freq2,
                                                                    self.sampling_freq) # apply components selection

                    filtered_temp[col]=body_acc
                    # filtered_temp['grav_'+col]= grav_acc
                else: 
                    filtered_temp[col] = t_signal

            filtered_temp.index = temp.index
            filtered_data.append(filtered_temp)

        filtered_data = pd.concat(filtered_data)
        filtered_data = pd.concat([filtered_data, df.iloc[:,-1]], axis=1)

        return filtered_data.reset_index()

    def get_the_sliding_index(self, data_x, data_y, flag="train"):
        """
        Only store the indices of each window to avoid excessive memory usage 
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """
        data_y = data_y.reset_index()
        data_x["activity_id"] = data_y["activity_id"]

        # Identify groups of consecuitve rows from the same subject
        data_x['act_block'] = (data_x['sub_id'].shift(1) != data_x['sub_id']).astype(int).cumsum()

        # 50% overlap, 50% displacement
        # Compared to a higher overlap, 
        # lower computational cost, memory usage, processing time
        # less precise activity boundaries, less smooth transitions, lower temporal resolution 
        if flag == "train":
            displacement = int(0.5 * self.windowsize)

        # 90% overlap, 10% displacement
        elif flag == "test":
            displacement = int(0.1 * self.windowsize)

        window_index = []
        for index in data_x.act_block.unique():

            temp_df = data_x[data_x["act_block"]==index]
            assert len(temp_df["sub_id"].unique()) == 1
            sub_id = temp_df["sub_id"].unique()[0]
            start = temp_df.index[0]
            end   = start + self.windowsize

            while end <= temp_df.index[-1]+1:

                if temp_df.loc[start:end-1,"activity_id"].mode().loc[0] not in self.drop_activities:
                    window_index.append([sub_id, start, end])

                start = start + displacement
                end   = start + self.windowsize

        return window_index
    
    def Sensor_filter_acoording_to_pos_and_type(self, select, filter, all_col_names, filtertype):
        """
        select  (list or None): What location should be chosen
        filter  (list or None): whether all sensors can be filtered 
        all_col_names     (list)  : Columns currently available for Filtering
        """ 
        if select is not None:
            if filter is None:
                raise Exception('This dataset cannot be selected by sensor {}!'.format(filtertype))
            else:
                col_names = []
                for col in all_col_names:
                    selected = False
                    for one_select in select:
                        assert one_select in filter
                        if one_select in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                return col_names
        else:
            return None
    

class Dataset(object):
    def __init__(self, dataset, flag):
        self.flag = flag
        if self.flag == "train":
            self.slidingwindows = dataset.train_slidingwindows
            self.window_index = dataset.train_window_index
        elif self.flag == "vali":
            self.slidingwindows = dataset.train_slidingwindows
            self.window_index = dataset.vali_window_index
        else:
            self.slidingwindows = dataset.test_slidingwindows
            self.window_index = dataset.test_window_index

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}

        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y

        self.input_length = self.slidingwindows[0][2]-self.slidingwindows[0][1]
        self.channel_in = self.data_x.shape[1]-2

    def __getitem__(self, index):
        index = self.window_index[index]
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]

        sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values
        sample_x = np.expand_dims(sample_x,0)

        sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]

        return sample_x, sample_y

    def __len__(self):
        return len(self.window_index)

