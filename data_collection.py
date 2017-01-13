import numpy as np
import pandas as pd
import cPickle
import pandas.io.sql as psql
from pandas_datareader import data as dt
import os



def get_ext_data(share_list, cols_list = ['volume'], start = '2000-01-01', source='yahoo'):
    # get sharedata using pandas data_reader. Default source: yahoo market data
    if type(share_list) != list:
        raise TypeError("share input must be in a list")
    try:
        price_data = dt.DataReader(share_list, source, start)
        price_data = price_data.to_frame().reset_index()
        try:
            price_data.drop(['Adj Close'], inplace = True, axis = 1)
        except Exception:
            pass
    except Exception as e:
        # add retry with different source?
        print (e)
    else:
        # set column names and order
        price_data.columns = ['timestamp', 'share', 'open', 'high', 'low', 'close', 'volume']
        price_data.sort_values(['share', 'timestamp'], inplace=True)
        # select columns to keep while preserving column order
        cols_list = [col for col in price_data.columns if col in cols_list]
        return price_data[cols_list]


class dataCollector(object):
    def __init__(self, instruments=None, series=None, conn=None):
        self.conn = conn
        self.series_to_include = series
        self.instruments = list(instruments)
        # minimum set of columns needed to operate + optional ones
        self.all_columns = list(set(['timestamp', 'share', 'open', 'close'] + self.series_to_include.keys()))

    def get_data(self):

        try:
            if not os.path.exists("./data"):
                os.makedirs("./data")
            with open("data/stockdata.data", 'rb') as f:
                dat = cPickle.load(f)
            # TODO: handle when some keys are missing
            dat = dat.loc[dat['share'].isin(self.instruments)][self.all_columns]
            print("data loaded from disk")

        except Exception as e:
            print(e)
            print("getting data externally...")
            dat = get_ext_data(self.instruments, self.all_columns)

            #with open("data/stockdata.data", 'wb') as f:
            #    cPickle.dump(dat, f)

        self.dat_raw = dat


class dataConstructor(dataCollector):
    def __init__(self, instruments, series, days_forward, days_back, conn):

        # initialize dataCollector class
        super(dataConstructor, self).__init__(instruments, series, conn)

        self.days_forward = days_forward
        self.days_back = days_back
        self.data_raw = None
        self.xlist = None
        self.ylist = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None

    def __generate_target(self, dat, rtype="open"):

        end = dat["close"].shift(-self.days_forward)
        if  rtype == "open":
            start = dat["open"].shift(-1)
        else:
            start = dat["close"]
        # keep for future use
        self.returns = (end - start) / start

        return np.where(self.returns > 0, 1, 0)

    def __generate_data_table(self, dat):

        # sort and generate target
        dat = dat.sort_values(["timestamp"]).set_index("timestamp")
        dat["target"] = self.__generate_target(dat)
        for s in self.series_to_include.keys():
            # OUTER LOOP: prepare start columns
            # for each timeseries (open high low close volume) create a column
            # x00, where x is the first letter in the series name and populate it with
            # the corresponding transformation. "return" will transform the series to
            # daily percentage change, "first" will normalize by dividing with the first
            # value in the dindow defined by "days_back". "zscore" will transform the given series
            # to z-scores
            value_at_start = 1
            if "return" in self.series_to_include[s]:
                # transform to returns
                dat["{}00".format(s[0])] = dat[[s]].pct_change(1)
            elif "first" in self.series_to_include[s]:
                # divide by the first value in window
                dat["{}00".format(s[0])] = dat[[s]]
                value_at_start = dat["{}00".format(s[0])].shift(self.days_back - 1)
            else:
                # do nothing (just create x0)
                dat["{}00".format(s[0])] = dat[[s]]
            # add a z-score transform of the values in the x0 column if desired.
            if "zscore" in self.series_to_include[s]:
                cl = dat["{}00".format(s[0])]
                dat["{}00".format(s[0])] = (cl - cl.mean()) / cl.std(ddof=0)
            # INNER LOOP: generate columns xi for the value in x0 shifted i steps
            # if not "first" then value_at_start = 1
            for i in range(self.days_back):
                dat["{}{}".format(s[0], i)] = dat["{}00".format(s[0])].shift(i) / value_at_start
            # drop tmp reference column x00
            dat.drop("{}00".format(s[0]), axis=1, inplace=True)

        # drop generated nans
        dat.dropna(inplace=True)
        # drop unnecessary columns
        dat.drop(self.series_to_include.keys() + ["share", "open", "close"], axis=1, inplace=True)
        return dat

    def __data_to_arrays(self, dat, flat):
        # return data in 1d and 2d form
        # generate array of targets (1 hot encoding)
        dat = dat.drop(["share", "timestamp"], axis=1)

        ylist = pd.get_dummies(dat.pop("target")).as_matrix()  # 1d targets
        xlist = dat.as_matrix()  # 1d

        if flat == False:
            xlist = np.array(map(lambda x: np.reshape(x, (len(self.series_to_include), -1)), xlist))  # 2d
        return xlist, ylist

    def generate_datasets(self, flat=True, test_split=None):

        if self.data_raw == None:
            self.get_data()

        # Create the dataset ------------------------------
        # generate variable table on subgroups by share
        dat = self.dat_raw.groupby("share").apply(lambda x: self.__generate_data_table(x))
        # -------------------------------------------------
        dat.reset_index(inplace=True)
        # dat.drop(["share"], axis = 1, inplace=True)

        self.timespan = dat.timestamp
        # generate final data arrays ----------------------
        if test_split != None:

            self.test_split = test_split

            train = dat[dat.timestamp < test_split]
            tests = dat[dat.timestamp >= test_split]

            self.xtrain, self.ytrain = self.__data_to_arrays(train, flat)
            self.xtest, self.ytest = self.__data_to_arrays(tests, flat)
            
	    #TODO: rethink this, maybe. It removes data points so only every nth record is
            # is kept if days_forward = n. I.e, if predicting one week forward it only trains on weekly data.
            self.xtrain = self.xtrain[::self.days_forward]
            self.ytrain = self.ytrain[::self.days_forward]

            print(
                "\nGenerated arrays with {} train observations and {} test observations \nstart date: {}, end date: {}".format(
                    len(self.xtrain), len(self.xtest), min(self.timespan), max(self.timespan)))

            print("\ntrainset shapes: x-arrays: {}, y-arrays: {}".format(self.xtrain.shape, self.ytrain.shape))

        else:
            self.xlist, self.ylist = self.__data_to_arrays(dat, flat)

            print("\nGenerated arrays with {} observations \nstart date: {}, end date: {}".format(
                len(self.xlist), min(self.timespan), max(self.timespan)))

            print("\ndataset shapes: x-arrays: {}, y-arrays: {}".format(self.xlist.shape, self.ylist.shape))

