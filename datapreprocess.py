import os,sys
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import helper as ut
import numpy as np
from random import seed, shuffle
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

class Datapreprocess:


    def read_data(self,load_data_size=None):

        """
            if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
            if it is a number, say 10000, then we will return randomly selected 10K examples
        """

        attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'] # all attributes
        int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'] # attributes with integer values -- the rest are categorical
        sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
        attrs_to_ignore = ['sex', 'race' ,'fnlwgt'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
        attrs_for_classification = set(attrs) - set(attrs_to_ignore)

        # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
        data_files = ["adult.data", "adult.test"]

        X = []
        y = []
        x_control = {}

        attrs_to_vals = {} # will store the values for each attribute for all users
        for k in attrs:
            if k in sensitive_attrs:
                x_control[k] = []
            elif k in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[k] = []

        for f in data_files:

            for line in open(f):
                line = line.strip()
                if line == "": continue # skip empty lines
                line = line.split(", ")
                if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
                    continue

                class_label = line[-1]
                if class_label in ["<=50K.", "<=50K"]:
                    class_label = -1
                elif class_label in [">50K.", ">50K"]:
                    class_label = +1
                else:
                    raise Exception("Invalid class label value")

                y.append(class_label)


                for i in range(0,len(line)-1):
                    attr_name = attrs[i]
                    attr_val = line[i]
                    # reducing dimensionality of some very sparse features
                    if attr_name == "native_country":
                        if attr_val!="United-States":
                            attr_val = "Non-United-Stated"
                    elif attr_name == "education":
                        if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                            attr_val = "prim-middle-school"
                        elif attr_val in ["9th", "10th", "11th", "12th"]:
                            attr_val = "high-school"

                    if attr_name in sensitive_attrs:
                        x_control[attr_name].append(attr_val)
                    elif attr_name in attrs_to_ignore:
                        pass
                    else:
                        attrs_to_vals[attr_name].append(attr_val)

        def labelencoding(d): # discretize the string attributes
            for attr_name, attr_vals in d.items():
                if attr_name in int_attrs: continue
                uniq_vals = sorted(list(set(attr_vals))) # get unique values

                # compute integer codes for the unique values
                val_dict = {}
                for i in range(0,len(uniq_vals)):
                    val_dict[uniq_vals[i]] = i

                # replace the values with their integer encoding
                for i in range(0,len(attr_vals)):
                    attr_vals[i] = val_dict[attr_vals[i]]
                d[attr_name] = attr_vals


        # convert the discrete values to their integer representations
        labelencoding(x_control)
        labelencoding(attrs_to_vals)


        # if the integer vals are not binary, we need to get one-hot encoding for them
        for attr_name in attrs_for_classification:
            attr_vals = attrs_to_vals[attr_name]
            if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
                X.append(attr_vals)

            else:
                attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
                for inner_col in attr_vals.T:
                    X.append(inner_col)


        # convert to numpy arrays for easy handline
        X = np.array(X, dtype=float).T
        y = np.array(y, dtype = float)
        for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

        # shuffle the data
        perm = range(0,len(y)) # shuffle the data before creating each fold
        perm = list(perm)
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        for k in x_control.keys():
            x_control[k] = x_control[k][perm]
        # add intercept to X before applying the linear classifier

        m, n = X.shape
        intercept = np.ones(m).reshape(m, 1)  # the constant b
        X = np.concatenate((intercept, X), axis = 1)
        # see if we need to subsample the data
        if load_data_size is not None:
           # print ("Loading only %d examples from the data", load_data_size)
            X = X[:load_data_size]
            y = y[:load_data_size]
            for k in x_control.keys():
                x_control[k] = x_control[k][:load_data_size]

        return X, y, x_control


    def train_test_split(self,x_all, y_all, x_control_all, train_fold_size):

        split_point = int(round(float(x_all.shape[0]) * train_fold_size))
        x_train = x_all[:split_point]
        x_test = x_all[split_point:]
        y_train = y_all[:split_point]
        y_test = y_all[split_point:]
        x_control_train = {}
        x_control_test = {}
        for k in x_control_all.keys():
            x_control_train[k] = x_control_all[k][:split_point]
            x_control_test[k] = x_control_all[k][split_point:]

        return x_train, y_train, x_control_train, x_test, y_test, x_control_test