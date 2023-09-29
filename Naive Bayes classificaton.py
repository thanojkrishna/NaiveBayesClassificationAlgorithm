import pandas as pd #pip install pandas , pip install openpyxl
import numpy as np 

###-------------------- User defined Funtions -------------------------***

# Creat UDF for binning numerical columns
def udf_binning(df, column_name, bin_size,custom_threshold=None):
    # Calculate min and max values
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    # Create bin boundaries
    if custom_threshold is not None and custom_threshold < max_value:
        bin_boundaries = list(range(min_value, custom_threshold, bin_size))
        bin_boundaries.append(max_value+1)
    else:
        bin_boundaries = list(range(min_value, max_value+1, bin_size))
        bin_boundaries.append(max_value+1)
    # Apply binning to create the binning column
    binning_column_name = f'{column_name}_Bins'
    df[binning_column_name] = pd.cut(df[column_name], bins=bin_boundaries, right=False, labels=False)
    #print(df)

#create UDF to handle missing values
def replace_unknown_with_mode(df, column_name):
    # Calculate the mode of the column (excluding '?' values)
    mode = df[df[column_name] != '?'][column_name].mode()[0]  
    # Replace '?' values with the mode
    df[column_name] = df[column_name].replace('?', mode)

def calculate_prior_probability(X_train,Y_train):
    for feature in features:
        likelihoods[feature] = {}
        for feat_val in np.unique(X_train[feature]):
            for outcome in np.unique(Y_train):
                likelihoods[feature].update({str(feat_val)+'_'+outcome:0})
                class_probability.update({outcome: 0})
            
""" P(Class Label) - Prior Class Probability """
def calculate_class_probability(Y_train):
    for outcome in np.unique(Y_train):
        outcome_count = sum(Y_train == outcome)
        class_probability[outcome] = outcome_count / train_size

""" P(features|Class label) - Likelihood """
def calculate_likelihood_probability(X_train,Y_train):
    for feature in features:
        for outcome in np.unique(Y_train):
            alpha=1
            outcome_count = sum(Y_train == outcome)+ (len(X_train[feature].unique()) * alpha)
            feature_likelihood=X_train[feature][Y_train[Y_train == outcome].index.values.tolist()].value_counts()
            #print(feature_likelihood)
            for feat_val, count in feature_likelihood.items():
                smoothed_count = count + alpha
                likelihoods[feature][str(feat_val) + '_' + outcome] = smoothed_count/outcome_count
                
#predict function
def predict(X_test,class_probability, likelihoods):
    predictions = []
    for index, row in X_test.iterrows():
        #print(index,row)
        max_class_probability = float("-inf")  
        predicted_class = None
        for target_class, class_probabilities in class_probability.items():
            posterior_probability=class_probabilities
            #print(target_class,class_probabilities)
            for feature in X_test.columns:
                feature_value = row[feature]
                likelihood_key = str(feature_value) + '_' + target_class
                #print(likelihood_key)
                if likelihood_key in likelihoods[feature]:
                    posterior_probability *= likelihoods[feature][likelihood_key]
            if posterior_probability > max_class_probability:
                max_class_probability = posterior_probability
                predicted_class = target_class
                #print(predicted_class)
        predictions.append(predicted_class)
    return predictions

#apply_laplace_smoothening to handle 0 likelihood records
def apply_laplace_smoothing(likelihoods,X_train,Y_train,alpha=1):
    for feature, conditional_probabilities in likelihoods.items():
        for key, probability in conditional_probabilities.items():
            alpha=1
            extracted_class_label = key.split('_')[-1]
            extracted_attribute_label = key.split('_')[0]
            if probability == 0 :
                outcome_count = sum(Y_train == extracted_class_label)+ (len(X_train[feature].unique()) * alpha)
                #print(extracted_class_label,extracted_attribute_label, feature, key, outcome_count,probability,(probability + alpha) /outcome_count)
                # Apply Laplace smoothing to entries with 0 probability
                likelihoods[feature][str(extracted_attribute_label) + '_' + extracted_class_label] = (probability + alpha) /outcome_count

#write UDF for calculating accuracy , precision,recall , F1 score and Confusion matrix
#calculate accuracy
def calculate_accuracy(actual, predicted,positive_label,negative_label):
    true_positives = sum(1 for a, p in zip(actual, predicted) if a == p == positive_label)
    true_negatives = sum(1 for a, p in zip(actual, predicted) if a == p == negative_label)
    false_positives = sum(1 for a, p in zip(actual, predicted) if a == negative_label and p == positive_label)
    false_negatives = sum(1 for a, p in zip(actual, predicted) if a == positive_label and p == negative_label)

    total = true_positives+true_negatives+false_positives+false_negatives
    accuracy = (true_positives+true_negatives) / total
    return accuracy

#calculate precision
def calculate_precision(actual, predicted, positive_label,negative_label):
    true_positives = sum(1 for a, p in zip(actual, predicted) if a == p == positive_label)
    false_positives = sum(1 for a, p in zip(actual, predicted) if a == negative_label and p == positive_label)
    precision = true_positives / (true_positives + false_positives)   
    return precision

#calculate recall
def calculate_recall(actual, predicted, positive_label,negative_label):
    true_positives = sum(1 for a, p in zip(actual, predicted) if a == p == positive_label)
    false_negatives = sum(1 for a, p in zip(actual, predicted) if a == positive_label and p == negative_label)
    recall = true_positives / (true_positives + false_negatives)
    return recall

#calculate F1 score
def calculate_f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall) 
    return f1_score

#calculate confusion matrix
def calculate_confusion_matrix(actual, predicted, positive_label, negative_label):
    true_positives = sum(1 for a, p in zip(actual, predicted) if a == p == positive_label)
    true_negatives = sum(1 for a, p in zip(actual, predicted) if a == p == negative_label)
    false_positives = sum(1 for a, p in zip(actual, predicted) if a == negative_label and p == positive_label)
    false_negatives = sum(1 for a, p in zip(actual, predicted) if a == positive_label and p == negative_label)
    confusion_matrix = {
        'True Positives': true_positives,
        'True Negatives': true_negatives,
        'False Positives': false_positives,
        'False Negatives': false_negatives
    }
    return confusion_matrix

# Evaluate on Testing Data
def Naive_Bayes(X_train,Y_train,X_test,Y_test):
    #calculate probabilities
    calculate_prior_probability(X_train,Y_train)
    calculate_class_probability(Y_train)
    calculate_likelihood_probability(X_train,Y_train)
    apply_laplace_smoothing(likelihoods,X_train,Y_train,alpha=1)

    actual_labels=Y_test.tolist()
    positive_label='>50K'
    negative_label='<=50K'
    # Calculate predictions 
    predictions = predict(X_test, class_probability, likelihoods)
    #  accuracy
    accuracy = calculate_accuracy(actual_labels,predictions,positive_label,negative_label)
    print("Accuracy:", accuracy)
    #  precision
    precision = calculate_precision(Y_test, predictions,positive_label,negative_label)
    print("Precision:", precision)
    #  recall
    #recall
    recall = calculate_recall(actual_labels, predictions, positive_label,negative_label)
    print("Recall:", recall)
    #  F1-score
    f1 = calculate_f1_score(precision, recall)
    print("F1 Score:", f1)
    #confusion matrix
    confusion = calculate_confusion_matrix(actual_labels, predictions, positive_label, negative_label)
    print("Confusion Matrix:")
    for key, value in confusion.items():
        print(key + ": ", value)
        
###--------------------------------------------- Adult data file pre processing
#read data from .data files
adult_data = pd.read_csv(r'C:\Users\patron\Desktop\Masters\Fall 2023\CSC 869 Data Mining\Mini Project\adult dataset\adult.data', delimiter=',', header=None) 
#set column headers 
adult_data.columns=['Age', 'Work_class', 'Final_Weight', 'Education', 'Education_Number', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_Per_Week', 'Native_Country', 'Income']
#map datatypes as per the data dictionary
adult_data_types={'Age':int, 'Work_class':'category', 'Final_Weight':int, 'Education':'category', 'Education_Number':int, 'Marital_Status':'category', 'Occupation':'category', 'Relationship':'category', 'Race':'category', 'Sex':'category', 'Capital_Gain':int, 'Capital_Loss':int, 'Hours_Per_Week':int, 'Native_Country':'category', 'Income':'category'}
adult_data=adult_data.astype(adult_data_types)
#Remove trailing and ending spaces for values
adult_data = adult_data.map(lambda x: x.strip() if isinstance(x, str) else x)

#From EDA , below bin sizes are observed to follow Gaussian Distirbution
udf_binning(adult_data,"Age",5)
udf_binning(adult_data,"Education_Number",4)
udf_binning(adult_data,"Hours_Per_Week",10)
udf_binning(adult_data,"Capital_Gain",2000,34001)
udf_binning(adult_data,"Capital_Loss",50)
udf_binning(adult_data,"Final_Weight",25000,500000)

# Handle missing values with Mode Imputation

replace_unknown_with_mode(adult_data,"Work_class")
replace_unknown_with_mode(adult_data,"Occupation")
replace_unknown_with_mode(adult_data,"Native_Country")
#drop columns for native country
#adult_data = adult_data[adult_data['Native_Country'] != '?' ]

#Keep only selected field for Naive Bayes Implementation
alias_mapping = {
    'Age_Bins': 'Age',
    'Education_Number_Bins': 'Education',
    'Hours_Per_Week_Bins':'Hours_Per_Week',
    'Capital_Gain_Bins':'Capital_Gain',
    'Capital_Loss_Bins':'Capital_Loss',
    'Final_Weight_Bins':'Final_Weight'
}
adult_data_final=adult_data[['Work_class',  'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native_Country',  'Age_Bins', 'Education_Number_Bins', 'Hours_Per_Week_Bins', 'Capital_Gain_Bins', 'Capital_Loss_Bins', 'Final_Weight_Bins','Income']].rename(columns=alias_mapping)

###--------------------------------------------- Naive Bayes Model Implementation
## Prepare Training, dev and test dataframes
# Create Train (70%), Dev (20%), Test (10%) sets
X=adult_data_final.drop([adult_data_final.columns[-1]], axis = 1) #only Features
Y=adult_data_final[adult_data_final.columns[-1]] # target variable - class label
# Set the seed for reproducibility
np.random.seed(42)
# dataframe randomization
indices = np.arange(len(X))
np.random.shuffle(indices)
# Train_dev_test splitting ratios
train_ratio = 0.7
dev_ratio = 0.2
test_ratio = 0.1
# sizes of each split
total_size = len(X)
train_size = int(train_ratio * total_size)
dev_size = int(dev_ratio * total_size)
train_indices = indices[:train_size]
dev_indices = indices[train_size:(train_size + dev_size)]
test_indices = indices[(train_size + dev_size):]

# Split the data and labels based on the selected indices
X_train = X.iloc[train_indices]
Y_train = Y.iloc[train_indices]

X_dev = X.iloc[dev_indices]
Y_dev = Y.iloc[dev_indices]

X_test = X.iloc[test_indices]
Y_test = Y.iloc[test_indices]

## create,Instaniate and map probabilities for likelyhoob and prior dictionaries  based on Training set
train_size = X_train.shape[0]
num_feats = X_train.shape[1]
features = list(X_train.columns)
#print(features)
#print(train_size)
#print(num_feats)
likelihoods = {}
class_probability = {}


###-----------------------------------Invoke Naive Bayes method to run & test the model

Naive_Bayes(X_train,Y_train,X_test,Y_test)

###-----------------------------------K Fold size 10 - validation

# K- fold cross validation
# # of folds
k = 10
fold_size = len(X) // k
accuracy_scores = []
# randomize dataframe indexes
indexes = np.arange(len(X))
np.random.shuffle(indices)

# Iterate through the folds
for i in range(k):
     # # randomize dataframe indexes
    shuffled_indices = np.arange(len(X))
    np.random.shuffle(shuffled_indices)
    # prepare training and testing subsets
    test_indices = shuffled_indices[i * fold_size: (i + 1) * fold_size]
    train_indices = [j for j in range(len(X)) if j not in test_indices]
    X_train = X.iloc[train_indices]
    Y_train = Y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    Y_test = Y.iloc[test_indices]
    positive_label='>50K'
    negative_label='<=50K'
    #calculate probabilities
    calculate_prior_probability(X_train,Y_train)
    calculate_class_probability(Y_train)
    calculate_likelihood_probability(X_train,Y_train)
    # Calculate predictions
    predictions = predict(X_test, class_probability, likelihoods)
    # Evaluate accuracy for this fold
    actual_labels=Y_test.tolist()
    accuracy = calculate_accuracy(actual_labels,predictions,positive_label,negative_label)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy over all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("K-Fold Average Accuracy:", average_accuracy)





