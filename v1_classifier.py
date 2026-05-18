import os
import cv2 as cv
import scipy.io as sio
import numpy as np
from scipy import signal
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib
import time

"""
This program implements a binary classification using a SVM. The two tasks to classify are: right hand motor imagery and left hand
moto imagery.
The Dataset is available at BCI competition IV, dataset 2A.
To understand the images, look at the PDF explaining how each trial is executed by each of the partecipants
The final purpose is to build a BCI system, that could be used by disable people, to interact with the space around them
"""

# Grid search Cross-Validation
def tune_params(features, labels, cv):
    parameters = {'kernel':['rbf','poly'],
                  'C':[0.01,0.1,1,10,100],
                  'gamma':['scale',0.01,0.1,1,10,100],
                  'degree':[1, 2, 3, 4, 5]
                  }

    svc = svm.SVC() #max_iter=100000
    clf = GridSearchCV(svc, parameters, cv=cv)
    clf.fit(features, labels)
    return clf.best_params_


fs = 250 # Sampling Frequency

# Paths and images
IMAGES = os.path.join(os.path.dirname(__file__), 'images') # Path to /images directory
WINDOW_NAME = 'BCI TRIAL' # We will display each trial, one after the other
IMAGE_DIM = (1200,800)
FIXATION_IMG_PATH = os.path.join(IMAGES, "fix.png")
TASK_IMG_PATH     = os.path.join(IMAGES, "task.jpeg")
BREAK_IMG_PATH    = os.path.join(IMAGES, "break.png")

task_img   = cv.imread(TASK_IMG_PATH)
task_image = cv.resize(task_img, IMAGE_DIM)

# It's a binary classification, we exluded tongue and feet motor imagery
FROM_IMAGE_ID_TO_NAME = {"1": "left_hand", "0": "right_hand"}

# Load the processed data, 72 epochs for each of the condition

data_RH = sio.loadmat('data/MI_mani_A01Trealtime.mat')['RH'] # 22, 500, 72
data_LH = sio.loadmat('data/MI_mani_A01Trealtime.mat')['LH'] # 22, 500, 72
labelRH = np.zeros(data_RH.shape[2])                         #0 : right hand
labelLH = np.ones(data_LH.shape[2])                          #1 : left hand
labels_all = np.append(labelRH, labelLH)
data_all = np.concatenate((data_RH,data_LH), axis =  2) # 22 x 500 x 138
n = labels_all.shape[0]

# Shuffle among different classes
np.random.seed(3) # Set a fixed seed, the random function will produce always the same randomness
a = np.array(np.random.rand(n)) # Create a random array of the same length of labels_all
idx = np.argsort(a) # Returns the index that would sort the aray
data_all_sorted = data_all[:,:,idx]
labels_all_sorted = labels_all[idx]

num_train_epochs = int(n*0.7)
train_trials = data_all_sorted[:,:,0:num_train_epochs] # 22 500 96
train_labels = labels_all_sorted[0:num_train_epochs]
test_trials = data_all_sorted[:,:,num_train_epochs:]   #22 500 42
test_labels = labels_all_sorted[num_train_epochs:]

# TRAIN MODEL
train_features = np.zeros((22,num_train_epochs))
for idx_chan in range(22):
    train_trials_at_chan = np.squeeze(train_trials[idx_chan,:,:]) # 500 x 96, extract data from a fixed channel

    f_iter       = np.zeros((num_train_epochs,int((fs/2)+1))) # 96 x 126
    Pxx_den_iter = np.zeros((num_train_epochs,int((fs/2)+1))) # 96 x 126
    Pxx_mean     = np.zeros(num_train_epochs) # 96

    for i in range(num_train_epochs):
        f_iter[i,:], Pxx_den_iter[i,:] = signal.welch(train_trials_at_chan[:,i], 250, nperseg=250)
        Pxx_mean[i] = np.mean(Pxx_den_iter[i,8:15]) #Mean alpha band

    train_features[idx_chan,:] = Pxx_mean # Feature Alpha band frequency built 22 x 96


# TRAIN MODEL SVM
train_features = train_features.T # 96 x 22
# K-Fold-Validation k=5
skf = StratifiedKFold(n_splits=5)
best_params = tune_params(train_features, train_labels, cv=skf)

accuracies_train    = []
accuracies_val      = []
accuracies_test     = []
conf_matrices_train = []
conf_matrices_val   = []
clf_best      = None
accuracy_best = None

# Per 5 iterazioni (k=5) skf.split crea degli indici di train e val diversi (80-20%), seleziono le epoche in quegli indici
for idx_train, idx_val in skf.split(train_features, train_labels):

    # Create train data and labels, and validation data and labels
    train_data, val_data = train_features[idx_train], train_features[idx_val]
    y_train, y_val = train_labels[idx_train], train_labels[idx_val]

    # Creo modello
    clf = svm.SVC(kernel=best_params['kernel'],
                  C=best_params['C'],
                  gamma=best_params['gamma'],
                  degree=best_params['degree'],
                  # max_iter=100
                  )

    # Train the model using the training sets
    clf.fit(train_data, y_train)
    y_pred_train = clf.predict(train_data) # Labels del train predette

    # PERFOMANCE: Train
    conf_matrix_train = confusion_matrix(y_train, y_pred_train) # Example: [ [38,2],[4,40] ]
    conf_matrices_train.append(conf_matrix_train) # Aggiungo questa matrice a una lista
    accuracies_train.append(metrics.accuracy_score(y_train, y_pred_train)) # Calcolo l'accuracy pred/ground truth e la aggiungo a una lista

    # Predict the response for test dataset
    # VALIDATION
    y_pred = clf.predict(val_data)
    conf_matrix_val = confusion_matrix(y_val, y_pred)
    conf_matrices_val.append(conf_matrix_val)
    accuracies_val.append(metrics.accuracy_score(y_val, y_pred))
    acc = metrics.accuracy_score(y_val, y_pred) # Salvo l'accuracy dell'attuale fold

    # Statements per salvare il modello associato all'accuracy val migliore
    if clf_best is None:
        clf_best = clf
        accuracy_best = acc
    else:
        if acc > np.array(accuracy_best):
            accuracy_best = acc
            clf_best = clf

joblib.dump(clf_best, 'svm_model.joblib') # Questo è il modello migliore

#ACCURACY VALIDATION
print(f'Validation accuracy: {np.mean(accuracies_val)*100}')




#TEST_MODEL
trials = np.array([10,12,15]) # Per semplicità seleziono solo alcune epoche di test
STAGE1_S = 2+1 # Aspetta 3 secondi prima di fare il display di queste immagini
STAGE2_S = 2
STAGE3_S = 2

for trial in trials:

    trial_to_display = np.zeros((22,500)) # Vorre fare il display di un'epoca alla volta
    P_test = np.zeros(22)                 # 22 valori di potenza spettrale in alpha
    j = 0
    useful_data = np.squeeze(test_trials[:,:,trial]) # Fisso un'epoca ---> 22 x 500

    test_feature  = None
    state         = None
    elapsed       = None

    for id_sample in range(750 + useful_data.shape[1] + 200): # Per ognuno dei 1750 sample (3 sec di fixation, 2 sec di immaginato, 2 sec di post)

        #VIDEO
        if id_sample < fs*3: # se il sample è < del 750-esimo

            #FIXATION for 3 seconds
            if state != "fixation":
                s = time.time() # Per ogni sample ottieni il tempo
                fixation_img = cv.imread(FIXATION_IMG_PATH)
                fixation_img = cv.resize(fixation_img, IMAGE_DIM)
                cv.imshow(WINDOW_NAME, fixation_img)
                e = time.time()
                elapsed = (e - s) * 1000

            state = "fixation"

        if fs*3 <= id_sample < fs*5:
            # TASK for 2 seconds, BREAK for 1 second
            if state != "task":
                t = int(STAGE1_S * 1000 - elapsed)
                cv.waitKey(t)
                s = time.time()
                image_id = int(test_labels[trial]) #0 RH, 1 LH
                image_path = os.path.join(IMAGES, f"{image_id}.png")
                image = cv.imread(image_path)
                image = cv.resize(image, IMAGE_DIM)
                cv.imshow(WINDOW_NAME, image)
                e = time.time()
                elapsed = (e - s) * 1000

            state = "task"
            trial_to_display[:,j] = np.squeeze(useful_data[:,id_sample-750])
            j = j+1 # Fino a j=500, avrò riempito il trial to display

        if id_sample >= fs*5 and test_feature is None: # dal sample 1250 in avanti faccio la classificazione
            s = time.time()
            for id_chan in range(22):
                f_iter[id_chan,:], Pxx_den_iter[id_chan,:] = signal.welch(trial_to_display[id_chan,:], 250, nperseg=250)
                P_test[id_chan] = np.mean(Pxx_den_iter[id_chan,8:15])

            test_feature = P_test
            # TEST del modello
            y = clf_best.predict(test_feature.reshape(1,-1))
            e = time.time()
            response_time = (e - s) * 1000 # ms

            if test_labels[trial] == 0:
                print('True class:')
                print('RIGHT HAND')
                print('%%%%%%%%%%%%%%%%%')
                true_im = "R"
            else:
                print('True class:')
                print('LEFT HAND')
                print('%%%%%%%%%%%%%%%%%')
                true_im = "L"

            if y ==0:
                y_name = 'RIGHT HAND'
                est_im = "R"
                print('Predicted class:')
                print('RIGHT HAND')
            else:
                y_name = 'LEFT HAND'
                est_im = "L"
                print('Predicted class:')
                print('LEFT HAND')


        if id_sample >= fs*5:
            if state != "break":
                cv.waitKey(int(STAGE2_S * 1000 - elapsed))
                s = time.time()
                break_image_path_2 = os.path.join(IMAGES, f"break{true_im}{est_im}.png")
                break_image = cv.imread(break_image_path_2)
                break_image = cv.resize(break_image, IMAGE_DIM)
                break_image = cv.putText(
                    break_image, f"Response time: {response_time:.2f} ms", (50, 100), cv.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 0), 2, cv.LINE_AA)
                cv.imshow(WINDOW_NAME, break_image)
                e = time.time()
                elapsed = (e - s) * 1000

            state = "break"

        if id_sample == useful_data.shape[1] - 1:
            cv.waitKey(int(STAGE3_S * 1000 - elapsed))

#out.write(WINDOW_NAME)































