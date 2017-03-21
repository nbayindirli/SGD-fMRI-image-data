import scipy.io as sio
import numpy as np
import math

starplus = sio.loadmat("data-starplus-04847-v7.mat")

###########################################
metadata = starplus['meta'][0,0]
# meta.study gives the name of the fMRI study
#
# meta.subject gives the identifier for the human subject
#
# meta.ntrials gives the number of trials in this dataset
#
# meta.nsnapshots gives the total number of images in the dataset
#
# meta.nvoxels gives the number of voxels (3D pixels) in each image
#
# meta.dimx gives the maximum x coordinate in the brain image. The minimum x coordinate is x=1. meta.dimy and
#       meta.dimzgive the same information for the y and z coordinates.
#
# meta.colToCoord(v,:) gives the geometric coordinate (x,y,z) of the voxel corresponding to column v in the data
#
# meta.coordToCol(x,y,z) gives the column index (within the data) of the voxel whose coordinate is (x,y,z)
#
# meta.rois is a struct array defining a few dozen anatomically defined Regions Of Interest (ROIs) in the brain.
#       Each element of the struct array defines on of the ROIs, and has three fields: "name" which gives the ROI name
#       (e.g., 'LIFG'), "coords" which gives the xyz coordinates of each voxel in that ROI, and "columns" which gives
#       the column index of each voxel in that ROI.
#
# meta.colToROI{v} gives the ROI of the voxel corresponding to column v in the data.
study      = metadata['study']
subject    = metadata['subject']
ntrials    = metadata['ntrials'][0][0]
nsnapshots = metadata['nsnapshots'][0][0]
dimx       = metadata['dimx'][0][0]
colToCoord = metadata['colToCoord']
coordToCol = metadata['coordToCol']
rois       = metadata['rois']
colToROI   = metadata['colToROI']
###########################################

###########################################
info = starplus['info'][0]
cond = info['cond']
# info: This variable defines the experiment in terms of a sequence of 'trials'. 'info' is a 1x54 struct array,
#       describing the 54 time intervals, or trials. Most of these time intervals correspond to trials during which the
#       subject views a single picture and a single sentence, and presses a button to indicate whether the sentence
#       correctly describes the picture. Other time intervals correspond to rest periods. The relevant fields of info
#       are illustrated in the following example:
#
# info(18) mint: 894 maxt: 948 cond: 2 firstStimulus: 'P' sentence: ''It is true that the star is below the plus.''
#       sentenceRel: 'below' sentenceSym1: 'star' sentenceSym2: 'plus' img: sap actionAnswer: 0 actionRT: 3613
#
# info.mint gives the time of the first image in the interval (the minimum time)
#
# info.maxt gives the time of the last image in the interval (the maximum time)
#
# info.cond has possible values 0,1,2,3. Cond=0 indicates the data in this segment should be ignored. Cond=1 indicates
#       the segment is a rest, or fixation interval. Cond=2 indicates the interval is a sentence/picture trial in which
#       the sentence is not negated. Cond=3 indicates the interval is a sentence/picture trial in which the sentence is
#       negated.
#
# info.firstStimulus: is either 'P' or 'S' indicating whether this trail was obtained during the session is which
#       Pictures were presented before sentences, or during the session in which Sentences were presented before
#       pictures. The first 27 trials have firstStimulus='P', the remained have firstStimulus='S'. Note this value is
#       present even for trials that are rest trials. You can pick out the trials for which sentences and pictures
#       were presented by selecting just the trials trials with info.cond=2 or info.cond=3.
#
# info.sentence gives the sentence presented during this trial. If none, the value is '' (the empty string).
#       The fields info.sentenceSym1, info.sentenceSym2, and info.sentenceRel describe the two symbols mentioned in the
#       sentence, and the relation between them.
#
# info.img describes the image presented during this trial. For example, 'sap' means the image contained a 'star above
#       plus'. Each image has two tokens, where one is above the other. The possible tokens are star (s), plus (p),
#       and dollar (d).
#
# info.actionAnswer: has values -1 or 0. A value of 0 indicates the subject is expected to press the answer button
#       during this trial (either the 'yes' or 'no' button to indicate whether the sentence correctly describes the
#       picture). A value of -1 indicates it is inappropriate for the subject to press the answer button during this
#       trial (i.e., it is a rest, or fixation trial).
#
# info.actionRT: gives the reaction time of the subject, measured as the time at which they pressed the answer button,
#       minus the time at which the second stimulus was presented. Time is in milliseconds. If the subject did not
#       press the button at all, the value is 0.
###########################################

###########################################
data = starplus['data']
# data: This variable contains the raw observed data. The fMRI data is a sequence of images collected over time, one
#       image each 500 msec. The data structure 'data' is a [54x1] cell array, with one cell per 'trial' in the
#       experiment. Each element in this cell array is an NxV array of observed fMRI activations. The element
#       data{x}(t,v) gives the fMRI observation at voxel v, at time t within trial x. Here t is the within-trial time,
#       ranging from 1 to info(x).len. The full image at time t within trial x is given by data{x}(t,:).
# Note the absolute time for the first image within trial x is given by info(x).mint.
###########################################


def hinge_loss(x, y, w, lmda):
    # TODO: Compute (regularized) Hinge Loss
    loss = 0                                # hinge loss term
    reg = 0                                 # regularization term

    for xl, yl in zip(x, y):                # merges x and y for ease of iteration
        xi = yl * np.dot(w, xl)             # calculate slack variable
        loss += max(0, 1 - xi)              # calculate hinge loss

    return loss + lmda * np.dot(w, w)


def sgd_hinge(x, y, max_iter, learning_rate, lmda):
    W = np.zeros(x.shape[1])
    # TODO: implement stochastic (sub) gradient descent with the hinge loss function
    condition = False
    num_steps = 0
    np.random.seed(1)

    while not condition and num_steps < max_iter:
        trial = np.random.randint(0, len(y))

        old_loss = hinge_loss(x, y, W, lmda)

        if not cond[trial] == 0:
            xi = y[trial] * np.dot(W, x[trial, :])                 # calculate slack variable
            gradient = -y[trial] * x[trial, :] if xi < 1 else 0    # calculate gradient

            gradient += 2 * lmda * W
            W -= learning_rate * gradient                          # update vector of parameters

            new_loss = hinge_loss(x, y, W, lmda)

            if abs(new_loss - old_loss) <= 0.0001:
                condition = True

            num_steps += 1

    return W


def logistic_loss(x, y, w, lmda):
    # TODO: Compute (regularized) Logistic Loss
    loss = 0                                # logistic loss term

    for xl, yl in zip(x, y):                # merges x and y for ease of iteration
        p = -yl * np.dot(w, xl)
        loss += math.log(1 + math.exp(p))   # calculate logistic loss

    return loss + lmda * np.dot(w, w)


def sgd_logistic(x, y, max_iter, learning_rate, lmda):
    W = np.zeros(x.shape[1])
    # TODO: implement stochastic gradient descent using the logistic loss function
    condition = False
    num_steps = 0
    np.random.seed(1)

    while not condition and num_steps < max_iter:
        trial = np.random.randint(0, len(y))

        old_loss = logistic_loss(x, y, W, lmda)

        if not cond[trial] == 0:
            p = -y[trial] * np.dot(W, x[trial, :])
            gradient = 1 / (1 + math.exp(p)) * math.exp(p) * -y[trial] * x[trial, :]
            gradient += 2 * lmda * W
            W -= learning_rate * gradient

            new_loss = logistic_loss(x, y, W, lmda)

            if abs(new_loss - old_loss) <= 0.0001:
                condition = True

            num_steps += 1

    return W


def cross_validation(x, y, sgd, lmda, learning_rate, max_iter=100, sample=range(20)):
    # Leave one out cross validation accuracy
    n_correct = 0.
    n_incorrect = 0.
    
    for i in sample:
        print "CROSS VALIDATION %s" % i
        
        training_indices = [j for j in range(x.shape[0]) if j != i]
        W = sgd(x[training_indices, ], y[training_indices, ], max_iter=max_iter, lmda=lmda, learning_rate=learning_rate)
        print W
        y_hat = np.sign(x[i, ].dot(W))

        if y_hat == y[i]:
            n_correct += 1
        else:
            n_incorrect += 1

    return n_correct / (n_correct + n_incorrect)


def accuracy(x, y, w):
    y_hat = np.sign(x.dot(w))
    correct = (y_hat == y)
    return float(sum(correct)) / len(correct)


def main():
    max_features = max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])

    # Inputs
    x = np.zeros((ntrials, max_features+1))
    for i in range(data.shape[0]):
        f = data[i][0].flatten()
        x[i, :f.shape[0]] = f
        x[i, f.shape[0]] = 1     # Bias

    # Outputs (+1 = Picture, -1 = Sentence)
    y = np.ones(ntrials)
    y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

    # Randomly permute the data
    np.random.seed(1)      # Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(ntrials)
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, ]
    y = y[permutation, ]

    # Cross validation
    # Development
    print "Accuracy (Logistic Loss):\t%s" % cross_validation(x, y, sgd_logistic, max_iter=100, lmda=1, learning_rate=0.0001, sample=range(20))
    # print "Accuracy (Hinge Loss):\t%s" % cross_validation(x, y, sgd_hinge, max_iter=100, lmda=0.1, learning_rate=0.015, sample=range(20))
    
    # Test
    print "Accuracy (Logistic Loss):\t%s" % cross_validation(x, y, sgd_logistic, max_iter=10, lmda=1, learning_rate=0.0001, sample=range(20, x.shape[0]))
    # print "Accuracy (Hinge Loss):\t%s" % cross_validation(x, y, sgd_hinge, max_iter=100, lmda=0.1, learning_rate=0.015, sample=range(20, x.shape[0]))


if __name__ == "__main__":
    main()
