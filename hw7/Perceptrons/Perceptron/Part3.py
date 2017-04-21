from sklearn.neural_network import MLPClassifier
import numpy as np


def import_images(image_file_path):
    '''
    Imports Images given file
    :param image_file_path: path to image file
    :returns: array of double arrays of images
    '''
    data = []
    with open(image_file_path) as f:
        image = []
        for line in f.read().splitlines():
            for c in line:
                image.append(convert_character(c))

            if len(image) % 784 == 0:
                data.append(image)

                image = []

    return np.asarray(data)


def convert_character(c):
    '''
    Converts character to float per documentation
    :param c: input character
    :returns: corresponding float
    '''
    if c == ' ':
        return 0.0
    elif c == '+':
        return 0.5
    elif c == '#':
        return 1.0


def import_labels(label_file_path):
    '''
    Imports Labels given file
    :param label_file_path: path to label file
    :returns: array of labels
    '''
    labels = []
    with open(label_file_path) as f:
        tmp = f.read().splitlines()

        for t in tmp:
            labels.append(int(t))

    return np.asarray(labels)


def part_3():
    # Import Training Data
    image_file_path = 'data/trainingimages'
    train_data = import_images(image_file_path)
    label_file_path = 'data/traininglabels'
    train_label = import_labels(label_file_path)

    # Import Test Data
    image_file_path = 'data/testimages'
    test_data = import_images(image_file_path)
    label_file_path = 'data/testlabels'
    test_label = import_labels(label_file_path)

    # HyperParameters
    act_funcs = ['logistic', 'tanh', 'relu']
    activation = act_funcs[0]
    hidden_layers = 3
    neurons = 5
    max_iter = 500
    shuffle = True
    verbose = True

    # Classify
    clf = MLPClassifier(activation=activation
                        , hidden_layer_sizes=(hidden_layers, neurons)
                        , max_iter=max_iter
                        , shuffle=shuffle
                        , verbose=verbose)
    clf.fit(train_data, train_label)
    predict = clf.predict(test_data)

    # Estimate Accuracy
    correct = 0.0
    num_classified = 0
    for index in range(len(predict)):
        if predict[index] == test_label[index]:
            correct += 1
        num_classified += 1

    print(correct / num_classified * 100)


if __name__ == "__main__":
    part_3()
