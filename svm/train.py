import time
import os
import argparse
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data
from svm_params import TRAINING_PROPS, IMAGE_PROPS


def train(epochs=TRAINING_PROPS.epochs, random_state=TRAINING_PROPS.random_state,
          kernel=TRAINING_PROPS.kernel, decision_function=TRAINING_PROPS.decision_function, gamma=TRAINING_PROPS.gamma,
          train_model=True, save_model=True):
    print("loading dataset " + IMAGE_PROPS.DATASET + "...")
    test = {}
    if train_model:
        data, validation = load_data(validation_data=True, test_data=False)
    else:
        data, validation, test = load_data(validation_data=True, test_data=True)

    if train_model:
        # Training phase
        print("building model...")
        model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel,
                    decision_function_shape=decision_function, gamma=gamma)

        print("start training...")
        print("--")
        print("kernel: {}".format(kernel))
        print("decision function: {} ".format(decision_function))
        print("max epochs: {} ".format(epochs))
        print("gamma: {} ".format(gamma))
        print("--")
        print("Training samples: {}".format(len(data['Y'])))
        print("Validation samples: {}".format(len(validation['Y'])))
        print("--")
        start_time = time.time()
        model.fit(data['X'], data['Y'])
        training_time = time.time() - start_time
        print("training time = {0:.1f} sec".format(training_time))

        if save_model:
            print("saving model...")
            with open(TRAINING_PROPS.model_path, 'wb') as f:
                cPickle.dump(model, f)

        print("evaluating...")
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        return validation_accuracy
    else:
        print("start evaluation...")
        print("loading pretrained model...")
        if os.path.isfile(TRAINING_PROPS.model_path):
            with open(TRAINING_PROPS.model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAINING_PROPS.model_path))
            exit()

        print("--")
        print("Validation samples: {}".format(len(validation['Y'])))
        print("Test samples: {}".format(len(test['Y'])))
        print("--")
        print("evaluating...")
        start_time = time.time()
        validation_accuracy = evaluate(model, validation['X'], validation['Y'])
        print("  - validation accuracy = {0:.1f}".format(validation_accuracy * 100))
        test_accuracy = evaluate(model, test['X'], test['Y'])
        print("  - test accuracy = {0:.1f}".format(test_accuracy * 100))
        print("  - evaluation time = {0:.1f} sec".format(time.time() - start_time))
        return test_accuracy


def evaluate(model, x, y):
    predicted_y = model.predict(x)
    accuracy = accuracy_score(y, predicted_y)
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", default="yes")
    parser.add_argument("-e", "--evaluate", default="no")
    args = parser.parse_args()
    if args.train.lower() == "yes":
        train()
    if args.evaluate.lower() == "yes":
        train(train_model=False)
