from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import time
from nolearn.lasagne import NeuralNet
import dill as pickle
import sys
import os.path
import csv
import glob

class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


class DRNeuralNet(NeuralNet):

    def __init__(self, report_funcs={}, select_weight_by = "valid_accuracy", save_network_to=None, **kargs):
        '''
            report_funcs: a dictionary of {name: report function}. The function takes argument of (y_predict, y_target).
            select_weight_by: string. Weights with highest report_funcs[select_weight_by] will be saved in the end.
        '''
        if not "more_params" in kargs:
            kargs['more_params'] = {}
        kargs['more_params']['report_funcs'] = report_funcs
        kargs['more_params']['select_weight_by'] = select_weight_by
        self.best_weight = None
        self.best_performance = None
        self.best_epoch = 0
        self.save_network_to = save_network_to

        return super(DRNeuralNet, self).__init__(**kargs)


    def fit(self):
        self.initialize()
        try:
            self.train_loop()
        except KeyboardInterrupt:
            print("saving weights to", self.save_network_to, "...")
            self.save_network()
            print("weights saved.")
            sys.exit()
            
        return self

    def _create_iter_funcs(self, layers, objective, update, input_type,
                       output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        output_layer = layers['output']
        objective_params = self._get_params_for('objective')
        obj = objective(output_layer, **objective_params)
        if not hasattr(obj, 'layers'):
            # XXX breaking the Lasagne interface a little:
            obj.layers = layers

        loss_train = obj.get_loss(X_batch, y_batch)
        loss_eval = obj.get_loss(X_batch, y_batch, deterministic=True)
        predict_proba = output_layer.get_output(X_batch, deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval
            predict = predict_proba

        all_params = self.get_all_params()
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)

        train_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_train],
            updates=updates,
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        eval_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_eval, accuracy, predict],
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        predict_iter = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=predict_proba,
            givens={
                X: X_batch,
                },
            )

        return train_iter, eval_iter, predict_iter


    #use our own train loop since we gonna mess up a lot with it
    def train_loop(self):

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        info = None
        best_valid_loss = np.inf
        best_train_loss = np.inf

        if self.verbose:
            print("""
 Epoch  |  Train loss  |  Valid loss  |  Train / Val  |  Valid acc  |  Dur
--------|--------------|--------------|---------------|-------------|-------\
""")

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            target_y = []
            predict_y = []

            t0 = time.time()

            for Xb, yb in self.batch_iterator_train():
                if self.regression:
                    yb = yb.astype("float32")
                    if len(yb.shape) == 1:
                        yb = yb[:,None]
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test():
                if self.regression:
                    yb = yb.astype("float32")
                    if len(yb.shape) == 1:
                        yb = yb[:,None]
                batch_valid_loss, accuracy, predict = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)
                target_y.append(np.array(yb).squeeze())
                predict_y.append(np.array(predict).squeeze())
                

            
            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            target_y = np.hstack(target_y)
            predict_y = np.hstack(predict_y)
            print(target_y.shape, predict_y.shape)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            if self.verbose:
                best_train = best_train_loss == avg_train_loss
                best_valid = best_valid_loss == avg_valid_loss
                print(" {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  "
                      "|  {:>11.6f}  |  {:>9}  |  {:>3.1f}s".format(
                          epoch,
                          ansi.BLUE if best_train else "",
                          avg_train_loss,
                          ansi.ENDC if best_train else "",
                          ansi.GREEN if best_valid else "",
                          avg_valid_loss,
                          ansi.ENDC if best_valid else "",
                          avg_train_loss / avg_valid_loss,
                          "{:.2f}%".format(avg_valid_accuracy * 100)
                          if not self.regression else "",
                          time.time() - t0,
                          ))

            info = dict(
                epoch=epoch,
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
                valid_accuracy=avg_valid_accuracy,
                )

            for k, f in self.more_params["report_funcs"].items():
                info[k] = f(predict_y, target_y)
                if k == self.more_params["select_weight_by"]:
                        
                    if self.best_performance is None or info[k] > self.best_performance:
                        self.best_performance = info[k]
                        self.best_weight = [w.get_value() for w in self.get_all_params()]
                        self.best_epoch = epoch
            self.train_history_.append(info)
            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    def save_network(self):
        with open(self.save_network_to, "wb") as f:
            pickle.dump(self, f, -1)

    def load_best_weight(self):
        for w1, w2 in zip(self.best_weight, self.get_all_params()):
            w2.set_value(w1)

    def predict(self, test_iter):
        res = []
        for Xb in test_iter():
            res.append(self.predict_iter_(Xb))
        res = np.vstack(res)
        if self.regression:
            return res
        else:
            return np.argmax(res, axis=1)


## report functions ##

def _cat_hist(arr, n):
    hist = np.bincount(arr)
    if hist.shape[0] < n:
        hist = np.hstack([hist, np.zeros(n-hist.shape[0])])
    return hist

def confusion_matrix(predict, target):
    target = np.round(target).astype("int32")
    n_cat = np.max(target) + 1
    print("target:",np.min(target), np.max(target))
    print("predict:",np.min(predict), np.max(predict))
    #n_cat = max(np.max(predict), np.max(target)) + 1
    predict = np.round(predict).clip(0, n_cat-1).astype("int32")
    confusion_mat = np.zeros((n_cat, n_cat))
    for p, r in zip(predict, target):
        confusion_mat[p, r] += 1
    return confusion_mat


def qkappa(predict, target):
    #in case prediction is result of regression, round up
    target = np.round(target).astype("int32")
    n_cat = np.max(target) + 1
    #n_cat = max(np.max(predict), np.max(target)) + 1
    predict = np.round(predict).clip(0, n_cat-1).astype("int32")
    pred_hist = _cat_hist(predict, n_cat)
    target_hist = _cat_hist(target, n_cat)
    confusion_matrix = np.zeros((n_cat, n_cat), dtype=int)
    for p, r in zip(predict, target):
        confusion_matrix[p, r] += 1
    weight_matrix = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            weight_matrix[i,j] = float(i-j) / (n_cat -1)
    weight_matrix **= 2
    expection_matrix = np.outer(pred_hist, target_hist)
    expection_matrix /= np.sum(expection_matrix) / np.sum(confusion_matrix)
    qkappa = 1 - (np.sum(weight_matrix * confusion_matrix) / \
                  np.sum(weight_matrix * expection_matrix))
    '''
    print("pred_hist:", pred_hist)
    print("target_hist:", target_hist)
    print("weight matrix")
    print(weight_matrix)
    print("expection matrix")
    print(expection_matrix)
    '''
    print("confusion matrix")
    print(confusion_matrix)
    print("qkappa:", qkappa)
    return qkappa


def get_data_by_class(label_dic):
    '''
    label_dic: a dictionary of {image_path: label}, where labels are int in range(n_class).
    return: a list of lists, where each list contains all the images for one class.
    '''
    n_class = int(max(label_dic.values()) + 1)
    imgs_by_class = []
    for i in range(n_class):
        imgs_by_class.append([])
    for f, l in label_dic.items():
        imgs_by_class[l].append(f)
    return imgs_by_class


def get_dr_data(img_glob, csv_path):
    prefix = os.path.dirname(img_glob)
    postfix = os.path.splitext(img_glob)[1]
    label_file = csv.reader(open(csv_path,'r'))
    #skip csv title
    label_file.next()
    label_dic = {os.path.join(prefix, "{}{}".format(x[0],postfix)):int(x[1]) for x in label_file}
    label_dic = {x: label_dic[x] for x in glob.glob(img_glob)}

    return get_data_by_class(label_dic)

def split_data(data, val_per_class):
    if type(val_per_class) is int:
        val_data = [x[:val_per_class] for x in data]
        train_data = [x[val_per_class:] for x in data]
    elif type(val_per_class) is float:
        n_val_data = [int(len(x) * val_per_class) for x in data]
        val_data = [x[:n_val] for x, n_val in zip(data, n_val_data)]
        train_data = [x[n_val:] for x, n_val in zip(data, n_val_data)]
        
    return train_data, val_data



