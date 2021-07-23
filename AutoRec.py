import tensorflow as tf
from tensorflow import keras
import argparse
from functions import dataloader
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="AutoRec-MovieLens.")
    parser.add_argument('--path', nargs='?', default='./datasets/ml-1m/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.dat',
                        help='Choose a dataset.')
    parser.add_argument('--kind', nargs='?', default='I',
                        help='Choose between I-AutoRec("I") or U-AutoRec("U")')
    parser.add_argument('--layers', nargs='+', default=[300],
                        help='num of layers and nodes of each layer ')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='Regularization for Encoder,Decoder.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--test_size', type=float,default=0.2,
                        help="test_proportion of datasets")
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')

    return parser.parse_args()


class AutoRec(keras.Model):
    def __init__(self,num_features,reg=0.1,layers = [300],**kwargs):
        super().__init__(**kwargs)
        self.num_features=num_features
        self.num_layers = len(layers)
        self.n_neuron = list(map(int, layers))
        self.layers_list = []
        for index in range(self.num_layers):
            layer = keras.layers.Dense(self.n_neuron[index],
                                       kernel_regularizer=keras.regularizers.l2(reg),
                                       activation=keras.activations.relu,
                                       name=f'layer{index}')
            self.layers_list.append(layer)
        self.layers_list.append(keras.layers.Dense(num_features,
                                                   kernel_regularizer=keras.regularizers.l2(reg),
                                                   ))

    def call(self, inputs):
        result = inputs
        for layer in self.layers_list:
            result = layer(result)

        return result

def MaskedMSELoss(y_true,y_pred):
    mask = y_true != 0
    #마스크 : 관측된 데이터에 대해서는 1, 관측되지 않은데이터는 0
    mask_float = tf.cast(mask,tf.float32)
    masked_error = tf.reduce_mean(tf.pow(tf.subtract(mask_float * y_pred,y_true),2))
    #기존에 관측되지 않았던 결과에 대해서는 마스킹을 진행하여 Loss 계산
    return masked_error


    cost2 = tf.reduce_sum(tf.pow(predicted-Y, 2))/(num_instances)


if __name__ == "__main__":
    args = parse_args()
    layers = args.layers
    reg = args.reg

    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience

    dataloader = dataloader.dataloader(args.path,args.dataset,args.test_size)

    if args.kind.lower() =="u":
        train_data,test_data = dataloader.make_user_autorec_input()
        num_features = dataloader.num_item

    elif args.kind.lower() =="i":
        train_data,test_data  = dataloader.make_item_autorec_input()
        num_features = dataloader.num_user

    model = AutoRec(num_features,reg,layers)

    if learner.lower() == "adagrad":
        model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate), loss= MaskedMSELoss,
                      metrics=[keras.metrics.RootMeanSquaredError()])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), lloss= MaskedMSELoss,
                      metrics=[keras.metrics.RootMeanSquaredError()])
    elif learner.lower() == "adam":
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss= MaskedMSELoss,
                      metrics=[keras.metrics.RootMeanSquaredError()])
    else:
        model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss= MaskedMSELoss,
                      metrics=[keras.metrics.RootMeanSquaredError()])

    early_stopping_callback = keras.callbacks.EarlyStopping(patience=patience,restore_best_weights=True)
    model_out_file = '%s-AutoRec%s.h5' % (args.kind,datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
    model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

    if args.out:
        history = model.fit(train_data, train_data, batch_size=batch_size, epochs=epochs,
                            validation_data=(test_data, test_data), callbacks=[early_stopping_callback,
                                                                                             model_check_cb]
                            )
    else:
        history = model.fit(train_data, train_data, batch_size=batch_size, epochs=epochs,
                            validation_data=(test_data, test_data), callbacks=[early_stopping_callback]

                            )

    pd.DataFrame(history.history["val_root_mean_squared_error"]).plot()
    plt.show()