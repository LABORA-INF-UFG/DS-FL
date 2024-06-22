import os
import pandas as pd
import tensorflow as tf
import flwr as fl
import time


class Client(fl.client.NumPyClient):

    def __init__(self, cid):
        self.cid = int(cid)

        self.cp_round = 0
        self.last_round = 0

        self.model = self.create_model()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        

    def create_model(self):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(units=128, activation='relu', input_shape=(7,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(6, activation="softmax")
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return model

    def load_data(self):
        path = "./home/data/motion_sense"

        print(f"{path}/{self.cid}_train.pickle")
        train = pd.read_pickle(f"{path}/{self.cid}_train.pickle")
        test = pd.read_pickle(f"{path}/{self.cid}_test.pickle")

        x_train = train.drop(['subject', 'activity', 'trial', '_activity'], axis=1)
        y_train = train['_activity']

        x_test = test.drop(['subject', 'activity', 'trial', '_activity'], axis=1)
        y_test = test['_activity']  

        return (x_train, y_train), (x_test, y_test)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        inicio = time.time()
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=128, validation_data=(self.x_test, self.y_test))        
        fim = time.time()

        self.cp_round = self.cp_round + 1
        self.last_round = config['server_round']

        return self.model.get_weights(), len(self.x_train), {'cid': self.cid, 'cp_round': self.cp_round}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy, "cid": self.cid, 'cp_round': self.cp_round, 'last_round': self.last_round}


def main():
    fl.client.start_numpy_client(server_address="server:8080", client=Client(cid=os.environ['CLIENT_ID']))


if __name__ == '__main__':
    main()