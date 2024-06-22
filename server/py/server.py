from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
import time
from flwr.common import Metrics, Parameters
from flwr.server import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

path_save_model = "/home/data/save"
last_server_round_saved = 0

class FedServerStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
                 sample_size,
                 min_fit_clients,
                 evaluate_fn,
                 initial_parameters,
                 ):

        self.initial_parameters = initial_parameters
        self.sample_size = sample_size
        self.selected_clients = []
        self.clients_last_round = []
        self.list_of_clients = []
        self.list_of_accuracies = []
       
        super().__init__(fraction_fit=0.1,
                         min_available_clients=min_fit_clients,
                         min_evaluate_clients=min_fit_clients,
                         min_fit_clients=min_fit_clients,
                         evaluate_fn=evaluate_fn,
                         initial_parameters=initial_parameters)
   

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    
    def aggregate_fit(self, server_round, results, failures):
        weights_results = []
       

        for _, fit_res in results:
            weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))

        if parameters_aggregated is not None:           
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(parameters_aggregated)
            
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""        
       
        config = {
            'server_round': server_round
        }

        if self.on_evaluate_config_fn is not None:          
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_ins = fl.common.EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )       

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients            
        ) 
        
        return [(client, evaluate_ins) for client in clients]

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""       

        self.clients_last_round = self.selected_clients

        config = {            
            "server_round": server_round
        }

        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients            
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):  

        local_list_clients = []
        self.list_of_clients = []
        self.list_of_accuracies = []

        accs = []
        for response in results:           
            client_id = response[1].metrics['cid']
            client_accuracy = float(response[1].metrics['accuracy'])

            accs.append(client_accuracy)

            local_list_clients.append((client_id, client_accuracy))

        self.list_of_clients = [str(client[0]) for client in local_list_clients]
        self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)

        # Aggregate loss
        loss_aggregated = fl.server.strategy.aggregate.weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {
            "accuracy": accuracy_aggregated
        }  
     
        return loss_aggregated, metrics_aggregated


df = pd.read_pickle(f"./home/data/motion_sense_test.pickle")
X = df.drop(['subject', 'activity', 'trial', '_activity'], axis=1)
y = df['_activity']


# Centralized Evaluate
def get_evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
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

    model.set_weights(parameters)  # Update model with the latest parameters
    loss, accuracy = model.evaluate(X, y)

    print(f"{server_round}: -------> accuracy_centralized: {accuracy}, loss_centralized: {loss}")

    if server_round != 0:
        model.save(f"{path_save_model}/model_{last_server_round_saved + server_round}.h5")

    return loss, {"accuracy": accuracy}


fedServerStrategy = FedServerStrategy(    
    sample_size=24,    
    min_fit_clients=24,
    evaluate_fn=get_evaluate_fn,
    initial_parameters=None
    # initial_parameters=fl.common.ndarrays_to_parameters(tf.keras.models.load_model(f"{path_save_model}/model_250.h5").get_weights())
)


# Start Flower server
def main():
    fl.server.start_server(
        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fedServerStrategy
    )

if __name__ == '__main__':
    inicio = time.time()
    main()
    fim = time.time()

    print(f"Tempo: {(fim - inicio)}")
