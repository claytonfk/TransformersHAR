import logging
import gc
import os

import h5py
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.decomposition import PCA

from .trajectory import load_weights, weight_encoder


def get_vectors(model, seed=None, trajectory=None):

    np.random.seed(seed)
    vector_x, vector_y = list(), list()
    weights = model.get_weights()
    v_names = [v.name for v in model.trainable_variables]
    weights = [w for w in weights if w.shape != ()]

    if trajectory:
        # this has to be re-written
        load_weights(model, trajectory)
        file_path = os.path.join(trajectory, ".trajectory", "model_weights.hdf5")

        with h5py.File(file_path, "r+") as f:
            differences = list()
            trajectory = np.array(f["weights"])
            for i in range(0, len(trajectory) - 1):
                differences.append(trajectory[i] - trajectory[-1])

            pca = PCA(n_components=2)
            pca.fit(np.array(differences))
            f["X"], f["Y"] = pca.transform(np.array(differences)).T

        vector_x = weight_encoder(model, pca.components_[0])
        vector_y = weight_encoder(model, pca.components_[1])

        return weights, vector_x, vector_y

    else:
        cast = np.array([1]).T
        for lidx, layer in enumerate(weights):
            # set standard normal parameters
            # filter-wise normalization
            
            #layer = layer.reshape((-1, layer.shape[-1]))
            
            
            k = len(layer.shape) - 1
            d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)

            dist_x = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[..., np.newaxis])
            dist_x = dist_x.reshape(d.shape)
            
            

            vector_x.append(
                (
                    dist_x * (cast * np.linalg.norm(layer, axis=k))[..., np.newaxis]
                ).reshape(d.shape)
            )

            d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)
                        
            
            dist_y = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[..., np.newaxis])
            dist_y = dist_y.reshape(d.shape)

            vector_y.append(
                (
                    dist_y * (cast * np.linalg.norm(layer, axis=k))[..., np.newaxis]
                ).reshape(d.shape)
            )
            

        return weights, vector_x, vector_y


def _obj_fn(old_weights, model, dataset, solution, loss, num_classes):

    loss =  tf.keras.losses.CategoricalCrossentropy()
    v_names = list(model.get_weight_paths().keys())
    idx = 0
    solution_ = []
    
    for lidx, old_layer in enumerate(old_weights):
        if 'encoder' in v_names[lidx] and 'dense' in v_names[lidx]:
            solution_.append(old_layer)
            idx += 1
        else:
            if old_layer.shape != ():
                shape = old_layer.shape
                solution_.append(solution[idx].reshape(shape))
                idx += 1
            else:
                solution_.append(old_layer)
        
    model.set_weights(solution_)
    avg_loss = 0
    
    num_batches = 10
    num_windows = 0
    for batch_index in range(num_batches):#dataset.num_train_batches):
        x, y = dataset.get_batch('train', batch_index)
        batch_size = x.shape[0]
        y_encoded = tf.one_hot(y, depth = num_classes, axis = 1, dtype = tf.int32)
        
        if model.model_name == 'retnet':
            predictions = model.call(x, force_recurrent = True)
        else:
            predictions = model.call(x, training = True)
            
        loss_value = loss(y_encoded, predictions)
        avg_loss   += loss_value.numpy()*batch_size
        num_windows += batch_size
            
        
    # predictions_argmax = tf.argmax(predictions, axis = 1)
    # seen_classes = list(set(y.reshape(-1)))
    

    # f1_value = f1_score(y, predictions_argmax, average= 'weighted', labels = seen_classes, zero_division = 0)

    # with tf.GradientTape() as tape:
    #     predictions = model.call(x, training = True)
    

    avg_loss = avg_loss/num_windows

    return [avg_loss, -1]



def build_mesh(
    model,
    dataset,
    loss,
    num_classes,
    grid_length,
    extension=1,
    filename="meshfile",
    verbose=True,
    seed=None,
    trajectory=None,
):
    
    old_weights = model.get_weights()


    logging.basicConfig(level=logging.INFO)

    z_keys = ['loss', 'f1_score']
    Z = list()

    # get vectors and set spacing
    origin, vector_x, vector_y = get_vectors(model, seed=seed, trajectory=trajectory)

    space = np.linspace(-extension, extension, grid_length)

    X, Y = np.meshgrid(space, space)

    
    for i in range(grid_length):
        for j in range(grid_length):
            solution = [origin[e] + X[i][j] * vector_x[e] + Y[i][j] * vector_y[e] for e in range(len(origin))]

            Z.append(_obj_fn(old_weights, model, dataset, solution, loss, num_classes))
            
            print(f"Iteration {j + 1}/{i+1} (grid length set to {grid_length})")

    model.set_weights(old_weights)
    
    Z = np.array(Z)
    os.makedirs("./files", exist_ok=True)

    with h5py.File("./files/{}.hdf5".format(filename), "w") as f:

        f["space"] = space
        original_results = _obj_fn(old_weights, model, dataset, origin, loss, num_classes)

        for i, metric in enumerate(z_keys):
            f["original_" + metric] = original_results[i]
            f[metric] = Z[:, i].reshape(X.shape)
        f.close()

    del Z
    gc.collect()
