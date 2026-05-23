# batch_sa.py
import numpy as np
import argparse
import os
import csv
from keras.models import load_model
from sa import get_ats, find_closest_at, get_sc, _get_kdes
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, required=True)
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--subset_dir", type=str, required=True)
parser.add_argument("--num_subsets", type=int, default=500)
parser.add_argument("--save_path", type=str, default="./tmp/")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--upper_bound", type=int, default=2)
parser.add_argument("--n_bucket", type=int, default=1000)
parser.add_argument("--var_threshold", type=float, default=1e-5)
parser.add_argument("--output", type=str, default="batch_lsa_dsa.csv")
args = parser.parse_args()
args.is_classification = True


if args.d == "mnist":
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train[:int(len(x_train)*0.9)].reshape(-1,28,28,1).astype("float32")/255.
    # x_train = x_train[:int(len(x_train) * 0.8)]
    x_test  = x_test.reshape(-1,28,28,1).astype("float32")/255.
    layer_names = ["dense_1"]
elif args.d == "emnist":
    from scipy import io as sio
    args.num_classes = 26
    mat = sio.loadmat('../../testing/datasets/emnist-letters.mat')
    data = mat['dataset']
    x_train = data['train'][0,0]['images'][0,0].reshape(-1,28,28,1).astype("float32")/255.
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = data['test'][0,0]['images'][0,0].reshape(-1,28,28,1).astype("float32")/255.
    layer_names = ["dense_1"]
elif args.d == "fmnist":
    from keras.datasets import fashion_mnist
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32")/255.
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = np.expand_dims(x_test,  -1).astype("float32")/255.
    layer_names = ["dense_1"]
elif args.d == "kmnist":
    x_train = np.load('../../testing/datasets/kmnist-train-imgs.npz')['arr_0']
    x_test  = np.load('../../testing/datasets/kmnist-test-imgs.npz')['arr_0']
    x_train = np.expand_dims(x_train,-1).astype("float32")/255.
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = np.expand_dims(x_test, -1).astype("float32")/255.
    layer_names = ["dense_1"]
elif args.d == "svhn":
    from scipy import io as sio
    train_data = sio.loadmat('../../testing/datasets/train_32x32.mat')
    test_data  = sio.loadmat('../../testing/datasets/test_32x32.mat')
    x_train = np.transpose(train_data['X'], (3,0,1,2)).astype("float32") / 255.
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = np.transpose(test_data['X'],  (3,0,1,2)).astype("float32") / 255.
    layer_names = ["dense_1"]
elif args.d == "reuters":
    from keras.datasets import reuters
    from keras.preprocessing import sequence
    args.num_classes = 46
    (x_train, _), (x_test, _) = reuters.load_data(num_words=20000)
    x_train = sequence.pad_sequences(x_train, maxlen=80).astype('int32')
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = sequence.pad_sequences(x_test,  maxlen=80).astype('int32')
    layer_names = ["lstm_1"]
elif args.d == "caltech101":
    from keras.preprocessing import image_dataset_from_directory
    args.num_classes = 102
    data_dir = '../../testing/datasets/caltech-101/101_ObjectCategories'
    def ds_to_numpy(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.numpy()); ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)
    train_ds = image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",   seed=42, image_size=(224,224), batch_size=32)
    val_ds   = image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=42, image_size=(224,224), batch_size=32)
    x_train, _ = ds_to_numpy(train_ds)
    # x_train = x_train[:int(len(x_train)*0.8)]
    x_test,  _ = ds_to_numpy(val_ds)
    layer_names = ["re_lu"]
    
elif args.d == "cifar10":
    from keras.datasets import cifar10
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.
    x_train = x_train[:int(len(x_train)*0.8)]
    x_test  = x_test.astype("float32") / 255.
    args.upper_bound = 100
    layer_names = ["global_average_pooling2d"]

model = load_model(args.model)
os.makedirs(args.save_path, exist_ok=True)

train_ats_path = os.path.join(args.save_path, f"{args.d}_train_{layer_names[0]}_ats.npy")
train_pred_path = os.path.join(args.save_path, f"{args.d}_train_pred.npy")

if os.path.exists(train_ats_path):
    print("Found cached train ATs, loading...")
    train_ats  = np.load(train_ats_path)
    train_pred = np.load(train_pred_path)
else:
    print("Computing train ATs...")
    print("Model layers and output dimensions:")
    for layer in model.layers:
        try:
            shape = layer.output.shape
            print(f"  {layer.name}: {shape}")
        except AttributeError:
            print(f"  {layer.name}: (multiple outputs)")
    train_ats, train_pred = get_ats(model, x_train, "train", layer_names,
                                     num_classes=args.num_classes,
                                     is_classification=True,
                                     save_path=(train_ats_path, train_pred_path))


if args.d == "svhn":
    train_pred = train_pred % 10
class_matrix = {}
all_idx = []
for i, label in enumerate(train_pred):
    if label not in class_matrix:
        class_matrix[label] = []
    class_matrix[label].append(i)
    all_idx.append(i)
all_idx_set = set(all_idx)

print("Labels in class_matrix:", sorted(class_matrix.keys()))
print("train_pred unique:", np.unique(train_pred))
print("Computing KDEs for LSA...")
kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args)


results = []
for subset_idx in tqdm(range(args.num_subsets), desc="Subsets"):
    ti_file = os.path.join(args.subset_dir, f"subset_{subset_idx}_inputs.npy")
    if not os.path.exists(ti_file):
        continue

    x_target = np.load(ti_file)
    if args.d in ["mnist","emnist","fmnist","kmnist"]:
        x_target = x_target.reshape(-1,28,28,1)
    elif args.d == "reuters":
        x_target = x_target.astype('int32')

    target_ats, target_pred = get_ats(model, x_target, f"subset{subset_idx}", layer_names,
                                       num_classes=args.num_classes,
                                       is_classification=True,
                                       save_path=None)  

    # LSA
    from sa import _get_lsa
    lsa = []
    for i, at in enumerate(target_ats):
        label = target_pred[i]
        if label not in kdes:
            continue
        lsa.append(_get_lsa(kdes[label], at, removed_cols))
    lsa_cov = get_sc(np.amin(lsa), args.upper_bound, args.n_bucket, lsa)
    avg_lsa  = np.mean(lsa)

    # DSA
    dsa = []
    for at, label in zip(target_ats, target_pred):
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _     = find_closest_at(a_dot, train_ats[list(all_idx_set - set(class_matrix[label]))])
        dsa.append(a_dist / b_dist)
    dsa_cov = get_sc(np.amin(dsa), args.upper_bound, args.n_bucket, dsa)
    avg_dsa  = np.mean(dsa)

    results.append((subset_idx, lsa_cov, avg_lsa, dsa_cov, avg_dsa))


with open(args.output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["subset_index","lsa_coverage","avg_lsa","dsa_coverage","avg_dsa"])
    writer.writerows(results)

print(f"Done → {args.output}")