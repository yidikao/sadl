import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical  # noqa
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
from scipy import io as sio

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument("--model", "-m", help="Model path", type=str, required=True)
    parser.add_argument("--target_input", "-ti", help="Target input file", type=str, required=True)
    args = parser.parse_args()
    assert args.d in ["mnist","emnist", "fmnist","kmnist", "cifar10","reuters","caltech101"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        split_idx = int(len(x_train) * 0.9)
        x_train = x_train[:split_idx]
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32") / 255.0  # [0, 1]
        x_test = x_test.astype("float32") / 255.0  # [0, 1]
        # Load pre-trained model.
        model = load_model(args.model)
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["dense_1"]

        # Load target set.
        x_target = np.load(args.target_input)
        x_target = x_target.reshape(-1, 28, 28, 1)
        x_target = x_target.astype("float32") / 255.0
    elif args.d == "emnist":
        args.num_classes = 26
        mat = sio.loadmat('../../testing/datasets/emnist-letters.mat')
        data = mat['dataset']
        x_train = data['train'][0, 0]['images'][0, 0]
        y_train = data['train'][0, 0]['labels'][0, 0]
        x_test = data['test'][0, 0]['images'][0, 0]
        y_test = data['test'][0, 0]['labels'][0, 0]

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32") / 255.0  # [0, 1]
        x_test = x_test.astype("float32") / 255.0  # [0, 1]

        # Load pre-trained model.
        model = load_model(args.model)
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["dense_1"]

        # Load target set.
        x_target = np.load(args.target_input)
        x_target = x_target.reshape(-1, 28, 28, 1)
        x_target = x_target.astype("float32") / 255.0
    elif args.d == "fmnist":
        args.num_classes = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
        x_train, x_test = x_train / 255., x_test / 255.
        y_train, y_test = to_categorical(y_train, args.num_classes), to_categorical(y_test, args.num_classes)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # Load pre-trained model.
        model = load_model(args.model)
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["dense_1"]

        # Load target set.
        x_target = np.load(args.target_input)
        x_target = x_target.reshape(-1, 28, 28, 1)
        x_target = x_target.astype("float32") / 255.0
    elif args.d == "kmnist":
        args.num_classes = 10
        x_train = np.load('../../testing/datasets/kmnist-train-imgs.npz')['arr_0']
        y_train = np.load('../../testing/datasets/kmnist-train-labels.npz')['arr_0']
        x_test = np.load('../../testing/datasets/kmnist-test-imgs.npz')['arr_0']
        y_test = np.load('../../testing/datasets/kmnist-test-labels.npz')['arr_0']
        x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
        x_train, x_test = x_train / 255., x_test / 255.
        y_train, y_test = to_categorical(y_train, args.num_classes), to_categorical(y_test, args.num_classes)
            # Load pre-trained model.
        model = load_model(args.model)
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["dense_1"]

        # Load target set.
        x_target = np.load(args.target_input)
        x_target = x_target.reshape(-1, 28, 28, 1)
        x_target = x_target.astype("float32") / 255.0
    elif args.d == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model(args.model)
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        x_target = np.load(args.target_input)
    elif args.d == "reuters":
        from keras.datasets import reuters
        from keras.preprocessing import sequence

        args.num_classes = 46
        max_words = 20000
        maxlen = 80

        (x_train, y_train), (x_test, y_test) = reuters.load_data()
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        x_train = x_train.astype('int32')
        x_test = x_test.astype('int32')
        model = load_model(args.model)
        model.summary()

        layer_names = ["lstm_1"]

        x_target = np.load(args.target_input)
    elif args.d == "caltech101":
        from keras.preprocessing import image_dataset_from_directory

        args.num_classes = 102
        img_size = (224, 224)
        data_dir = '../../testing/datasets/caltech-101/101_ObjectCategories'

        train_ds = image_dataset_from_directory(data_dir,
                                                validation_split=0.2,
                                                subset="training",
                                                seed=42,
                                                image_size=img_size,
                                                batch_size=32)
        val_ds = image_dataset_from_directory(data_dir,
                                            validation_split=0.2,
                                            subset="validation",
                                            seed=42,
                                            image_size=img_size,
                                            batch_size=32)

        x_train, y_train = [], []
        for imgs, labels in train_ds:
            x_train.append(imgs.numpy())
            y_train.append(labels.numpy())
        x_train = np.concatenate(x_train)

        x_test, y_test = [], []
        for imgs, labels in val_ds:
            x_test.append(imgs.numpy())
            y_test.append(labels.numpy())
        x_test = np.concatenate(x_test)

        model = load_model(args.model)
        model.summary()

        layer_names = ["global_average_pooling2d"]

        x_target = np.load(args.target_input)  # shape: (N, 224, 224, 3), already float32


    if args.lsa:
        test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)

        target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
        )
        avg_lsa = np.mean(target_lsa)

        if len(target_lsa) == 0:
            print("ERROR: target_lsa is empty!")
            exit(1)

        with open('mt-lsa.csv', 'w') as f:
           f.write(f'{target_cov},{avg_lsa}\n')

    if args.dsa:
        test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)

        target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        )
        avg_dsa = np.mean(target_dsa)
        auc = compute_roc_auc(test_dsa, target_dsa)
        buckets = np.digitize(target_dsa, np.linspace(np.amin(target_dsa), args.upper_bound, args.n_bucket))
        unique_buckets, bucket_counts = np.unique(buckets, return_counts=True)
        print(infog("\nBucket Distribution:"))
        print(f"Total unique buckets: {len(unique_buckets)} / {args.n_bucket}")
        print(f"Bucket occupancy: {bucket_counts}")
        print(f"Occupied bucket indices: {unique_buckets}")
        print(f"Min samples in bucket: {np.min(bucket_counts)}")
        print(f"Max samples in bucket: {np.max(bucket_counts)}")
        print(f"Avg samples per bucket: {np.mean(bucket_counts):.2f}")

        print(infog("Bucket distribution saved to ./tmp/bucket_distribution.npy"))
        print(infog("ROC-AUC: " + str(auc * 100)))
        with open('mt-dsa.csv', 'w') as f:
           f.write(f'{target_cov},{avg_dsa}\n')



    print(infog("{} coverage: ".format(args.target) + str(target_cov)))
