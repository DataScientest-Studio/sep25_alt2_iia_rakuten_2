"""Génère les notebooks du parcours de classification supervisée des images.

Parcours : 00 données et splits → 01 CNN de zéro (baseline) → 02 transfer learning EfficientNetB0
→ 03 comparaison → 04 interprétabilité Grad-CAM. Les notebooks déjà exécutés ne sont pas écrasés,
sauf avec l'option --force.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "image_classification"


def source(text: str) -> list[str]:
    text = dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(cells: list[dict], kernel_name: str = "python3", display_name: str = "Python 3") -> dict:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"  # Identifiants stables : diffs Git lisibles.
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": display_name, "language": "python", "name": kernel_name},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_CONFIG = r'''
from pathlib import Path
import json
import os
import random

import certifi
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Python installé depuis python.org sur macOS ne référence pas toujours le trousseau
# système. On fournit à urllib/Keras un bundle CA explicite sans désactiver SSL.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

def find_project_root() -> Path:
    """Permet d'exécuter le notebook depuis sa racine ou son propre dossier."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data" / "raw_x.csv").exists() and (candidate / "images").exists():
            return candidate
    raise FileNotFoundError("Racine du projet introuvable (data/raw_x.csv et images/ attendus).")

ROOT = find_project_root()
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images" / "image_train"
ARTIFACTS_DIR = ROOT / "artifacts" / "image_classification"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(ARTIFACTS_DIR / ".keras"))

print("Racine :", ROOT)
print("Artefacts :", ARTIFACTS_DIR)
print("Certificats SSL :", os.environ["SSL_CERT_FILE"])
'''


DATASET_FUNCTIONS = r'''
import tensorflow as tf

tf.keras.utils.set_random_seed(SEED)

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

with open(ARTIFACTS_DIR / "class_mapping.json", encoding="utf-8") as handle:
    class_mapping = json.load(handle)

index_to_code = [int(value) for value in class_mapping["index_to_code"]]
code_to_index = {code: index for index, code in enumerate(index_to_code)}
NUM_CLASSES = len(index_to_code)

manifest = pd.read_csv(ARTIFACTS_DIR / "image_manifest.csv")
manifest["class_index"] = manifest["prdtypecode"].map(code_to_index)
assert manifest["class_index"].notna().all()

def decode_resize(path, label, weight):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, *TARGET_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)  # Pixels dans [0, 255] : chaque modèle gère sa mise à l'échelle.
    return image, tf.cast(label, tf.int32), tf.cast(weight, tf.float32)

def make_dataset(frame, training=False, transform=None, weighted=False):
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["class_index"].astype("int32").to_numpy()
    weights = frame["sample_weight"].astype("float32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_resize, num_parallel_calls=AUTOTUNE)
    if transform is not None:
        ds = ds.map(transform, num_parallel_calls=AUTOTUNE)
    if not weighted:
        ds = ds.map(lambda x, y, w: (x, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_df = manifest.query("split == 'train'").copy()
val_df = manifest.query("split == 'validation'").copy()
print(f"Train={len(train_df):,} | Validation={len(val_df):,} | Classes={NUM_CLASSES}")
'''


TRAINING_UTILS = r'''
import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

def build_augmentation():
    """Augmentation identique pour tous les modèles : seule l'architecture change d'un run à l'autre."""
    return keras.Sequential(
        [
            layers.RandomRotation(0.03, fill_mode="reflect", seed=SEED),
            layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="augmentation",
    )

def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

class EpochTimer(keras.callbacks.Callback):
    """Ajoute la durée de chaque époque au journal CSV."""

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["epoch_time_s"] = time.perf_counter() - self._start

def callbacks_for(run_dir, checkpoint_name, patience, append):
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        EpochTimer(),  # Doit précéder CSVLogger pour que la durée soit journalisée.
        keras.callbacks.ModelCheckpoint(
            run_dir / checkpoint_name, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
        keras.callbacks.CSVLogger(run_dir / "training_log.csv", append=append),
    ]

def count_params(model):
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    return {"params_total": int(model.count_params()), "params_trainable": trainable}

def save_run_summary(run_dir, **values):
    """Fusionne les informations du run dans run_summary.json, relu par le notebook 03."""
    path = run_dir / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.update(values)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

def evaluate_on_validation(model, dataset, frame):
    probabilities = model.predict(dataset, verbose=1)
    y_true = frame["class_index"].to_numpy()
    y_pred = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(top_k_accuracy_score(
            y_true, probabilities, k=3, labels=np.arange(NUM_CLASSES)
        )),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, y_true, y_pred

def plot_history(log_path, title, phase_boundary=None):
    log = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(log["epoch"] + 1, log[metric], marker="o", label="train")
        ax.plot(log["epoch"] + 1, log[f"val_{metric}"], marker="o", label="validation")
        if phase_boundary is not None:
            ax.axvline(phase_boundary + 0.5, color="grey", linestyle="--", label="début du fine-tuning")
        ax.set_xlabel("Époque")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    plt.show()
'''


CATEGORY_NAMES = r'''
category_names = {
    10: "Livres occasion",
    2280: "Journaux et revues occasion",
    2403: "Livres, BD et magazines",
    2522: "Fournitures papeterie et accessoires bureau",
    2705: "Livres neufs",
    40: "Jeux vidéos, CDs, équipements, câbles, neufs",
    50: "Accessoires gaming",
    60: "Consoles de jeu",
    2462: "Jeux vidéos occasion",
    2905: "Jeux vidéos pour PC",
    1140: "Figurines, objets pop culture",
    1160: "Cartes de jeux",
    1180: "Figurines et jeux de rôles",
    1280: "Jouets enfants",
    1281: "Jeux société enfants",
    1300: "Modélisme",
    1302: "Jeux de plein air, habits",
    1560: "Mobilier général",
    2582: "Mobilier de jardin",
    1320: "Puériculture, accessoires bébé",
    2220: "Animalerie",
    2583: "Piscine et accessoires",
    2585: "Outillage de jardin, équipements extérieurs",
    1920: "Linge de maison, oreillers, coussins",
    2060: "Décoration",
    1301: "Chaussettes bébés, petites photos",
    1940: "Confiserie",
}
missing_category_names = sorted(set(index_to_code) - set(category_names))
if missing_category_names:
    raise ValueError(f"Libellés manquants pour les catégories : {missing_category_names}")
readable_class_labels = [f"{code} — {category_names[code]}" for code in index_to_code]

def short_label(class_index):
    code = index_to_code[int(class_index)]
    return f"{code} {category_names[code]}"
'''


MODEL_LOADING = r'''
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def build_legacy_efficientnet():
    """Architecture du premier run EfficientNet, pour charger ses poids sauvés par un Keras plus récent."""
    augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.03, fill_mode="reflect", seed=SEED),
            layers.RandomZoom(0.10, fill_mode="reflect", seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="augmentation",
    )
    inputs = keras.Input(shape=(*TARGET_SIZE, 3), name="image")
    x = augmentation(inputs)
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(*TARGET_SIZE, 3)
    )
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30, seed=SEED)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="prdtypecode")(x)
    return keras.Model(inputs, outputs)

def load_evaluation_model(path):
    try:
        return keras.models.load_model(path, compile=False)
    except TypeError as error:
        # Keras 3.15 ajoute ces champs à VarianceScaling ; Keras 3.12 (Metal)
        # ne sait pas les désérialiser. Les poids restent compatibles.
        if "input_axes" not in str(error) and "output_axes" not in str(error):
            raise
        print(f"Chargement rétrocompatible de {path}")
        model = build_legacy_efficientnet()
        model.load_weights(path)
        return model

def clahe_numpy(image):
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    if float(lightness.std()) >= 40.0:
        return image_uint8.astype(np.float32)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(
        cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB
    ).astype(np.float32)

def apply_clahe_tf(image, label, weight):
    image = tf.numpy_function(clahe_numpy, [image], Tout=tf.float32)
    image.set_shape((*TARGET_SIZE, 3))
    return image, label, weight

def first_existing(*paths):
    return next((path for path in paths if path.exists()), None)

TRANSFER_DIR = ARTIFACTS_DIR / "efficientnet_transfer"
# Premier run EfficientNet (notebooks archive_v1) : modèle fine-tuné disponible, sans checkpoint de phase 1.
LEGACY_TRANSFER_DIR = ARTIFACTS_DIR / "efficientnet_baseline"

candidate_experiments = {
    "cnn_scratch": {
        "label": "1. CNN entraîné de zéro",
        "run_dir": ARTIFACTS_DIR / "cnn_scratch",
        "path": ARTIFACTS_DIR / "cnn_scratch" / "best_model.keras",
        "transform": None,
        "gradcam_layer": "features",
    },
    "efficientnet_frozen": {
        "label": "2. EfficientNetB0 gelé (feature extraction)",
        "run_dir": TRANSFER_DIR,
        "path": TRANSFER_DIR / "feature_extraction_best.keras",
        "transform": None,
        "gradcam_layer": "efficientnetb0",
    },
    "efficientnet_finetuned": {
        "label": "3. EfficientNetB0 fine-tuné",
        "run_dir": TRANSFER_DIR if (TRANSFER_DIR / "best_model.keras").exists() else LEGACY_TRANSFER_DIR,
        "path": first_existing(TRANSFER_DIR / "best_model.keras", LEGACY_TRANSFER_DIR / "best_model.keras"),
        "transform": None,
        "gradcam_layer": "efficientnetb0",
    },
    "efficientnet_clahe": {
        "label": "Ablation : fine-tuné + CLAHE",
        "run_dir": ARTIFACTS_DIR / "efficientnet_clahe",
        "path": ARTIFACTS_DIR / "efficientnet_clahe" / "best_model.keras",
        "transform": apply_clahe_tf,
        "gradcam_layer": "efficientnetb0",
    },
}
experiments = {
    name: run for name, run in candidate_experiments.items()
    if run["path"] is not None and run["path"].exists()
}
for name, run in candidate_experiments.items():
    status = run["path"] if name in experiments else "absent"
    print(f"{run['label']:<45} {status}")

PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)
'''



notebooks: dict[str, dict] = {}


notebooks["00_preparer_donnees_et_splits.ipynb"] = notebook([
    md('''
    # 00 — Préparer les données et les splits

    Ce notebook construit le **contrat de données unique** utilisé par tous les modèles image.

    Objectifs :

    - joindre les caractéristiques et les labels par `Id` ;
    - vérifier l'existence et la lisibilité de chaque image ;
    - détecter les images uniformes et les doublons perceptuels ;
    - empêcher un même groupe de doublons d'apparaître dans train et validation ;
    - exclure les groupes impossibles à apprendre, associés à plusieurs labels ;
    - sauvegarder un manifeste reproductible et le mapping des 27 classes.

    Le calcul du pHash sur toutes les images est coûteux la première fois. Son résultat est mis en cache.
    '''),
    code(COMMON_CONFIG),
    md('''
    ## 1. Charger et joindre les tables

    Une jointure explicite sur `Id` évite le risque silencieux d'un désalignement entre X et Y.
    '''),
    code('''
    raw_x = pd.read_csv(DATA_DIR / "raw_x.csv")
    raw_y = pd.read_csv(DATA_DIR / "raw_y.csv")

    required_x = {"Id", "productid", "imageid"}
    required_y = {"Id", "prdtypecode"}
    assert required_x.issubset(raw_x.columns), required_x - set(raw_x.columns)
    assert required_y.issubset(raw_y.columns), required_y - set(raw_y.columns)
    assert raw_x["Id"].is_unique and raw_y["Id"].is_unique

    data = raw_x.merge(raw_y, on="Id", how="inner", validate="one_to_one")
    assert len(data) == len(raw_x) == len(raw_y)
    data["image_filename"] = (
        "image_" + data["imageid"].astype(str)
        + "_product_" + data["productid"].astype(str) + ".jpg"
    )
    data["image_path"] = data["image_filename"].map(lambda name: str((IMAGE_DIR / name).resolve()))
    data.head()
    '''),
    code('''
    exists = data["image_path"].map(os.path.isfile)
    print(f"Images trouvées : {exists.sum():,}/{len(data):,}")
    if not exists.all():
        display(data.loc[~exists, ["Id", "image_path"]].head())
        raise FileNotFoundError(f"{(~exists).sum()} images sont absentes.")

    counts = data["prdtypecode"].value_counts().sort_index()
    display(counts.rename("effectif").to_frame())
    print("Ratio classe majoritaire/minoritaire :", round(counts.max() / counts.min(), 2))
    '''),
    md('''
    ## 2. Qualité et empreinte perceptuelle

    Le pHash identifie ici les images visuellement identiques avec la même empreinte. Les pixels ne sont
    pas chargés en mémoire simultanément. Une image uniforme est définie par un écart-type en niveaux de
    gris inférieur à 5, en cohérence avec l'EDA existante.
    '''),
    code('''
    import hashlib
    from PIL import Image, UnidentifiedImageError
    import imagehash
    from tqdm.auto import tqdm

    HASH_CACHE = ARTIFACTS_DIR / "image_hashes.csv"

    def inspect_image(path):
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                gray = np.asarray(image.convert("L"), dtype=np.float32)
                return str(imagehash.phash(image)), float(gray.std()), True
        except (OSError, UnidentifiedImageError):
            return None, np.nan, False

    if HASH_CACHE.exists():
        image_info = pd.read_csv(HASH_CACHE)
        print("Cache chargé :", HASH_CACHE)
    else:
        rows = []
        for row in tqdm(data[["Id", "image_path"]].itertuples(index=False), total=len(data)):
            phash, gray_std, readable = inspect_image(row.image_path)
            rows.append({"Id": row.Id, "phash": phash, "gray_std": gray_std, "readable": readable})
        image_info = pd.DataFrame(rows)
        image_info.to_csv(HASH_CACHE, index=False)

    assert image_info["Id"].is_unique
    data = data.merge(image_info, on="Id", validate="one_to_one")
    data["uniform_image"] = data["gray_std"] < 5
    print(data[["readable", "uniform_image"]].value_counts())
    '''),
    md('''
    ## 3. Construire les groupes et contrôler les conflits

    Toutes les lignes partageant le même pHash reçoivent le même `duplicate_group`. Si une même image est
    associée à plusieurs codes produit, le signal visuel est contradictoire : le groupe est conservé dans
    le rapport d'audit, mais exclu de l'entraînement supervisé image.
    '''),
    code('''
    data["duplicate_group"] = pd.factorize(data["phash"], sort=True)[0]
    group_stats = data.groupby("duplicate_group").agg(
        group_size=("Id", "size"),
        label_count=("prdtypecode", "nunique"),
    )
    data = data.join(group_stats, on="duplicate_group")
    data["label_conflict"] = data["label_count"] > 1
    data["eligible"] = data["readable"] & ~data["uniform_image"] & ~data["label_conflict"]

    audit = pd.Series({
        "lignes_total": len(data),
        "images_illisibles": (~data["readable"]).sum(),
        "images_uniformes": data["uniform_image"].sum(),
        "lignes_dans_groupes_dupliques": (data["group_size"] > 1).sum(),
        "groupes_avec_labels_contradictoires": group_stats["label_count"].gt(1).sum(),
        "lignes_eligibles": data["eligible"].sum(),
    })
    display(audit.to_frame("valeur"))
    '''),
    md('''
    ## 4. Split stratifié et groupé

    `StratifiedGroupKFold` préserve au mieux les proportions de classes tout en garantissant qu'un groupe
    de doublons ne traverse jamais la frontière train/validation. Le fold 0 est utilisé comme validation.

    Le `sample_weight` inverse la taille du groupe : dix copies de la même image ne pèseront pas dix fois
    plus qu'une image unique. Les poids sont renormalisés autour de 1 pour garder une échelle de loss stable.
    '''),
    code('''
    from sklearn.model_selection import StratifiedGroupKFold

    model_data = data.loc[data["eligible"]].copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(splitter.split(
        model_data,
        y=model_data["prdtypecode"],
        groups=model_data["duplicate_group"],
    ))
    model_data["split"] = "train"
    model_data.loc[val_idx, "split"] = "validation"

    overlap = set(model_data.loc[train_idx, "duplicate_group"]) & set(model_data.loc[val_idx, "duplicate_group"])
    assert not overlap
    assert set(model_data["prdtypecode"].unique()) == set(model_data.loc[val_idx, "prdtypecode"].unique())

    model_data["sample_weight"] = 1.0 / model_data["group_size"]
    train_mean = model_data.loc[model_data["split"] == "train", "sample_weight"].mean()
    model_data["sample_weight"] /= train_mean

    split_table = pd.crosstab(model_data["prdtypecode"], model_data["split"])
    split_table["validation_pct"] = 100 * split_table["validation"] / split_table.sum(axis=1)
    display(split_table)
    '''),
    md('''
    ## 5. Sauvegarder le contrat de données

    Les codes métier ne sont pas nécessairement contigus. Le mapping les transforme en indices `0..26`
    pour la couche softmax et permet ensuite de revenir au `prdtypecode` original.
    '''),
    code('''
    columns = [
        "Id", "productid", "imageid", "image_path", "prdtypecode", "phash",
        "duplicate_group", "group_size", "sample_weight", "split",
    ]
    manifest = model_data[columns].copy()
    manifest.to_csv(ARTIFACTS_DIR / "image_manifest.csv", index=False)

    index_to_code = sorted(int(code) for code in manifest["prdtypecode"].unique())
    mapping = {"index_to_code": index_to_code}
    with open(ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, ensure_ascii=False)

    split_table.to_csv(ARTIFACTS_DIR / "split_summary.csv")
    data.to_csv(ARTIFACTS_DIR / "image_quality_audit.csv", index=False)
    print("Manifeste :", ARTIFACTS_DIR / "image_manifest.csv")
    print("Mapping   :", ARTIFACTS_DIR / "class_mapping.json")
    '''),
])


notebooks["01_cnn_from_scratch.ipynb"] = notebook([
    md('''
    # 01 — Baseline : CNN entraîné de zéro

    Ce notebook établit le **point de référence** du parcours image : un réseau convolutif simple, initialisé
    aléatoirement, qui n'apprend qu'à partir des images Rakuten. Son score, bon ou mauvais, mesure ce que les
    données permettent d'apprendre sans aucune connaissance préalable.

    Le protocole est strictement identique à celui du notebook 02 (transfer learning) :

    - même manifeste, même split groupé et mêmes poids d'échantillon (notebook 00) ;
    - même résolution 224 × 224, même augmentation, même perte, mêmes callbacks ;
    - même tête de classification (pooling global, dropout 0,30, softmax à 27 classes).

    Seuls l'architecture du corps convolutif et son initialisation changent. L'écart mesuré dans le
    notebook 03 sera donc attribuable au transfert de connaissances d'ImageNet.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    code(TRAINING_UTILS),
    md('''
    ## Architecture

    Le corps convolutif empile cinq blocs `Conv 3×3 → BatchNorm → ReLU` :

    | Bloc | Filtres | Sortie | Rôle |
    | --- | --- | --- | --- |
    | 1 | 32 (stride 2) + max pooling | 56 × 56 | réduit vite la résolution pour limiter le coût |
    | 2 | 64 + max pooling | 28 × 28 | motifs simples : bords, textures |
    | 3 | 128 + max pooling | 14 × 14 | motifs composés |
    | 4 | 256 | 14 × 14 | parties d'objets |
    | 5 | 256 | 14 × 14 | dernière couche convolutive, utilisée par Grad-CAM |

    - La couche `Rescaling(1/255)` est intégrée au modèle : le pipeline fournit des pixels dans `[0, 255]`,
      comme pour EfficientNet, et chaque modèle gère sa propre mise à l'échelle.
    - La BatchNorm stabilise l'apprentissage depuis une initialisation aléatoire.
    - Le corps est regroupé dans un sous-modèle nommé `features`, dont la sortie 14 × 14 × 256 sert de
      support aux cartes Grad-CAM du notebook 04.
    - Le réseau compte environ 1 million de paramètres, tous entraînables, contre environ 4 millions
      (pré-entraînés) pour EfficientNetB0.
    '''),
    code('''
    def conv_block(x, filters, name, strides=1, pool=True):
        x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)
        if pool:
            x = layers.MaxPooling2D(name=f"{name}_pool")(x)
        return x

    def build_cnn_scratch():
        image = keras.Input(shape=(*TARGET_SIZE, 3), name="pixels")
        x = layers.Rescaling(1.0 / 255, name="rescaling")(image)
        x = conv_block(x, 32, "block1", strides=2)
        x = conv_block(x, 64, "block2")
        x = conv_block(x, 128, "block3")
        x = conv_block(x, 256, "block4", pool=False)
        x = conv_block(x, 256, "block5", pool=False)
        features = keras.Model(image, x, name="features")

        inputs = keras.Input(shape=(*TARGET_SIZE, 3), name="image")
        x = build_augmentation()(inputs)
        x = features(x)
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = layers.Dropout(0.30, seed=SEED)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="prdtypecode")(x)
        return keras.Model(inputs, outputs, name="cnn_scratch")
    '''),
    code('''
    RUN_NAME = "cnn_scratch"
    RUN_DIR = ARTIFACTS_DIR / RUN_NAME
    train_ds = make_dataset(train_df, training=True, weighted=True)
    val_ds = make_dataset(val_df, training=False, weighted=False)

    model = build_cnn_scratch()
    compile_model(model, learning_rate=1e-3)
    model.get_layer("features").summary()
    model.summary()
    params = count_params(model)
    print(params)
    '''),
    md('''
    ## Entraînement

    - Adam avec un taux d'apprentissage de 1e-3, classique pour un réseau initialisé aléatoirement.
    - Jusqu'à 30 époques : partant de zéro, le réseau converge plus lentement qu'une tête posée sur des
      représentations déjà apprises. L'arrêt anticipé (patience 5) coupe l'entraînement dès que la perte de
      validation ne progresse plus.
    - Le meilleur modèle selon `val_loss` est sauvegardé dans `cnn_scratch/best_model.keras`.
    '''),
    code('''
    EPOCHS = 30
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_for(RUN_DIR, "best_model.keras", patience=5, append=False),
    )
    train_time_min = (time.perf_counter() - start) / 60
    model.save(RUN_DIR / "final_model.keras")
    print(f"Durée d'entraînement : {train_time_min:.1f} min")
    '''),
    md('''
    ## Courbes d'apprentissage

    L'écart entre les courbes d'entraînement et de validation indique un éventuel surapprentissage ; un
    plateau précoce de la validation indique au contraire une capacité limitée du modèle.
    '''),
    code('''
    plot_history(RUN_DIR / "training_log.csv", "CNN entraîné de zéro")
    '''),
    md('''
    ## Évaluation rapide sur la validation

    On recharge les poids du meilleur checkpoint (`best_model.keras`) avant d'évaluer. Le notebook 03
    réalise la comparaison complète avec les modèles transférés.
    '''),
    code('''
    model.load_weights(RUN_DIR / "best_model.keras")
    metrics, y_true, y_pred = evaluate_on_validation(model, val_ds, val_df)
    display(pd.Series(metrics).to_frame("validation").style.format("{:.4f}"))
    print(classification_report(y_true, y_pred, target_names=[str(code) for code in index_to_code]))

    save_run_summary(
        RUN_DIR,
        label="CNN entraîné de zéro",
        epochs_run=len(history.history["loss"]),
        train_time_min=round(train_time_min, 1),
        **params,
        validation=metrics,
    )
    '''),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


notebooks["02_transfer_learning_efficientnet.ipynb"] = notebook([
    md('''
    # 02 — Amélioration : transfer learning EfficientNetB0

    Ce notebook améliore la baseline du notebook 01 en réutilisant un réseau **pré-entraîné sur ImageNet**
    (1,3 million d'images, 1 000 classes). Ses couches ont déjà appris des contours, des textures et des formes
    génériques ; il reste à les adapter aux 27 catégories Rakuten.

    L'entraînement se fait en deux étapes, chacune sauvegardée et évaluée séparément :

    1. **Feature extraction** : le backbone est gelé, seule la nouvelle tête apprend
       (`feature_extraction_best.keras`) ;
    2. **Fine-tuning** : les 30 dernières couches du backbone sont dégelées et ajustées avec un taux
       d'apprentissage très faible (`best_model.keras`).

    Le protocole (données, split, augmentation, perte, callbacks) est identique à celui du notebook 01.
    EfficientNet Keras intègre sa propre mise à l'échelle et attend des pixels dans `[0, 255]`.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    code(TRAINING_UTILS),
    md('''
    ## Architecture

    - **Backbone** : EfficientNetB0 sans sa tête ImageNet (`include_top=False`), soit environ 4 millions de
      paramètres. Sa sortie est une carte 7 × 7 × 1280.
    - **Tête** : identique à celle du CNN du notebook 01 (pooling global, dropout 0,30, softmax à 27 classes).
    - Le backbone est appelé avec `training=False` : ses couches BatchNorm restent en mode inférence, même
      une fois dégelées, pour ne pas dégrader leurs statistiques ImageNet avec des lots de 32 images.
    '''),
    code('''
    def build_efficientnet():
        inputs = keras.Input(shape=(*TARGET_SIZE, 3), name="image")
        x = build_augmentation()(inputs)
        local_weights = ARTIFACTS_DIR / ".keras" / "models" / "efficientnetb0_notop.h5"
        weights = str(local_weights) if local_weights.exists() else "imagenet"
        print("Poids EfficientNet :", weights)
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=(*TARGET_SIZE, 3),
        )
        backbone.trainable = False
        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = layers.Dropout(0.30, seed=SEED)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="prdtypecode")(x)
        return keras.Model(inputs, outputs, name="efficientnet_transfer"), backbone

    def unfreeze_last_layers(backbone, count=30):
        backbone.trainable = True
        for layer in backbone.layers[:-count]:
            layer.trainable = False
        # Les statistiques de BatchNorm restent figées sur ce dataset relativement petit.
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    '''),
    code('''
    RUN_NAME = "efficientnet_transfer"
    RUN_DIR = ARTIFACTS_DIR / RUN_NAME
    train_ds = make_dataset(train_df, training=True, weighted=True)
    val_ds = make_dataset(val_df, training=False, weighted=False)

    model, backbone = build_efficientnet()
    compile_model(model, learning_rate=1e-3)
    model.summary()
    params_frozen = count_params(model)
    print(params_frozen)
    '''),
    md('''
    ## Étape 1 — Feature extraction (backbone gelé)

    Seule la tête apprend (environ 35 000 paramètres). Elle est initialisée aléatoirement : si le backbone était
    libre dès le départ, ses premiers gradients, importants, détruiraient les représentations ImageNet. Un taux
    d'apprentissage de 1e-3 convient, puisque seuls des poids neufs sont ajustés.
    '''),
    code('''
    EPOCHS_FROZEN = 8
    start = time.perf_counter()
    history_frozen = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks_for(RUN_DIR, "feature_extraction_best.keras", patience=4, append=False),
    )
    time_frozen_min = (time.perf_counter() - start) / 60

    # Le fine-tuning repart du meilleur état de l'étape 1, qui est aussi celui évalué ici.
    model.load_weights(RUN_DIR / "feature_extraction_best.keras")
    metrics_frozen, _, _ = evaluate_on_validation(model, val_ds, val_df)
    display(pd.Series(metrics_frozen).to_frame("feature extraction").style.format("{:.4f}"))
    save_run_summary(
        RUN_DIR,
        label="EfficientNetB0 transfer learning",
        feature_extraction={
            "epochs_run": len(history_frozen.history["loss"]),
            "train_time_min": round(time_frozen_min, 1),
            **params_frozen,
            "validation": metrics_frozen,
        },
    )
    '''),
    md('''
    ## Étape 2 — Fine-tuning des 30 dernières couches

    - **Couches dégelées** : les 30 dernières couches du backbone, les plus profondes. Elles encodent des
      motifs spécifiques à ImageNet, qu'il est utile d'adapter aux produits Rakuten. Les premières couches,
      qui détectent des motifs génériques (bords, textures), restent gelées.
    - **BatchNorm** : maintenue gelée, pour la raison donnée plus haut.
    - **Taux d'apprentissage** : 1e-5, cent fois plus faible qu'à l'étape 1. Un pas trop grand effacerait
      les représentations pré-entraînées (oubli catastrophique).
    '''),
    code('''
    unfreeze_last_layers(backbone, count=30)
    compile_model(model, learning_rate=1e-5)

    unfrozen = [layer.name for layer in backbone.layers if layer.trainable]
    print(f"{len(unfrozen)} couches dégelées (hors BatchNorm) :")
    print(", ".join(unfrozen))
    params_finetune = count_params(model)
    print(params_finetune)
    '''),
    code('''
    start_epoch = len(history_frozen.history["loss"])
    start = time.perf_counter()
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=start_epoch,
        epochs=start_epoch + 12,
        callbacks=callbacks_for(RUN_DIR, "best_model.keras", patience=4, append=True),
    )
    time_finetune_min = (time.perf_counter() - start) / 60
    model.save(RUN_DIR / "final_model.keras")
    print(f"Durée du fine-tuning : {time_finetune_min:.1f} min")
    '''),
    md('''
    ## Courbes d'apprentissage

    La ligne pointillée marque le passage au fine-tuning. Un saut de la validation juste après cette ligne
    montre ce qu'apporte l'adaptation du backbone.
    '''),
    code('''
    plot_history(RUN_DIR / "training_log.csv", "EfficientNetB0 — transfer learning", phase_boundary=start_epoch)
    '''),
    md('''## Évaluation rapide du modèle fine-tuné'''),
    code('''
    model.load_weights(RUN_DIR / "best_model.keras")
    metrics_finetune, y_true, y_pred = evaluate_on_validation(model, val_ds, val_df)
    display(pd.DataFrame({
        "feature extraction": metrics_frozen,
        "fine-tuning": metrics_finetune,
    }).style.format("{:.4f}"))
    print(classification_report(y_true, y_pred, target_names=[str(code) for code in index_to_code]))

    save_run_summary(
        RUN_DIR,
        fine_tuning={
            "epochs_run": len(history_finetune.history["loss"]),
            "train_time_min": round(time_finetune_min, 1),
            "unfrozen_layers": len(unfrozen),
            **params_finetune,
            "validation": metrics_finetune,
        },
    )
    '''),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


notebooks["03_evaluer_comparer_et_predire.ipynb"] = notebook([
    md('''
    # 03 — Comparer la baseline et les modèles transférés

    Ce notebook évalue sur **exactement les mêmes 16 475 images de validation** :

    1. le CNN entraîné de zéro (notebook 01, baseline) ;
    2. EfficientNetB0 avec backbone gelé (notebook 02, étape 1) ;
    3. EfficientNetB0 fine-tuné (notebook 02, étape 2) ;
    4. si disponible, la variante CLAHE du premier parcours (ablation, `archive_v1`).

    Il produit le tableau de progression, les courbes d'apprentissage, l'analyse par classe, la matrice de
    confusion du meilleur modèle, ses erreurs les plus confiantes et, si `data/raw_x_test.csv` existe, les
    prédictions du jeu de test. Le meilleur modèle est choisi sur le F1 pondéré ; le F1 macro est surveillé
    pour ne pas sacrifier les petites classes.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    code(CATEGORY_NAMES),
    code(MODEL_LOADING),
    code('''
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score
    '''),
    md('''
    ## 1. Prédictions sur la validation

    Les probabilités sont sauvegardées dans `predictions/` pour être réutilisées par le notebook 04.
    '''),
    code('''
    required = {"cnn_scratch", "efficientnet_finetuned"}
    if not required.issubset(experiments):
        raise FileNotFoundError("Exécuter d'abord les notebooks 01 et 02 : " + ", ".join(sorted(required - set(experiments))))
    if "efficientnet_frozen" not in experiments:
        print("Checkpoint de feature extraction absent : exécuter le notebook 02 pour compléter la progression.")

    y_true = val_df["class_index"].to_numpy()
    predictions = {}
    loaded_models = {}
    for name, run in experiments.items():
        print(f"\\n{run['label']}")
        model = load_evaluation_model(run["path"])
        ds = make_dataset(val_df, transform=run["transform"], weighted=False)
        probabilities = model.predict(ds, verbose=1)
        predictions[name] = {"probabilities": probabilities, "labels": probabilities.argmax(axis=1)}
        loaded_models[name] = model
        np.savez_compressed(
            PREDICTIONS_DIR / f"val_{name}.npz", Id=val_df["Id"].to_numpy(), probabilities=probabilities
        )
    '''),
    md('''
    ## 2. Tableau de progression

    Les paramètres et durées proviennent des fichiers `run_summary.json` écrits par les notebooks 01 et 02
    (absents pour les runs du premier parcours).
    '''),
    code('''
    def run_info(name):
        path = experiments[name]["run_dir"] / "run_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if name == "efficientnet_frozen":
            summary = summary.get("feature_extraction", {})
        elif name == "efficientnet_finetuned" and "fine_tuning" in summary:
            frozen_time = summary.get("feature_extraction", {}).get("train_time_min", 0)
            summary = dict(summary["fine_tuning"])
            summary["train_time_min"] = round(summary["train_time_min"] + frozen_time, 1)
        elif name == "efficientnet_finetuned":
            summary = {}
        return summary

    rows = []
    for name, result in predictions.items():
        info = run_info(name)
        rows.append({
            "experiment": name,
            "modèle": experiments[name]["label"],
            "accuracy": accuracy_score(y_true, result["labels"]),
            "top3_accuracy": top_k_accuracy_score(
                y_true, result["probabilities"], k=3, labels=np.arange(NUM_CLASSES)
            ),
            "f1_weighted": f1_score(y_true, result["labels"], average="weighted"),
            "f1_macro": f1_score(y_true, result["labels"], average="macro"),
            "paramètres": loaded_models[name].count_params(),
            "paramètres entraînés": info.get("params_trainable"),
            "durée (min)": info.get("train_time_min"),
        })
    metrics = pd.DataFrame(rows).set_index("experiment")
    metrics.to_csv(ARTIFACTS_DIR / "model_comparison.csv")
    display(metrics.style.format({
        "accuracy": "{:.4f}", "top3_accuracy": "{:.4f}", "f1_weighted": "{:.4f}", "f1_macro": "{:.4f}",
        "paramètres": "{:,.0f}", "paramètres entraînés": "{:,.0f}", "durée (min)": "{:.0f}",
    }, na_rep="–"))

    CHAMPION = metrics["f1_weighted"].idxmax()
    BASELINE = "cnn_scratch"
    gain = metrics.loc[CHAMPION, "f1_weighted"] - metrics.loc[BASELINE, "f1_weighted"]
    print(f"Champion selon le F1 pondéré : {experiments[CHAMPION]['label']}")
    print(f"Gain par rapport à la baseline : {gain:+.4f} de F1 pondéré")
    '''),
    code('''
    plot_metrics = metrics[["accuracy", "f1_weighted", "f1_macro"]].rename(index=lambda name: experiments[name]["label"])
    ax = plot_metrics.plot.barh(figsize=(10, 1.2 + 0.9 * len(plot_metrics)), width=0.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score sur la validation")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.set_title("Progression : de la baseline au modèle fine-tuné")
    plt.tight_layout()
    plt.show()
    '''),
    md('''
    ## 3. Courbes d'apprentissage

    Accuracy de validation par époque. La ligne pointillée marque le début du fine-tuning d'EfficientNet.
    '''),
    code('''
    fig, ax = plt.subplots(figsize=(10, 4.5))
    curves = {
        "CNN entraîné de zéro": experiments["cnn_scratch"]["run_dir"],
        "EfficientNetB0 (gelé puis fine-tuné)": experiments["efficientnet_finetuned"]["run_dir"],
    }
    for label, run_dir in curves.items():
        log_path = run_dir / "training_log.csv"
        # Relancer un notebook d'entraînement vide ce journal dès la première époque.
        if not log_path.exists() or log_path.stat().st_size == 0:
            print(f"Journal absent ou vide, courbe ignorée : {log_path}")
            continue
        log = pd.read_csv(log_path)
        ax.plot(log["epoch"] + 1, log["val_accuracy"], marker="o", label=label)
    transfer_summary = experiments["efficientnet_finetuned"]["run_dir"] / "run_summary.json"
    if transfer_summary.exists():
        boundary = json.loads(transfer_summary.read_text())["feature_extraction"]["epochs_run"]
        ax.axvline(boundary + 0.5, color="grey", linestyle="--", label="début du fine-tuning")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy de validation")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    '''),
    md('''
    ## 4. Analyse par classe

    Le gain du transfert n'est pas uniforme : la comparaison par classe montre quelles catégories en
    profitent le plus, et lesquelles restent difficiles même pour le meilleur modèle.
    '''),
    code('''
    per_class = pd.DataFrame({
        "prdtypecode": index_to_code,
        "category_name": [category_names[code] for code in index_to_code],
        "support": np.bincount(y_true, minlength=NUM_CLASSES),
    })
    for name, result in predictions.items():
        per_class[f"f1_{name}"] = f1_score(
            y_true, result["labels"], labels=np.arange(NUM_CLASSES), average=None, zero_division=0
        )
    per_class["gain_transfert"] = per_class[f"f1_{CHAMPION}"] - per_class["f1_cnn_scratch"]
    per_class = per_class.sort_values("gain_transfert", ascending=False)
    per_class.to_csv(ARTIFACTS_DIR / "per_class_comparison.csv", index=False)
    display(per_class.style.format({column: "{:.3f}" for column in per_class.columns if column.startswith(("f1_", "gain"))}))
    '''),
    code('''
    ordered = per_class.sort_values(f"f1_{CHAMPION}")
    labels = [f"{code} — {name}" for code, name in zip(ordered["prdtypecode"], ordered["category_name"])]
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(positions - 0.2, ordered["f1_cnn_scratch"], height=0.4, label=experiments["cnn_scratch"]["label"])
    ax.barh(positions + 0.2, ordered[f"f1_{CHAMPION}"], height=0.4, label=experiments[CHAMPION]["label"])
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 sur la validation")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("F1 par catégorie : baseline contre meilleur modèle")
    plt.tight_layout()
    plt.show()
    '''),
    md('''## 5. Matrice de confusion et confusions principales du meilleur modèle'''),
    code('''
    champion_pred = predictions[CHAMPION]["labels"]
    matrix = confusion_matrix(y_true, champion_pred, labels=np.arange(NUM_CLASSES), normalize="true")
    plt.figure(figsize=(22, 18))
    sns.heatmap(
        matrix, cmap="Blues", vmin=0, vmax=1,
        xticklabels=readable_class_labels, yticklabels=readable_class_labels,
    )
    plt.title(f"Matrice de confusion normalisée — {experiments[CHAMPION]['label']}")
    plt.xlabel("Catégorie prédite")
    plt.ylabel("Catégorie réelle")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    counts = confusion_matrix(y_true, champion_pred, labels=np.arange(NUM_CLASSES))
    pairs = [
        {
            "catégorie réelle": readable_class_labels[i],
            "prédite à tort": readable_class_labels[j],
            "images": int(counts[i, j]),
            "part de la classe réelle": matrix[i, j],
        }
        for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j
    ]
    top_confusions = pd.DataFrame(pairs).sort_values("images", ascending=False).head(10)
    top_confusions.to_csv(ARTIFACTS_DIR / "top_confusions.csv", index=False)
    display(top_confusions.style.format({"part de la classe réelle": "{:.1%}"}).hide(axis="index"))
    '''),
    md('''
    ## 6. Erreurs les plus confiantes

    Ces exemples révèlent souvent des labels ambigus, des images peu informatives ou des confusions métier
    récurrentes. Le notebook 04 montre, avec Grad-CAM, sur quelles zones le modèle s'est appuyé.
    '''),
    code('''
    champion_proba = predictions[CHAMPION]["probabilities"]
    errors = val_df.copy().reset_index(drop=True)
    errors["true_index"] = y_true
    errors["pred_index"] = champion_pred
    errors["confidence"] = champion_proba.max(axis=1)
    errors = errors.query("true_index != pred_index").sort_values("confidence", ascending=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 12))
    for ax, row in zip(axes.flat, errors.head(12).itertuples()):
        ax.imshow(plt.imread(row.image_path))
        ax.set_title(
            f"vrai : {short_label(row.true_index)}\\n"
            f"prédit : {short_label(row.pred_index)}\\n"
            f"confiance {row.confidence:.0%}", fontsize=8,
        )
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    '''),
    md('''
    ## 7. Prédire le jeu de test (optionnel)

    Cette section attend `data/raw_x_test.csv` avec les colonnes `Id`, `imageid` et `productid`. Elle produit
    `submission_image.csv` au format `Id,prdtypecode` et `predictions_image_labeled.csv` avec le libellé métier.
    '''),
    code('''
    TEST_CSV = DATA_DIR / "raw_x_test.csv"
    if not TEST_CSV.exists():
        print("Étape ignorée : ajouter", TEST_CSV, "pour générer les prédictions de test.")
    else:
        test = pd.read_csv(TEST_CSV)
        required_columns = {"Id", "imageid", "productid"}
        assert required_columns.issubset(test.columns), required_columns - set(test.columns)
        test["image_path"] = test.apply(
            lambda row: str((ROOT / "images" / "image_test" /
                f"image_{row.imageid}_product_{row.productid}.jpg").resolve()), axis=1
        )
        assert test["image_path"].map(os.path.isfile).all()
        test["class_index"] = 0
        test["sample_weight"] = 1.0

        champion_run = experiments[CHAMPION]
        test_ds = make_dataset(test, transform=champion_run["transform"], weighted=False)
        test_probabilities = loaded_models[CHAMPION].predict(test_ds, verbose=1)
        test_codes = [index_to_code[index] for index in test_probabilities.argmax(axis=1)]
        submission = pd.DataFrame({"Id": test["Id"], "prdtypecode": test_codes})
        submission.to_csv(ARTIFACTS_DIR / "submission_image.csv", index=False)

        labeled_predictions = submission.copy()
        labeled_predictions["category_name"] = labeled_predictions["prdtypecode"].map(category_names)
        labeled_predictions.to_csv(ARTIFACTS_DIR / "predictions_image_labeled.csv", index=False)
        display(labeled_predictions.head())
    '''),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


notebooks["04_interpretabilite_gradcam.ipynb"] = notebook([
    md('''
    # 04 — Interprétabilité : Grad-CAM

    Un bon score ne garantit pas que le modèle regarde le produit : il peut s'appuyer sur un fond, un
    logo ou un artefact récurrent. Ce notebook ouvre la boîte noire avec **Grad-CAM** et compare ce que
    regardent la baseline (CNN de zéro) et le modèle fine-tuné.

    **Principe de Grad-CAM** (Selvaraju et al., 2017) :

    1. on calcule le score de la classe étudiée, avant le softmax ;
    2. on dérive ce score par rapport aux cartes d'activation de la dernière couche convolutive ;
    3. la moyenne spatiale de ces gradients donne le poids de chaque canal pour cette classe ;
    4. la somme pondérée des cartes, passée dans une ReLU, localise les zones qui **augmentent** le score ;
    5. la carte est normalisée entre 0 et 1, puis agrandie à 224 × 224 et superposée à l'image.

    La dernière couche convolutive offre le meilleur compromis entre sens sémantique et résolution spatiale.
    Cette résolution diffère selon le modèle : 14 × 14 pour le CNN de zéro, 7 × 7 pour EfficientNetB0. Les
    cartes d'EfficientNet sont donc plus grossières.

    Le notebook se termine par un **test de fidélité** : masquer les zones désignées par Grad-CAM doit faire
    chuter la confiance du modèle bien plus que masquer d'autres zones de même taille.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    code(CATEGORY_NAMES),
    code(MODEL_LOADING),
    md('''## 1. Charger les modèles et leurs prédictions'''),
    code('''
    GRADCAM_MODELS = ["cnn_scratch", "efficientnet_finetuned"]
    missing = [name for name in GRADCAM_MODELS if name not in experiments]
    if missing:
        raise FileNotFoundError(f"Modèles absents : {missing}. Exécuter les notebooks 01 et 02.")

    models = {name: load_evaluation_model(experiments[name]["path"]) for name in GRADCAM_MODELS}
    labels = {name: experiments[name]["label"] for name in GRADCAM_MODELS}

    val = val_df.reset_index(drop=True)
    y_true = val["class_index"].to_numpy()
    probabilities = {}
    for name in GRADCAM_MODELS:
        cached = PREDICTIONS_DIR / f"val_{name}.npz"
        if cached.exists() and np.array_equal(np.load(cached)["Id"], val["Id"].to_numpy()):
            probabilities[name] = np.load(cached)["probabilities"]
        else:
            probabilities[name] = models[name].predict(make_dataset(val), verbose=1)
        val[f"pred_{name}"] = probabilities[name].argmax(axis=1)
        val[f"conf_{name}"] = probabilities[name].max(axis=1)
        print(f"{labels[name]} : accuracy {np.mean(val[f'pred_{name}'] == y_true):.3f}")
    '''),
    md('''
    ## 2. Implémentation

    Le modèle est découpé en deux : le corps convolutif (`features` pour le CNN, `efficientnetb0` pour le
    modèle transféré), puis la tête. Le score avant softmax est recalculé à partir des poids de la couche
    dense, ce qui évite la saturation du softmax. L'augmentation est ignorée : elle est inactive en
    inférence.
    '''),
    code('''
    def load_images(paths):
        return np.stack([
            decode_resize(path, 0, 1.0)[0].numpy() for path in paths
        ])

    def gradcam(name, images, class_indices=None):
        """Retourne les cartes Grad-CAM (N, 224, 224), les probabilités et les classes expliquées."""
        model = models[name]
        feature_model = model.get_layer(experiments[name]["gradcam_layer"])
        pooling = model.get_layer("global_average_pooling")
        classifier = model.get_layer("prdtypecode")
        x = tf.convert_to_tensor(images, tf.float32)
        with tf.GradientTape() as tape:
            feature_maps = feature_model(x, training=False)
            tape.watch(feature_maps)
            logits = tf.matmul(pooling(feature_maps), classifier.kernel) + classifier.bias
            if class_indices is None:
                class_indices = tf.argmax(logits, axis=1)
            class_indices = tf.cast(class_indices, tf.int32)
            scores = tf.gather(logits, class_indices, batch_dims=1)
        gradients = tape.gradient(scores, feature_maps)
        channel_weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        cams = tf.nn.relu(tf.reduce_sum(channel_weights * feature_maps, axis=-1))
        cams = cams / (tf.reduce_max(cams, axis=(1, 2), keepdims=True) + 1e-8)
        cams = tf.image.resize(cams[..., None], TARGET_SIZE, method="bilinear")[..., 0]
        return cams.numpy(), tf.nn.softmax(logits).numpy(), class_indices.numpy()

    def show_cam(ax, image, cam, title):
        ax.imshow(image.astype("uint8"))
        ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    def prediction_title(name, row):
        pred = getattr(row, f"pred_{name}")
        mark = "✓" if pred == row.class_index else "✗"
        return f"{labels[name]}\\n{mark} {short_label(pred)} ({getattr(row, f'conf_{name}'):.0%})"

    def compare_models(rows, title):
        rows = list(rows.itertuples())
        if not rows:
            print(f"Aucune image pour : {title}")
            return
        images = load_images([row.image_path for row in rows])
        cams = {name: gradcam(name, images)[0] for name in GRADCAM_MODELS}
        fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)))
        axes = np.atleast_2d(axes)
        for i, row in enumerate(rows):
            axes[i, 0].imshow(images[i].astype("uint8"))
            axes[i, 0].set_title(f"Image — vraie classe\\n{short_label(row.class_index)}", fontsize=8)
            axes[i, 0].axis("off")
            for j, name in enumerate(GRADCAM_MODELS, start=1):
                show_cam(axes[i, j], images[i], cams[name][i], prediction_title(name, row))
        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
        plt.show()
    '''),
    md('''
    ## 3. Les deux modèles ont raison : regardent-ils la même chose ?

    Huit images de catégories différentes, bien classées par les deux modèles. On attend du modèle
    fine-tuné qu'il se concentre davantage sur l'objet, là où le CNN de zéro peut s'appuyer sur des
    indices diffus (fond, couleur dominante, bords de l'image).
    '''),
    code('''
    both_right = val[(val["pred_cnn_scratch"] == y_true) & (val["pred_efficientnet_finetuned"] == y_true)]
    sample_both = (
        both_right.sample(frac=1, random_state=SEED)
        .groupby("class_index").head(1)
        .head(8)
    )
    compare_models(sample_both, "Images bien classées par les deux modèles")
    '''),
    md('''
    ## 4. Ce que le transfert corrige

    Images mal classées par le CNN de zéro mais bien classées par le modèle fine-tuné. La comparaison
    des cartes montre l'indice que la baseline a manqué.
    '''),
    code('''
    fixed = val[(val["pred_cnn_scratch"] != y_true) & (val["pred_efficientnet_finetuned"] == y_true)]
    print(f"{len(fixed):,} images corrigées par le transfert ({len(fixed) / len(val):.1%} de la validation)")
    broken = val[(val["pred_cnn_scratch"] == y_true) & (val["pred_efficientnet_finetuned"] != y_true)]
    print(f"{len(broken):,} images dégradées par le transfert ({len(broken) / len(val):.1%} de la validation)")
    sample_fixed = fixed.sort_values("conf_efficientnet_finetuned", ascending=False).groupby(
        "class_index", group_keys=False).head(1).head(6)
    compare_models(sample_fixed, "Erreurs de la baseline corrigées par le transfer learning")
    '''),
    md('''
    ## 5. Erreurs confiantes du modèle fine-tuné

    Pour chaque erreur, deux cartes : les zones qui soutiennent la classe **prédite** et celles qui
    soutiennent la classe **réelle**. Si les deux pointent vers le même objet, l'image est ambiguë entre
    les deux catégories ; si la classe prédite s'appuie sur un détail secondaire, le modèle a suivi un
    indice trompeur.
    '''),
    code('''
    name = "efficientnet_finetuned"
    wrong = val[val[f"pred_{name}"] != y_true].sort_values(f"conf_{name}", ascending=False)
    wrong = wrong.groupby("class_index", group_keys=False).head(1).head(6)
    images = load_images(wrong["image_path"])
    cams_pred, probs, _ = gradcam(name, images)
    cams_true, _, _ = gradcam(name, images, wrong["class_index"].to_numpy())

    fig, axes = plt.subplots(len(wrong), 3, figsize=(11, 3.6 * len(wrong)))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(wrong.itertuples()):
        pred = getattr(row, f"pred_{name}")
        axes[i, 0].imshow(images[i].astype("uint8"))
        axes[i, 0].set_title("Image", fontsize=8)
        axes[i, 0].axis("off")
        show_cam(axes[i, 1], images[i], cams_pred[i],
                 f"Classe prédite : {short_label(pred)}\\n(probabilité {probs[i, pred]:.0%})")
        show_cam(axes[i, 2], images[i], cams_true[i],
                 f"Classe réelle : {short_label(row.class_index)}\\n(probabilité {probs[i, row.class_index]:.0%})")
    fig.suptitle("Erreurs confiantes : indices pour la classe prédite et pour la classe réelle", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    plt.show()
    '''),
    md('''
    ## 6. Une carte par catégorie

    Pour chaque catégorie, l'image la mieux reconnue par le modèle fine-tuné. Ces cartes résument les
    indices visuels typiques de chaque classe : couverture d'un livre, forme d'un meuble, emballage, etc.
    '''),
    code('''
    best_per_class = (
        val[val["pred_efficientnet_finetuned"] == y_true]
        .sort_values("conf_efficientnet_finetuned", ascending=False)
        .groupby("class_index").head(1)
        .sort_values("class_index")
    )
    images = load_images(best_per_class["image_path"])
    cams, _, _ = gradcam("efficientnet_finetuned", images)
    fig, axes = plt.subplots(5, 6, figsize=(18, 16))
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, cam, row in zip(axes.flat, images, cams, best_per_class.itertuples()):
        show_cam(ax, image, cam, f"{short_label(row.class_index)}\\n({row.conf_efficientnet_finetuned:.0%})")
    fig.suptitle("Grad-CAM du modèle fine-tuné — exemple le plus confiant de chaque catégorie", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    plt.show()
    '''),
    md('''
    ## 7. Test de fidélité : les zones Grad-CAM comptent-elles vraiment ?

    Une carte peut être jolie sans refléter le raisonnement du modèle. Pour le vérifier, on masque 20 % des
    pixels de chaque image (remplacés par la couleur moyenne de l'image) selon trois stratégies, puis on
    mesure la chute de probabilité de la classe initialement prédite :

    - **zones les plus importantes** selon Grad-CAM ;
    - **zones Grad-CAM d'une autre image** : même forme de masque, mais placée sans lien avec l'image, ce
      qui sert de témoin ;
    - **zones les moins importantes** selon Grad-CAM.

    Si Grad-CAM est fidèle, la première stratégie doit faire chuter la confiance bien plus que les deux
    autres. L'échantillon compte jusqu'à 20 images par catégorie.
    '''),
    code('''
    MASK_FRACTION = 0.20
    rng = np.random.default_rng(SEED)
    fidelity_sample = val.sample(frac=1, random_state=SEED).groupby("class_index").head(20)

    def mask_by_cam(images, cams, mode):
        flat = cams.reshape(len(cams), -1) + rng.uniform(0, 1e-6, size=(len(cams), cams[0].size))
        k = int(MASK_FRACTION * flat.shape[1])
        order = np.argsort(-flat if mode == "top" else flat, axis=1)[:, :k]
        mask = np.zeros_like(flat, dtype=bool)
        np.put_along_axis(mask, order, True, axis=1)
        mask = mask.reshape(cams.shape)[..., None]
        fill = images.mean(axis=(1, 2), keepdims=True)
        return np.where(mask, fill, images)

    fidelity_rows = []
    for name in GRADCAM_MODELS:
        for start in range(0, len(fidelity_sample), BATCH_SIZE):
            batch = fidelity_sample.iloc[start:start + BATCH_SIZE]
            images = load_images(batch["image_path"])
            cams, probs, classes = gradcam(name, images)
            base = probs[np.arange(len(batch)), classes]
            variants = {
                "zones les plus importantes": mask_by_cam(images, cams, "top"),
                "zones d'une autre image (témoin)": mask_by_cam(images, np.roll(cams, 1, axis=0), "top"),
                "zones les moins importantes": mask_by_cam(images, cams, "bottom"),
            }
            for strategy, masked in variants.items():
                masked_probs = models[name].predict_on_batch(masked)
                masked_probs = np.asarray(masked_probs)
                fidelity_rows.append(pd.DataFrame({
                    "modèle": labels[name],
                    "masque": strategy,
                    "chute de confiance": base - masked_probs[np.arange(len(batch)), classes],
                    "prédiction changée": masked_probs.argmax(axis=1) != classes,
                }))

    fidelity = pd.concat(fidelity_rows, ignore_index=True)
    fidelity_summary = fidelity.groupby(["modèle", "masque"], sort=False).mean()
    fidelity_summary.to_csv(ARTIFACTS_DIR / "gradcam_fidelity.csv")
    display(fidelity_summary.style.format({"chute de confiance": "{:+.3f}", "prédiction changée": "{:.1%}"}))
    print(f"Images testées par modèle : {len(fidelity_sample)}")
    '''),
    code('''
    from matplotlib.ticker import PercentFormatter

    ax = fidelity_summary["prédiction changée"].unstack("masque").plot.barh(figsize=(10, 3.8), width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Part des prédictions changées après masquage de 20 % des pixels")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt=lambda value: f"{value:.0%}", fontsize=8, padding=2)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Fidélité de Grad-CAM")
    plt.tight_layout()
    plt.show()
    '''),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


def has_outputs(path: Path) -> bool:
    content = json.loads(path.read_text(encoding="utf-8"))
    return any(cell.get("outputs") for cell in content["cells"] if cell["cell_type"] == "code")


if __name__ == "__main__":
    # Usage : python tools/build_image_classification_notebooks.py [--force] [notebook.ipynb ...]
    force = "--force" in sys.argv
    selected = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    unknown = set(selected) - set(notebooks)
    if unknown:
        sys.exit(f"Notebooks inconnus : {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, content in notebooks.items():
        if selected and filename not in selected:
            continue
        target = OUT / filename
        if target.exists() and has_outputs(target) and not force:
            # Ne jamais effacer des résultats d'entraînement par une simple régénération.
            print("Ignoré (contient des sorties, utiliser --force) :", target)
            continue
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=1)
        print("Écrit :", target)
