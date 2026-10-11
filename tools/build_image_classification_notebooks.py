"""Génère les notebooks du parcours de classification supervisée des images."""

from __future__ import annotations

import json
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
    image = tf.cast(image, tf.float32)  # EfficientNet attend des pixels dans [0, 255].
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


MODEL_FUNCTIONS = r'''
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
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
    return keras.Model(inputs, outputs), backbone

def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

def callbacks_for(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            run_dir / "best_model.keras", monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
        keras.callbacks.CSVLogger(run_dir / "training_log.csv", append=True),
    ]

def unfreeze_last_layers(backbone, count=30):
    backbone.trainable = True
    for layer in backbone.layers[:-count]:
        layer.trainable = False
    # Les statistiques de BatchNorm restent figées sur ce dataset relativement petit.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
'''


TRAINING_CELLS = r'''
RUN_DIR = ARTIFACTS_DIR / RUN_NAME
train_ds = make_dataset(train_df, training=True, transform=TRAIN_TRANSFORM, weighted=True)
val_ds = make_dataset(val_df, training=False, transform=VAL_TRANSFORM, weighted=False)
run_callbacks = callbacks_for(RUN_DIR)

model, backbone = build_model()
compile_model(model, learning_rate=1e-3)
model.summary()
'''


TRAIN_EXECUTION = r'''
EPOCHS_FROZEN = 8
history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=run_callbacks,
)
'''


FINETUNE_EXECUTION = r'''
unfreeze_last_layers(backbone, count=30)
compile_model(model, learning_rate=1e-5)

start_epoch = len(history_frozen.history["loss"])
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=start_epoch,
    epochs=start_epoch + 12,
    callbacks=run_callbacks,
)

model.save(RUN_DIR / "final_model.keras")
print("Modèle sauvegardé dans", RUN_DIR)
'''


QUICK_EVAL = r'''
from sklearn.metrics import classification_report, f1_score

best_model = keras.models.load_model(RUN_DIR / "best_model.keras")
probabilities = best_model.predict(val_ds, verbose=1)
y_pred = probabilities.argmax(axis=1)
y_true = val_df["class_index"].to_numpy()

print("F1 weighted :", f1_score(y_true, y_pred, average="weighted"))
print("F1 macro    :", f1_score(y_true, y_pred, average="macro"))
print(classification_report(y_true, y_pred, target_names=[str(x) for x in index_to_code]))
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


notebooks["01_entrainer_efficientnet_baseline.ipynb"] = notebook([
    md('''
    # 01 — Baseline EfficientNetB0

    Ce notebook entraîne un classifieur à 27 classes par **transfer learning** :

    1. apprentissage de la tête de classification avec le backbone ImageNet gelé ;
    2. fine-tuning des 30 dernières couches avec un learning rate faible ;
    3. sauvegarde automatique du meilleur modèle selon la loss de validation.

    Aucun CLAHE n'est appliqué : ce run constitue la référence expérimentale. EfficientNet Keras inclut
    déjà sa couche de rescaling et attend des pixels flottants dans l'intervalle `[0, 255]`.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    md('''
    ## Pipeline et architecture

    Le redimensionnement conserve le ratio et ajoute du padding. L'augmentation reste modérée ; le flip
    horizontal est volontairement absent car beaucoup d'images contiennent du texte, des jaquettes ou des logos.
    '''),
    code(MODEL_FUNCTIONS),
    code('''
    RUN_NAME = "efficientnet_baseline"
    TRAIN_TRANSFORM = None
    VAL_TRANSFORM = None
    '''),
    code(TRAINING_CELLS),
    md('''## Phase 1 — Backbone gelé'''),
    code(TRAIN_EXECUTION),
    md('''## Phase 2 — Fine-tuning partiel'''),
    code(FINETUNE_EXECUTION),
    md('''## Contrôle rapide

    Le notebook 03 réalisera la comparaison complète. Cette cellule permet déjà de détecter un run défaillant.
    '''),
    code(QUICK_EVAL),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


notebooks["02_entrainer_efficientnet_clahe.ipynb"] = notebook([
    md('''
    # 02 — EfficientNetB0 avec CLAHE conditionnel

    Ce notebook répète exactement le protocole de la baseline en modifiant **une seule variable** : les images
    dont l'écart-type du canal de luminance LAB est inférieur à 40 reçoivent un CLAHE. Cette expérience d'ablation
    dira si le traitement améliore réellement la généralisation.

    OpenCV est appelé avec `tf.numpy_function` : cette variante peut être plus lente côté CPU. Il faut comparer
    les métriques, mais aussi le temps par époque.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    md('''## Prétraitement CLAHE'''),
    code('''
    import cv2

    CONTRAST_THRESHOLD = 40.0

    def clahe_numpy(image):
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        if float(lightness.std()) >= CONTRAST_THRESHOLD:
            return image_uint8.astype(np.float32)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(lightness)
        result = cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB)
        return result.astype(np.float32)

    def apply_clahe_tf(image, label, weight):
        image = tf.numpy_function(clahe_numpy, [image], Tout=tf.float32)
        image.set_shape((*TARGET_SIZE, 3))
        return image, label, weight
    '''),
    code(MODEL_FUNCTIONS),
    code('''
    RUN_NAME = "efficientnet_clahe"
    TRAIN_TRANSFORM = apply_clahe_tf
    VAL_TRANSFORM = apply_clahe_tf
    '''),
    code(TRAINING_CELLS),
    md('''## Phase 1 — Backbone gelé'''),
    code(TRAIN_EXECUTION),
    md('''## Phase 2 — Fine-tuning partiel'''),
    code(FINETUNE_EXECUTION),
    md('''## Contrôle rapide'''),
    code(QUICK_EVAL),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


notebooks["03_evaluer_comparer_et_predire.ipynb"] = notebook([
    md('''
    # 03 — Évaluer, comparer et prédire

    Ce notebook compare la baseline et la variante CLAHE sur **exactement les mêmes observations**. Il produit :

    - accuracy, top-3 accuracy, F1 weighted et F1 macro ;
    - rapport et matrice de confusion normalisée ;
    - comparaison par classe ;
    - inspection visuelle des erreurs les plus confiantes ;
    - prédictions du test si `data/raw_x_test.csv` est disponible.

    Le choix final doit privilégier le F1 weighted, tout en surveillant le F1 macro afin de ne pas sacrifier
    les petites classes.
    '''),
    code(COMMON_CONFIG),
    code(DATASET_FUNCTIONS),
    code('''
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, top_k_accuracy_score,
    )

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
    '''),
    md('''## 1. Charger les modèles et obtenir les probabilités'''),
    code('''
    from tensorflow import keras

    experiments = {
        "baseline": {
            "path": ARTIFACTS_DIR / "efficientnet_baseline" / "best_model.keras",
            "transform": None,
        },
        "clahe": {
            "path": ARTIFACTS_DIR / "efficientnet_clahe" / "best_model.keras",
            "transform": apply_clahe_tf,
        },
    }

    missing = [str(run["path"]) for run in experiments.values() if not run["path"].exists()]
    if missing:
        raise FileNotFoundError("Modèles manquants. Exécuter les notebooks 01 et 02 :\\n" + "\\n".join(missing))

    y_true = val_df["class_index"].to_numpy()
    predictions = {}
    for name, run in experiments.items():
        ds = make_dataset(val_df, transform=run["transform"], weighted=False)
        model = keras.models.load_model(run["path"])
        probabilities = model.predict(ds, verbose=1)
        predictions[name] = {"probabilities": probabilities, "labels": probabilities.argmax(axis=1)}
    '''),
    md('''## 2. Tableau de synthèse'''),
    code('''
    metric_rows = []
    for name, result in predictions.items():
        y_pred = result["labels"]
        proba = result["probabilities"]
        metric_rows.append({
            "experiment": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "top3_accuracy": top_k_accuracy_score(y_true, proba, k=3, labels=np.arange(NUM_CLASSES)),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
        })
    metrics = pd.DataFrame(metric_rows).set_index("experiment").sort_values("f1_weighted", ascending=False)
    display(metrics.style.format("{:.4f}"))
    metrics.to_csv(ARTIFACTS_DIR / "model_comparison.csv")
    CHAMPION = metrics.index[0]
    print("Champion selon F1 weighted :", CHAMPION)
    '''),
    md('''## 3. Analyse par classe et matrice de confusion'''),
    code('''
    reports = {}
    for name, result in predictions.items():
        report = classification_report(
            y_true, result["labels"], labels=np.arange(NUM_CLASSES),
            target_names=[str(code) for code in index_to_code], output_dict=True, zero_division=0,
        )
        reports[name] = pd.DataFrame(report).T
        reports[name].to_csv(ARTIFACTS_DIR / f"classification_report_{name}.csv")

    per_class = pd.DataFrame({
        "prdtypecode": index_to_code,
        "f1_baseline": [reports["baseline"].loc[str(code), "f1-score"] for code in index_to_code],
        "f1_clahe": [reports["clahe"].loc[str(code), "f1-score"] for code in index_to_code],
    })
    per_class["delta_clahe"] = per_class["f1_clahe"] - per_class["f1_baseline"]
    display(per_class.sort_values("delta_clahe"))
    per_class.to_csv(ARTIFACTS_DIR / "per_class_comparison.csv", index=False)
    '''),
    code('''
    champion_pred = predictions[CHAMPION]["labels"]
    matrix = confusion_matrix(y_true, champion_pred, normalize="true")
    plt.figure(figsize=(15, 13))
    sns.heatmap(
        matrix, cmap="Blues", vmin=0, vmax=1,
        xticklabels=index_to_code, yticklabels=index_to_code,
    )
    plt.title(f"Matrice de confusion normalisée — {CHAMPION}")
    plt.xlabel("Code prédit")
    plt.ylabel("Code réel")
    plt.tight_layout()
    plt.show()
    '''),
    md('''## 4. Examiner les erreurs les plus confiantes

    Ces exemples révèlent souvent des labels ambigus, des images non informatives ou des confusions métier
    récurrentes. Ils sont généralement plus utiles qu'une optimisation aveugle des hyperparamètres.
    '''),
    code('''
    champion_proba = predictions[CHAMPION]["probabilities"]
    errors = val_df.copy().reset_index(drop=True)
    errors["true_index"] = y_true
    errors["pred_index"] = champion_pred
    errors["confidence"] = champion_proba.max(axis=1)
    errors = errors.query("true_index != pred_index").sort_values("confidence", ascending=False)

    sample = errors.head(12)
    fig, axes = plt.subplots(3, 4, figsize=(14, 12))
    for ax, row in zip(axes.flat, sample.itertuples()):
        image = plt.imread(row.image_path)
        ax.imshow(image)
        ax.set_title(
            f"vrai={index_to_code[row.true_index]} | prédit={index_to_code[row.pred_index]}\\n"
            f"confiance={row.confidence:.1%}", fontsize=9,
        )
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    '''),
    md('''## 5. Prédire le jeu de test (optionnel)

    Cette section attend `data/raw_x_test.csv` avec les colonnes `Id`, `imageid` et `productid`. Elle produit
    `artifacts/image_classification/submission_image.csv` au format `Id,prdtypecode`.
    '''),
    code('''
    TEST_CSV = DATA_DIR / "raw_x_test.csv"
    if not TEST_CSV.exists():
        print("Étape ignorée : ajouter", TEST_CSV, "pour générer les prédictions de test.")
    else:
        test = pd.read_csv(TEST_CSV)
        required = {"Id", "imageid", "productid"}
        assert required.issubset(test.columns), required - set(test.columns)
        test["image_path"] = test.apply(
            lambda row: str((ROOT / "images" / "image_test" /
                f"image_{row.imageid}_product_{row.productid}.jpg").resolve()), axis=1
        )
        assert test["image_path"].map(os.path.isfile).all()
        test["class_index"] = 0
        test["sample_weight"] = 1.0

        champion_run = experiments[CHAMPION]
        champion_model = keras.models.load_model(champion_run["path"])
        test_ds = make_dataset(test, transform=champion_run["transform"], weighted=False)
        test_probabilities = champion_model.predict(test_ds, verbose=1)
        test_codes = [index_to_code[index] for index in test_probabilities.argmax(axis=1)]
        submission = pd.DataFrame({"Id": test["Id"], "prdtypecode": test_codes})
        submission.to_csv(ARTIFACTS_DIR / "submission_image.csv", index=False)
        display(submission.head())
    '''),
], kernel_name="rakuten-metal", display_name="Rakuten TensorFlow Metal")


OUT.mkdir(parents=True, exist_ok=True)
for filename, content in notebooks.items():
    with open(OUT / filename, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=1)
    print("Écrit :", OUT / filename)
