# Classification multimodale de produits Rakuten

Ce dépôt explore la prédiction du code produit `prdtypecode` à partir du texte et de l'image d'une annonce.

## Parcours de classification supervisée des images

Les nouveaux notebooks sont dans `notebooks/image_classification/` et doivent être exécutés dans cet ordre :

1. `00_preparer_donnees_et_splits.ipynb` — audit, doublons et split groupé ;
2. `01_cnn_from_scratch.ipynb` — baseline : CNN entraîné de zéro ;
3. `02_transfer_learning_efficientnet.ipynb` — amélioration : EfficientNetB0 gelé (feature extraction)
   puis fine-tuné, chaque étape sauvegardée et évaluée ;
4. `03_evaluer_comparer_et_predire.ipynb` — progression baseline → transfert, analyse par classe,
   confusions, erreurs et prédiction test ;
5. `04_interpretabilite_gradcam.ipynb` — cartes Grad-CAM comparées et test de fidélité par masquage.

Les notebooks sont générés par `tools/build_image_classification_notebooks.py`. Le script n'écrase pas
un notebook déjà exécuté, sauf avec `--force`. Le premier parcours (EfficientNet seul et ablation CLAHE)
est conservé avec ses résultats dans `notebooks/image_classification/archive_v1/`.

Installation recommandée dans un environnement Python isolé :

```bash
python -m pip install -r requirements-image.txt
```

Sur un Mac Apple Silicon, utiliser plutôt l'environnement Metal dédié :

```bash
conda create --prefix .conda/rakuten-metal python=3.10 -y
conda run --prefix .conda/rakuten-metal python -m pip install -r requirements-metal.txt
conda run --prefix .conda/rakuten-metal python -m ipykernel install --user --name rakuten-metal --display-name "Rakuten — TensorFlow Metal"
```

Avant l'entraînement, vérifier que TensorFlow affiche un périphérique `GPU` avec
`tf.config.list_physical_devices("GPU")`.

Les notebooks détectent automatiquement la racine du projet. Ils attendent `data/raw_x.csv`,
`data/raw_y.csv`, `images/image_train/` et, pour la prédiction finale, `data/raw_x_test.csv`.
Les modèles, manifestes et rapports sont écrits dans `artifacts/image_classification/`.

Les notebooks historiques d'exploration restent disponibles dans `Exploration_donnees_&_Dataviz/`.
