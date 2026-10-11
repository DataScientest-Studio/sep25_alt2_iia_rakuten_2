# Classification multimodale de produits Rakuten

Ce dépôt explore la prédiction du code produit `prdtypecode` à partir du texte et de l'image d'une annonce.

## Parcours de classification supervisée des images

Les nouveaux notebooks sont dans `notebooks/image_classification/` et doivent être exécutés dans cet ordre :

1. `00_preparer_donnees_et_splits.ipynb` — audit, doublons et split groupé ;
2. `01_entrainer_efficientnet_baseline.ipynb` — baseline EfficientNetB0 ;
3. `02_entrainer_efficientnet_clahe.ipynb` — expérience CLAHE conditionnelle ;
4. `03_evaluer_comparer_et_predire.ipynb` — comparaison, analyse des erreurs et prédiction test.

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
