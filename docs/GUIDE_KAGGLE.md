# Guide — lancer l'expérience sur Kaggle

Cette version est **recommandée** : rien à installer, un GPU gratuit, environ **2 à 3 heures** de calcul.
Préparation : environ 20 minutes, une seule fois.

## Relance version 3 (après la relecture de Nesma), environ 1 h 30

1. **Résultats du premier lancement** : si la session Kaggle d'hier est fermée, déposez `results_ibmed.zip` comme dataset privé, nommé par exemple `ibmed-pilot`. Ces résultats servent d'analyse de sensibilité, et les réponses de MedGemma sur SLAKE sont réutilisées, ce qui évite environ 1 h de calcul.
2. **Dataset `ibmed-expe`** : ouvrez-le, cliquez sur **New Version**, puis déposez le nouveau `expe_kaggle.zip` (il contient le split VQA-RAD corrigé).
3. **Notebook** : importez le nouveau `kaggle_ibmed.ipynb` (File → Import Notebook). Vérifiez ensuite :
    - GPU T4 x2 et Internet activés ;
    - le secret `HF_TOKEN` ;
    - les entrées `ibmed-expe`, `vqa-rad` et `ibmed-pilot`.
4. Lancez **Run All**. Vérifiez ensuite deux lignes :
    - cellule 5 : `Pilot kept for sensitivity analysis: True` ;
    - cellule 7 : `SLAKE MedGemma answers reused: True`.
5. À la fin, téléchargez **`results_ibmed_v3.zip`** dans l'onglet Output.

## Ce dont vous avez besoin

- Un compte **Kaggle** avec le téléphone vérifié (Settings → Phone verification). Sans cette vérification, pas de GPU ni d'Internet dans les notebooks.
- Un compte **HuggingFace**.
- Le paquet `expe_kaggle.zip`.

## Étape 1 — HuggingFace (5 min)

1. Sur huggingface.co : Settings → Access Tokens → **New token**, type *Read*. Copiez-le (il commence par `hf_`).
2. Ouvrez la page `google/medgemma-4b-it` et acceptez la licence (*Health AI Developer Foundations*). **Sans cet accord, le modèle ne se télécharge pas.**

## Étape 2 — Déposer les scripts sur Kaggle (5 min)

1. Décompressez `expe_kaggle.zip`.
2. Sur Kaggle : **Datasets → New Dataset**, glissez le dossier `expe/`, nommez-le `ibmed-expe` et laissez-le **Private**. Cliquez sur **Create**.
3. *(Recommandé)* Téléchargez VQA-RAD depuis le dépôt OSF cité par l'article (https://osf.io/89kps/), puis déposez-le de la même façon comme dataset privé `vqa-rad`. Il doit contenir `VQA_RAD Dataset Public.json` et le dossier `VQA_RAD Image Folder`.
   - Si vous sautez cette étape, le notebook essaie de le télécharger tout seul.
   - En dernier recours, il prend une version HuggingFace sans métadonnées. Les analyses par type de question et la permutation par région sont alors perdues pour VQA-RAD.

## Étape 3 — Ouvrir le notebook (5 min)

1. **Code → New Notebook**, puis **File → Import Notebook**, et choisissez `kaggle_ibmed.ipynb` (il est dans `expe/`).
2. Dans le panneau de droite, onglet **Settings** :
    - **Accelerator → GPU T4 x2** ;
    - **Internet → On**.
3. Menu **Add-ons → Secrets → Add a new secret** : nom `HF_TOKEN`, valeur = votre jeton. Cochez la case pour l'attacher au notebook.
4. **Add Input** (panneau de droite) : ajoutez `ibmed-expe`, et `vqa-rad` si vous l'avez créé.

## Étape 4 — Lancer (2 à 3 h)

1. Cliquez sur **Run All**.
2. **Surveillez la cellule 8** (environ 15 minutes après le lancement). Elle teste MedGemma sur 5 questions par jeu et affiche ses réponses :
    - si elles valent `yes` / `no` et que le taux « unparsed » est bas : tout va bien, laissez tourner ;
    - si elles sont vides, bizarres ou en majorité « unparsed » : **arrêtez** (Stop) et envoyez une capture de cette cellule.
3. Si la session s'interrompt (déconnexion, limite de 12 h), **relancez uniquement la cellule 9**. Elle reprend là où elle s'est arrêtée.

Vous pouvez fermer le navigateur pendant le calcul avec **Save Version → Save & Run All (Commit)** : Kaggle exécute alors le notebook en arrière-plan.

## Étape 5 — Récupérer les résultats

À la fin, la cellule 11 affiche les deux figures et le tableau principal.
Onglet **Output** → téléchargez **`results_ibmed.zip`** et envoyez-le à Souaad.

## Problèmes fréquents

| Message | Cause | Solution |
| --- | --- | --- |
| `Scripts not found` (cellule 2) | Le dataset `ibmed-expe` n'est pas ajouté | Add Input → `ibmed-expe` |
| `401` / `403` / `gated repo` | Licence MedGemma non acceptée, ou jeton absent | Étape 1.2 ; vérifiez le secret `HF_TOKEN` |
| `No GPU` / `CUDA not available` | Accélérateur non activé | Settings → GPU T4 x2, puis relancer |
| `Could not resolve host` | Internet désactivé | Settings → Internet → On (téléphone vérifié) |
| `CUDA out of memory` | Autre processus sur le GPU | Run → Restart & clear outputs, puis Run All |
# Archival execution guide

> This guide comes from the original Kaggle execution package. The real-data
> runs have since been completed and audited. For the publication-facing
> statistical procedure and results, use `../reproducibility/` and read
> `STATISTICAL_PROVENANCE.md`. Do not interpret prospective wording below as
> the current status of the experiment.

