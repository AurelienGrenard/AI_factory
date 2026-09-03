# Etat de l'audit de validation

## Objet

Ce document conserve la provenance, la couverture, les exclusions et les
preuves du dernier passage de `query.md`. Les problemes non resolus sont
exclusivement decrits dans `response.md`; les constats fermes sont dans
`closed.md`.

## Provenance du passage

- Date du passage : 2026-08-26.
- Separation documentaire : 2026-08-27.
- Branche : `refactor/unify-cuda-model-contracts`.
- Revision auditee : `3bfb6a56449f60ce856f7e0734e2b72d60da1b7a`.
- Worktree deja modifie avant l'audit : oui, 133 fichiers modifies, 9 supprimes
  et 77 non suivis.
- Aucun dataset, cache, notebook, statut ou probleme de code n'a ete corrige.
- Comptabilite : 2 constats non resolus dans `response.md`, aucun constat dans
  `closed.md`.

## Registre des preuves

| Ref. | Preuve executee ou inspectee | Resultat utile |
|---|---|---|
| V1 | Inventaire des 376 `dataset.yaml`, blocs `validation`, notebooks et caches persistants | 329 blocs de validation, 327 pending, 326 notebooks references absents, 2 available et 71 caches. |
| V2 | `python -m unittest discover -s validation/tests -v`; `ctest --test-dir build-dev -L cached_reference --output-on-failure` | 40 tests Python : 2 echecs et 7 erreurs; 0/71 tests de cache passent. |
| V3 | Lecture de `validation/reference_price_dataset.py`, des caches JSON, YAML rough et generateurs correspondants | Fingerprints des donnees/policy presents; provenance du generateur/build absente; metadata rough obsoletes. |

Les regenerations Premia/QuantLib, la publication d'artefacts et les executions
GPU ont ete exclues. Les echecs ont ete observes en lecture et par les tests
fail-closed existants; ils n'ont pas ete contournes.

## Couverture courante

| Audit | Statut | Date | Revision | Perimetre et exclusions | Preuves |
|---|---|---|---|---|---|
| Validation and reproducibility | partiel | 2026-08-26 | `3bfb6a5`; worktree modifie | Catalogues, caches persistants, empreintes, provenance, notebooks, tests Python/CTest et metadata rough. Regeneration Premia/QuantLib, publication et GPU exclues. | V1, V2, V3 |
