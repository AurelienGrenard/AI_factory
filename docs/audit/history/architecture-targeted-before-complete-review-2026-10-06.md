# Tableau de bord de l'audit principal

Référentiel actif : **v10.1 — 2026-10-02**.

## Passage ciblé du 2026-10-06 — factorisation et arborescence

- **Mandat :** reprise des axes II et III après les corrections
  `NUM-035`, `FACTOR-003` et `BOUNDARY-008`. Ce passage examine les
  frontières de `src/common`, le moteur LSM, les compositions rough
  terminales, des templates de génération et les façades d'exercice gelé.
- **Verdict :** **non conforme sur les portions examinées**. Trois
  régressions historiques sont rouvertes dans [response.md](../response.md) :
  `BOUNDARY-001`, `STRUCT-001` et `BOUNDARY-005`. Les trois corrections
  du passage précédent restent décrites dans [closed.md](../closed.md).
- **Limite :** la revue ne couvre pas exhaustivement les 3 315 fichiers
  `src/common`, `src/model`, `src/product` et templates comptés lors
  de l'inventaire ; elle ne valide pas toutes les combinaisons
  modèle × produit × moteur ni tous les parcours de découverte.

## Snapshot et preuves

- **Source :** branche `main`, `HEAD`
  `03f72d81045330cd8a234b83c877292379cd0623`, worktree sale
  (377 entrées de statut, 186 fichiers non suivis). Snapshot sous
  `/tmp/ai_factory_arch_audit_20261006/` :
  `tracked.patch` SHA-256
  `4a220d3473705fe2021637661da386b2c89e79e99c051598e0eb01c1f68529ae`
  et `untracked.tar.gz` SHA-256
  `213aebba7923319447ce5ce4b18d5634ce145f94b8d59231de57966bef40f740`.
- **Matériel et outils :** build isolé SM89/CUDA 12.9.41 sous
  `/tmp/ai_factory_audit_20261005/build12`. Ce passage est une lecture
  statique ; aucun prix, kernel ou dataset n'a été modifié ni recalculé.
- **Recherche :** inventaire des includes directs de `src/common` vers
  `model`, `product`, `curve`, `catalog` et `tools` : deux headers
  fautifs trouvés, détaillés dans `BOUNDARY-001` et `STRUCT-001`.
  Recherche des consommateurs des six façades `frozen_exercise` :
  zéro dans `src`, `tools`, `tests`, `cmake`, `catalog`.
  L'existence d'un client externe n'a pas été établie.
- **Contrôle existant :** `python3 maintainer/tools/cuda/check_model_layout.py`
  passait au passage précédent, mais ce succès ne couvre pas les deux
  dépendances directes identifiées ici.
- **Parcours de découverte :** sur G2/Bermudan, le domaine et la
  dynamique sont sous `src/model/fixed_income/g2`, le calendrier et le
  payoff sous `src/product/bermudan_swaption`, la composition de prix et
  gradients sous `src/model/fixed_income/g2/product`, et les recettes et
  targets dans `catalog/model/fixed_income/g2` et
  `cmake/generated/CapabilityManifest.cmake`. Ce parcours atteint le
  moteur LSM commun, où il rencontre la dépendance `tools`.
  Sur rough Bergomi/digital, le binding modèle-produit sous
  `src/model/equity/rough/rough_bergomi/product` conduit au graphe terminal
  générique par `gaussian_european_option_graph.cuh`, d'où le détour décrit
  dans `STRUCT-001`. Les parcours de génération et reprise sont documentés
  depuis le README racine vers `docs/dataset-generation-workflow.md`, sans
  exécution d'une campagne pendant ce passage.
- **Codegen :** deux templates de sensibilités LSM American et Bermudan
  totalisent 646 et 647 lignes, avec 345 lignes identiques dans des blocs
  communs. Ce seul comptage ne démontre pas une duplication métier :
  ils partagent déjà le moteur LSM et diffèrent par l'exercice, le
  calendrier et la policy produit. Aucun constat n'est ouvert sur leur
  longueur seule.

## Couverture et verdict

| Axe | Couverture | Verdict | Preuve et limite |
|---|---|---|---|
| I. Véracité | non réexaminée | indéterminé | trois corrections antérieures vérifiées sur tests ciblés ; pas de nouvelle référence numérique |
| II. Factorisation | partielle | non conforme | dépendance `src` → `tools` et composition produit dans `common` ; autres moteurs et exercices d'extension non parcourus |
| III. Arborescence et lisibilité | partielle | non conforme | nom de graphe générique sous option européenne et six façades sans consommateur interne ; parcours complets et clients externes non qualifiés |
| IV. Performance | non examinée | indéterminé | aucun profil |
| V. CMake | non examinée | indéterminé | aucun audit de target |
| VI. Hygiène et artefacts | non examinée | indéterminé | snapshot pris, inventaire de rétention non exécuté |

## Registres

[response.md](../response.md) contient sept constats ouverts :
`PRODUCT-001`, `NUM-028` à `NUM-030`, `BOUNDARY-001`,
`STRUCT-001` et `BOUNDARY-005`. Les résolutions historiques des trois
identifiants rouverts restent dans [closed.md](../closed.md). Le tableau de
bord précédent est conservé dans
[history/truth-and-fixes-before-architecture-review-2026-10-06.md](truth-and-fixes-before-architecture-review-2026-10-06.md).
