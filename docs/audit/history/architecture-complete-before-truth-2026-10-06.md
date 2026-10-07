# Tableau de bord de l'audit principal

Référentiel actif : **v10.1 — 2026-10-02**. Passage du **2026-10-06** : audit complet des axes **II. Factorisation** et **III. Arborescence, lisibilité et dépendances**. Les autres axes conservent leur état antérieur ; la lecture de chemins numériques ou CMake ne constitue pas leur certification.

## Verdict

| Axe | Couverture de ce passage | Verdict | Motif |
|---|---|---|---|
| I. Véracité | non réexaminée de bout en bout | indéterminé | aucune nouvelle référence indépendante par famille ; `PRODUCT-001` et `NUM-028` à `NUM-030` demeurent ouverts |
| II. Factorisation | complète au niveau des unités architecturales déclarées | non conforme | dépendance `src/common → tools`, produit concret dans `common`, géométrie gradients dans le lanceur prix LSM, concept de raffinement incomplet |
| III. Arborescence et lisibilité | complète au niveau des racines, owners, includes, entrées codegen et parcours demandés | non conforme | graphe terminal générique caché sous option européenne, 13 façades sans consommateur interne, nom rough pour la source de vérité equity, Quick start dépendant de scripts non suivis |
| IV. Performance | non examinée | indéterminé | aucune campagne admissible |
| V. CMake | aperçu structurel seulement | indéterminé | configuration isolée réussie ; pas de matrice d'incrémentalité |
| VI. Hygiène et artefacts | aperçu structurel seulement | indéterminé | inventaire local et statut des scripts consultés ; pas de décision de rétention |

La couverture « complète » des axes II et III signifie que toutes les **familles et responsabilités** présentes ont été croisées avec le manifeste et le graphe des dépendances, puis que les cœurs de chaque stratégie ont été examinés. Elle ne signifie pas une lecture manuelle ligne par ligne de chaque wrapper généré ni une validation des chiffres financiers, du débit GPU ou des binaires d'un autre profil. Les onze constats non résolus sont dans [response.md](../response.md). Aucun autre constat n'est ouvert sur la seule longueur d'un fichier ou le nombre de paramètres template.

## Snapshot

- Branche `main`, `HEAD` `03f72d81045330cd8a234b83c877292379cd0623`, worktree sale avant ce passage : **378 entrées de statut**, dont **187 non suivies**. Copie de reconstruction sous `/tmp/ai_factory_full_audit_20261006/` : `tracked.patch` SHA-256 `178c378252267a936bd9ded46f8fac093550aab9acf5f1a2ab9ef6b02102c099` (1 285 448 octets) et `untracked.tar.gz` SHA-256 `2e861b3aa3a499a5c673f0dead870805b696d2bd7cb74b892abb591e1f72aebf` (84 043 octets). Les retouches d'audit de ce passage viennent après cette copie.
- Configuration isolée `/tmp/ai_factory_full_audit_20261006/build` : C++20, GCC 14.3.0, CUDA 12.9.41, cuFFTDx/mathDx 25.12.1, SM89, Ninja, `BUILD_TESTING=ON`. GPU RTX 4090 Laptop 16 376 MiB. La compilation CUDA a été interrompue après la clarification du mandat ; aucun résultat numérique n'en est déduit.
- Regénération complète isolée de **7 031 sorties** plus le manifeste de provenance : 7 031 fichiers identiques au dépôt ; seules deux empreintes du manifeste différaient, puis ont été recalculées sans changement des autres champs. Le contrôle zéro diff final a été rejoué après les retouches et passe.

## Inventaire et graphe des responsabilités

- Manifeste typé : **13 moteurs**, **25 modèles**, **26 produits**, **3 courbes**, **439 bindings prix**, **387 bindings delta**, **313 bindings gradients**, **2 409 recettes disponibles**. Aucun dataset différé. Les 2 409 générateurs et recettes ont été vérifiés à leur chemin physique : **696 sous `catalog`**, **1 713 sous `work/catalog`**, zéro absent et zéro collision de chemin.
- Fichiers physiques : `src/common` 242, `src/model/equity` 2 433, `src/model/fixed_income` 313, `src/product` 168. Les 25 modèles possèdent paramètres, loader, dynamique, sample et compositions ; les 26 produits possèdent paramètres, loader et policy. Les analytics restent présentes uniquement lorsque le modèle en offre. Ce contrôle de présence ne remplace pas l'audit mathématique.
- Balayage de tous les `#include` dans `src`, `tools`, `catalog`, `tests`, `learning` : quatre includes inverses directs, regroupés dans `BOUNDARY-001` (un `src/common → tools`) et `STRUCT-001` (trois `src/common → product`). Aucun `src/product → model`, `src/model → tools/catalog/learning`, ni `src/curve → model/product` direct. Trente includes entre modèles concrets sont des compositions explicites Bates/Heston, Hull–White/OU, CIR++/CIR, G2++/G2 ; deux produits Phoenix incluent leur policy commune sous `src/product`.
- Le catalogue généré et CMake consomment le manifeste. Le contrôle `maintainer/tools/cuda/check_catalog_generators.py` couvre les 2 409 recettes et passe ; `maintainer/tools/cuda/check_model_layout.py` classe 2 404 fichiers modèle-produit et 313 fichiers d'infrastructure, 150 templates nommés, 2 316 fichiers générés et 1 319 écrits à la main. Ce dernier contrôle ne repère pas les quatre includes inverses, d'où la preuve complémentaire ci-dessus.
- Les dossiers `learning/common`, `learning/gan` et `learning/deep_pricing` conservent les briques réutilisables et leurs entrées de formation. Aucune importation `learning → work/catalog/tools` trouvée dans les modules Python examinés. Les expériences locales restent sous `work/experiments`; aucun include direct de cette racine dans `src`.

## Revue par famille

| Famille | Briques lues et frontière vérifiée | Conclusion architecture |
|---|---|---|
| Closed form scalaire et coopérative | `common/closed_form`, analytics modèle, Jamshidian commun, policy `product/european_swaption`, compositions de taux | séparation formule/modèle/produit réelle ; pas de moteur universel inutile relevé |
| MC markovien terminal et chemin | schedule, dynamics, `monte_carlo_kernel`, policies produit, préparation et graphes gradients terminal/path | mêmes dynamics/payoffs réutilisés ; topologies différenciées justifiées par calendrier et nœuds |
| LSM American et Bermudan | regressor/bases, simulate/régression/update/finalize, policies produit et les deux replays | moteur commun justifié ; `BOUNDARY-001`, `FACTOR-004`, `FACTOR-005` restent à corriger |
| Rough FFT et lift N-facteurs | kernels/schedules Volterra, graphes terminaux et de chemin, wrappers modèles, delta historique | partage des graphes réel ; `STRUCT-001` laisse une composition produit dans `common` |
| Samples markoviens, N-facteurs et FFT | `common/sample`, moteur FFT, bindings modèles et templates | stratégies distinctes gardées ; grille canonique de samples clairement fixée par l'API actuelle, pas d'abstraction supplémentaire recommandée sans nouveau contrat |
| Modèles ajustés de taux | Hull–White/OU, CIR++/CIR, G2++/G2, courbes et compositions | réutilisation du processus de base intentionnelle ; pas de copie de modèle constatée au niveau des interfaces |
| Produits terminaux et de chemin | 26 owners, payoff/schedule/pricing policy, Phoenix partagé, gradients produit | partage Phoenix légitime ; treize façades sans consommateur, dont six sous `product` (`BOUNDARY-005`) |
| Codegen et catalogue | manifeste `sample_manifest` → `manifest` → `capability_manifest`, renderer et 150 templates, recettes et CMake | génération reproductible et matrice cohérente ; `NAME-014` masque la portée de la table produit equity |

Le nombre de templates n'est pas en soi un défaut. Les paramètres principaux correspondent ici à des variations vérifiables : dynamique, calendrier, observation, côté, courbe, ordre de dérivation, topologie CUDA ou type de fit. Les constats de surcodage sont circonscrits aux **13 façades sans consommateur** et aux deux couplages explicites du moteur LSM ; une réécriture générale des kernels n'est pas justifiée par les preuves présentes.

## Exercices d'extension à blanc

| Changement | Propriétaire et modifications attendues | Friction constatée |
|---|---|---|
| Paramètre modèle dérivable | `model/<nom>/parameters`, domaine, dynamics/analytics, `price_gradients/device_preparation`, loi dans `sample_manifest`, tests | propriétaire clair ; la préparation modèle reste spécifique |
| Paramètre produit ou courbe | `product/<nom>` ou `curve/<nom>`, loader, adapter de préparation correspondant, recette sélectionnée | propriétaire clair ; pas de liste de kernels à modifier |
| Produit terminal equity | `src/product/<produit>`, déclaration produit/variant dans le manifeste, templates de binding existants | `ROUGH_PRODUCT_BINDINGS` est la source unique mais son nom masque sa portée (`NAME-014`) |
| Produit dépendant du chemin | même owner produit, handler/calendrier, contrat d'observation, manifeste et tests | schedules dense/régulier sont distincts ; aucune branche centrale supplémentaire requise pour un contrat déjà supporté |
| Modèle MC | dossier modèle, contrat de transition et sample, `sample_manifest`, adapter de sensibilités, compositions générées | coût concentré dans le modèle et la déclaration ; exceptions de RNG demandent une policy explicite |
| Formule fermée ou coopérative | analytics modèle, policy produit, engine/binding template si formule nouvelle | deux distributions de travail existent déjà ; pas de variante booléenne cachée identifiée |
| Nouvelles innovations ou FFT | dynamique modèle, contexte RNG, policy Volterra/FFT et spécialisation de binding | propriétaire numérique explicite ; migration des fichiers générés reste à comparer zéro diff |

Ces exercices sont des traces de fichiers et de dépendances, **pas** des extensions implémentées ni des preuves de compilation d'une nouvelle famille.

## Parcours de découverte

1. Dynamique/domaine : `sample_manifest.py` donne le modèle, `src/model/<famille>/<modèle>/parameters.hpp`, `parameter_domain.hpp` si présent, puis `dynamics.cuh` et son implémentation.
2. Produit : `src/product/<produit>/parameters.hpp`, `pricing_policy.cuh`, calendrier ou continuation sous le même owner.
3. Composition : `src/model/**/<modèle>/product/[<courbe>/]<produit>.cuh/.cu` ou template de binding correspondant.
4. Prix recette → kernel : `catalog` ou `work/catalog`, `generator.cpp`, `tools/pricing`, binding modèle-produit, moteur `common`. La variante locale passe par `tools/datasets/catalog_layout.py`.
5. Gradient : `SensitivityRequest`, plan `common/price_gradients`, adapter modèle/produit, policy `mono` ou `node_graph`, reconstruction. Rough terminal détourne encore par un header d'option européenne (`STRUCT-001`).
6. Target/exécutable : identité du dataset dans `capability_manifest.py`, `cmake/generated/CapabilityManifest.cmake`, puis target `generate_*` et binaire du build choisi.
7. Lancer/reprendre : README → `docs/dataset-generation-workflow.md` → `tools/run_generator.py` et contrôleur de checkpoints. Le script d'entrée est non suivi au snapshot (`DOC-004`).
8. Limites/preuves : `docs/cuda/README.md`, README modèle, `docs/audit/{response,closed,status}.md`, références indépendantes sous `docs/validation` sans les certifier ici.

Sur 99 fichiers Markdown parcourus, 420 liens locaux ont été inspectés. Quatre liens relatifs cassés dans l'ancien tableau de bord archivé ont été corrigés ; un cinquième signal correspond à une expression mathématique interprétée à tort comme lien par le scan. Les commandes de Quick start ont été vérifiées contre `git ls-files`.

## Contrôles exécutés et limites

- `python3 maintainer/tools/cuda/check_model_layout.py` : passe.
- `python3 maintainer/tools/cuda/check_catalog_generators.py` : passe, 2 409 recettes.
- Tests Python structure/codegen : deux erreurs initiales dues à la lecture des chemins logiques du catalogue ; correction dans `maintainer/tests/codegen/test_price_gradient_coverage.py`, puis **53 tests sur 53 passent** après correction.
- Régénération isolée de toutes les familles : **7 031 sorties identiques**, provenance recalculée, comparaison finale **zéro diff**.
- Aucun prix, gradient, sample, profil de performance ou dataset n'a été recalculé pour conclure sur la véracité ou la vitesse.

Le passage ciblé précédent est conservé dans [history/architecture-targeted-before-complete-review-2026-10-06.md](architecture-targeted-before-complete-review-2026-10-06.md). Le tableau antérieur se trouve dans [history/truth-and-fixes-before-architecture-review-2026-10-06.md](truth-and-fixes-before-architecture-review-2026-10-06.md).
