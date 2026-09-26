# Migration du compteur Philox commun et choix des flux par modèle

**État :** mapping V2 commun intégré ; qualification en cours, décrite par
[NUM-031](../audit/response.md#num-031--réserver-les-domaines-philox-aux-sources-qui-exigent-une-isolation).
Le compteur `(path_bas, path_haut, groupe, domaine)` est commun à tous les
modèles stochastiques dans `src/common/philox.cuh`. Le domaine zéro porte le
flux unique ; `src/common/philox_domains.cuh` ajoute des sources séparées
quand une consommation variable le demande. Merton, Kou, Bates,
Variance-Gamma et CIR/CIR++ utilisent plusieurs flux ; Heston QE, Vasicek et
les autres modèles à tirages fixes en utilisent un seul. Toutes les recettes
stochastiques déclarent `philox_source_step_v2` et publient sous `/v2/` ;
l'URL `/v1/` reste réservée aux anciennes bases. Les datasets déjà générés restent des artefacts de
l'ancien binaire ; la migration du code ne les régénère pas. Leur empreinte
de recette dans `generation.yaml` permet de les distinguer des nouvelles
recettes, même si le chemin local de travail est réutilisé.

Le [plan global prix-gradients](price-gradients-migration-plan.md) indique à
quelle étape ce mapping est requis pour migrer les sensibilités markoviennes
puis rough ; le présent document possède uniquement l'adressage Philox.

Ce plan concerne l'adressage **à l'intérieur d'une trajectoire**. Les domaines de
seeds `RngDomainSpec` du catalogue séparent déjà les recettes et les flux de
génération ; ils ont une autre responsabilité. Le [contrat actif des dynamiques](model-dynamics-contract.md#suite-aléatoire-philox)
est `(path_bas, path_haut, groupe, domaine)` pour tous les modèles, sous la clé
`base_seed + global_row`. Le domaine vaut zéro pour un flux unique ; les
modèles multi-flux encodent source et pas dans le domaine.

| Voie | Flux physiques | Affectation |
|---|---:|---|
| Heston QE, Vasicek, G2, autres modèles à consommation fixe, rough FFT | 1 | domaine zéro continu ; les facteurs browniens distincts ne demandent pas automatiquement plusieurs flux |
| Merton, Kou, Bates | 3 | source 1 diffusion, 2 compte de Poisson, 3 marques de sauts |
| Variance-Gamma | 2 | source 1 Gamma, 2 Brownien |
| CIR et CIR++ | 2 | source 1 compte de Poisson, 2 Gamma |

Cette affectation décrit l’adressage des tirages actuels. Le couplage
événementiel des marques Merton/Bates est intégré aux adapters
`price_gradients` ; sa qualification multi-seeds et forte intensité reste
suivie par NUM-032.

## Règle de choix du nombre de flux

- Garder Philox-4x32-10 et une clé de ligne `make_key(base_seed + global_row)`.
  Ni le lot, ni la géométrie CUDA, ni l'indice du scénario bumpé n'entrent dans
  l'adresse : le prix et ses sensibilités reçoivent les mêmes innovations.
- **Garder une seule `UniformSequence` par chemin par défaut, dans le domaine
  zéro du compteur V2.** Si le modèle
  consomme un nombre fixe de tirages dans un ordre fixe à chaque pas, les
  facteurs browniens et les variables auxiliaires peuvent partager ce flux.
  Heston QE tire aujourd'hui ses deux normales et son uniforme même si le
  schéma n'utilise pas les trois valeurs ; Vasicek consomme ses deux normales
  conjointes dans un ordre fixe. Ces cas sont les témoins mono-flux, pas des
  candidats obligatoires à trois ou deux domaines. Une consommation variable ne reste mono-flux que si tous les scénarios
  partagent réellement le même tirage canonique et si la coordonnée bumpée ne
  change pas sa consommation. Les sensibilités d’intensité et de maturité des
  modèles à sauts utilisent désormais les domaines et le couplage événementiel
  prévus à cet effet.
- Donner néanmoins un **identifiant logique stable** à chaque facteur
  brownien indépendant et à chaque autre source. Un domaine Philox distinct
  est justifié lorsque sa consommation variable ou conditionnelle peut décaler
  un autre facteur ou le pas suivant, ou lorsqu'une invariance CRN explicite
  ne peut pas être garantie par le flux unique. Les browniens corrélés des
  sous-jacents restent construits à partir des facteurs indépendants ; un
  bump de corrélation conserve les innovations du même chemin, que les
  facteurs soient dans un ou plusieurs domaines physiques.
- Pour les sources qu'il faut isoler, former le compteur à partir de
  `(path_bas, path_haut, groupe, domaine)`, les quatre champs étant des
  `uint32`. Le domaine identifie la source ; le pas est adressé dans le groupe
  et sa composante pour un coût fixe, ou dans un sous-domaine de pas pour une
  loi à rejet. Une proposition à vérifier pour les sources variables est
  `domaine = (source_id << 24) | step_index`, avec `source_id` dans `1..255`
  et `step_index` dans `0..2^24-1` : le domaine zéro reste alors réservé au
  flux fixe. Cette proposition ne couvre que 255 sources isolées ; ces bornes
  et la répartition des sorties seront figées seulement après inventaire et
  mesures. Garder la convention historique séparée par version.
- Ne pas maintenir automatiquement un `UniformSequence` et un
  `NormalPairCache` par domaine durant toute la trajectoire : chaque contexte
  contient aussi curseur et groupe de quatre valeurs, ce qui peut accroître
  les registres et les appels Philox. Comparer des domaines tirés à la demande
  ou par pas, avec un seul groupe vivant à la fois, au flux unique de référence.
- Conserver un adressage injectif et borné. Le premier groupe peut valoir zéro,
  car deux sources ont déjà des domaines distincts. Les consommations variables
  au pas `t` ne doivent déplacer ni un autre facteur ni le pas `t+1`. Les
  codes de source doivent être distincts dans **tout le pricer composé**, y
  compris ses policies modèle et produit. Les marques de sauts doivent avoir
  une adresse stable par événement commun. Le rangement de l'identifiant
  d'événement, du groupe de tirage dans cet événement et des éventuels rejets
  doit être injectif, y compris lorsque les marques et le compte de Poisson
  consomment des nombres variables de groupes. Son encodage et ses bornes
  seront spécifiés avant intégration. Un cas qui dépasse le contrat échoue
  explicitement ; il ne rabat jamais les indices par modulo.
- Distinguer la **version du mapping interne** de la version des réservations
  de seeds `RngDomainSpec`. Une nouvelle recette ou version de dataset déclare
  le mapping employé et ses graines ; les recettes historiques gardent leur
  convention et leurs empreintes dans leur preuve de génération. Tant qu'une
  ancienne recette reste active, elle garde aussi son mapping. Les alias CRN
  prix/prix-gradients doivent pointer vers la même réservation de la nouvelle
  version. Aucune parité bit à bit n'est annoncée entre anciens et nouveaux
  mappings.

Le [plan des gradients de saut](jump-price-gradient-migration-plan.md) possède
la décision de tirer les marques centrales événement par événement, le
couplage des comptes et les lois par modèle. Le présent plan possède seulement
l'adresse Philox des sources et leur isolation, y compris le replay par
événement ; les deux chantiers peuvent partager leurs campagnes de preuve.

Le [pilote SM89](../performance-reports/philox-domain-pilot-sm89-2026-09-22.md)
consigne les premiers tests GPU et mesures Merton de la migration active.
Le [contrôle du compteur commun](../performance-reports/philox-unified-sm89-2026-09-22.md)
consigne la parité mono-flux, les vérifications rough FFT et les mesures après
unification.

## Ordre de réalisation

1. **Inventorier les sources et leurs consommations.** Pour chaque moteur de
   pricing et de sampling, relever facteurs indépendants, tirages auxiliaires,
   représentation agrégée ou événementielle des marques, nombre fixe ou
   variable de tirages, lois à rejet, dépendance à l'état, index de
   pas/événement, maxima de grille, usage par prix, gradients et exercice gelé.
   Inclure exact terminal,
   markovien à pas, CIR/Gamma, LSM, rough N-facteurs et FFT. Distinguer un
   facteur brownien des plusieurs normales nécessaires pour observer ce même
   brownien : Vasicek taux/intégrale consomme deux normales par pas mais n'a
   qu'un facteur brownien ; Heston QE a deux facteurs et un uniforme auxiliaire.
   Classer chaque voie `mono-flux suffisant` ou `isolation à démontrer` ; le
   nombre de browniens n'est pas à lui seul un motif de séparation.
2. **Fixer et prouver le format d'adresse des voies isolées.** Réserver les
   codes de source dans le modèle, définir le rangement
   pas/groupe/composante/événement et les gardes d'entiers. Vérifier notamment
   que le rangement des normales ne calcule pas inutilement un Box--Muller par
   facteur et par pas.
   Mettre un encodeur unique dans `src/common` à côté de `philox.cuh`, sans
   recopier l'algorithme Philox. Les modèles décrivent leurs sources ; les
   kernels communs ne connaissent pas les noms Heston, Poisson ou Vasicek.
   L'API évite de garder un groupe de quatre uniformes et un cache normal par
   domaine dans les registres pendant tout le chemin.
3. **Faire un pilote hors des recettes publiées.** Mesurer d'abord Heston QE et
   Vasicek mono-flux comme témoins de coût et de reproductibilité V2. Sur Merton, comparer
   un tirage canonique mono-flux qui conserve réellement le CRN requis à des
   adresses isolées pour le Poisson, le brownien et les marques. La
   consommation variable du Poisson est un cas d'épreuve, pas une décision
   préalable d'utiliser plusieurs domaines. Comparer un contexte par domaine
   aux tirages à la demande ; mesurer la voie événementielle Merton selon son
   plan propriétaire. Garder les chemins actuels tant que les pilotes ne sont
   pas qualifiés. Le couplage des comptes et marques est suivi par NUM-032.
4. **Étendre sélectivement par familles après mesure.** Couvrir ensuite les autres moteurs
   markoviens, les produits à exercice gelé, les facteurs de taux, les sauts
   supplémentaires et enfin les moteurs rough/FFT, dont les paires complexes
   et l'accès aléatoire aux normales ont leur propre mapping. Préserver le
   tirage central unique et le replay commun prix/gradients à chaque étape.
5. **Versionner les consommateurs et la provenance.** Propager le choix de
   mapping dans les recettes, le codegen, les métadonnées et les checkpoints de
   génération. Une reprise refuse une version, seed ou binaire incompatibles.
   Ne basculer une recette publiée qu'avec une nouvelle identité/version et
   les validations requises. Les anciens datasets restent liés à leur recette
   d'origine et à l'archive de sources gelées ; retirer ultérieurement un
   consommateur `price_delta` du catalogue actif ne réécrit pas cette preuve.
6. **Actualiser les contrats actifs après qualification.** Mettre à jour
   `model-dynamics-contract.md`, les contrats prix/gradients et les règles de
   provenance seulement quand la voie correspondante est intégrée. Fermer
   NUM-031 avec les preuves et la portée réelle, sans annoncer une couverture
   rough ou multi-sous-jacent non testée.

## Preuves et critères de passage

- **Adresse et reproductibilité :** tests de non-collision et de bornes sur
  `path=0`, `2^32-1`, `2^32`, les derniers groupes/pas/sources ; rejet des
  débordements. Tester aussi deux policies composées et plusieurs groupes
  pour un même événement. Rejeu bit à bit sous changement de batch, taille
  de bloc, sélection de sensibilités et reprise de campagne. Les bases
  historiques ne sont pas comparées bit à bit au nouveau binaire : elles se
  rejouent depuis leurs sources gelées et leur ancienne recette. Le décalage
  d'empreinte recette rend explicite l'incompatibilité d'un cache local V1
  avec la nouvelle recette V2.
- **Isolation et CRN :** provoquer des rejets Poisson/Gamma et des branches QE
  différentes ; prouver que les autres facteurs et les pas suivants gardent
  leurs tirages. Comparer les innovations effectivement utilisées par prix et
  gradients, y compris les scénarios où le central est réutilisé. Vérifier que
  chaque identifiant d'événement redonne la même innovation primitive ; les
  comptes emboîtés, valeurs de marque et prix relèvent de NUM-032.
- **Lois et prix :** vérifier normales indépendantes et corrélées, moments
  et marges de tirage des pilotes, puis vérifier que leur loi de prix ne change
  pas avec le seul nouvel adressage. La qualification des sommes de sauts,
  compensateurs et différences finies appartient à NUM-032. Ne pas exiger
  l'égalité bit à bit entre mappings.
- **CUDA et performance :** exécuter les tests CUDA pertinents et Compute
  Sanitizer (`memcheck`, `racecheck`, `initcheck`, `synccheck`). Sur le même GPU
  et la même toolchain, mesurer selon le
  [protocole de performance](../performance-regression-protocol.md) le temps
  kernel, l'appel public, le pipeline et la génération complète, avec
  registres, spills, shared/local memory et SASS. Inclure chemins courts/longs,
  un ou plusieurs facteurs, prix seuls, gradients et LSM/FFT quand intégrés.
  Comparer explicitement la pression registre et les appels Philox du flux
  unique, des domaines conservés en état et des domaines tirés à la demande.
  Une régression significative déclenche une révision du rangement des tirages
  ou de la stratégie, pas une bascule silencieuse des recettes.

La primitive Philox et `UniformSequence` restent dans `philox.cuh` ; les
transformations de lois y acceptent désormais une source d'uniformes générique.
Le nouvel adressage est isolé dans `philox_domains.cuh`, sans coût d'état pour
les modèles qui conservent le flux unique.
