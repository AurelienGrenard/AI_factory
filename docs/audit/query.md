# Référentiel d'audit du dépôt C++/CUDA

Version du référentiel : **10.1 — 2026-10-02**.

Ce document définit les questions, les preuves et les verdicts de l'audit
principal d'AI Factory. Il ne contient ni les résultats d'un passage précis, ni
un plan de refactorisation, ni une autorisation implicite de modifier le dépôt.

## Mission

Déterminer si le projet :

1. calcule les quantités mathématiques et financières annoncées ;
2. partage le code au bon niveau entre prix, gradients, Hessiennes et samples ;
3. se parcourt et se comprend sans mémoire préalable du projet ;
4. emploie des algorithmes et des stratégies CUDA adaptés à ses charges ;
5. possède un graphe CMake exact, minimal et reproductible ;
6. ne conserve aucun fichier, target, exécutable ou artefact sans propriétaire.

La justesse précède toute autre qualité. Une optimisation, une factorisation ou
une simplification qui change silencieusement la quantité calculée est un
défaut. La factorisation vise des invariants réels ; elle ne vise ni le minimum
de fichiers ni un moteur universel. La performance se juge sur le pipeline
réel, avec son coût numérique, ses ressources et ses transferts.

## Axes et ordre de décision

| Axe | Question principale | Préfixes usuels des constats |
|---|---|---|
| I. Véracité | Calcule-t-on la bonne quantité ? | `NUM`, `ANALYTICS`, `PRODUCT`, `TEST` |
| II. Factorisation | Les responsabilités partagées sont-elles placées au bon niveau ? | `FACTOR`, `POLICY`, `BOUNDARY` |
| III. Arborescence et lisibilité | Un lecteur trouve-t-il et comprend-il naturellement le code ? | `STRUCT`, `NAME`, `DOC` |
| IV. Performance | Le pipeline utilise-t-il correctement algorithmes, GPU et mémoire ? | `PERF`, `CUDA` |
| V. CMake | Le graphe de build décrit-il exactement le projet ? | `BUILD` |
| VI. Hygiène et artefacts | Chaque fichier et chaque sortie ont-ils un propriétaire et un usage ? | `HYGIENE`, `STRUCT`, `BUILD` |

Ces préfixes ne renumérotent pas les constats historiques. Un constat transversal
possède un axe principal et mentionne ses axes secondaires. Il n'est jamais
recopié dans plusieurs sections.

Les tests, le codegen, les sanitizers et les références numériques sont des
**moyens de preuve transversaux**. Ils ne constituent pas des axes indépendants.

---

# Mandat et méthode

## Périmètre

L'audit principal couvre :

- `src` : bibliothèques financières, numériques, CUDA et runtime ;
- `learning` : code réutilisable de préparation, réseaux et entraînement ;
- `tools` : codegen, génération, inspection, publication et performance ;
- `catalog` : recettes permanentes et générateurs minimaux ;
- `tests` : preuves fonctionnelles, numériques, structurelles et de performance ;
- `cmake`, `CMakeLists.txt` et `CMakePresets.json` ;
- la documentation maintenue ;
- le build principal `build/`, les racines locales `work/`, `datasets/` et
  `artifacts/` lorsqu'elles existent ;
- tout autre dossier à la racine, dont la présence doit être expliquée.

`validation/**` et `docs/validation/**` sont régis par le
[référentiel de validation indépendant](../validation/query.md). L'audit
principal peut lire leurs interfaces et leurs résultats publiés, mais ne les
certifie pas et ne les modifie pas implicitement.

## Snapshot obligatoire

Avant toute conclusion, enregistrer :

- date, branche et `HEAD` ;
- état de l'index et du worktree, y compris fichiers non suivis ;
- diff et contenus non suivis nécessaires pour reconstruire un worktree sale ;
- toolchains, options CMake, GPU et dépendances pertinentes ;
- hashes des binaires, inputs, recettes et preuves effectivement utilisés ;
- disponibilité du GPU, alimentation et limites des expériences autorisées.

Un commit ne décrit pas un worktree sale. Une preuve historique n'est
réutilisable que si son source, sa configuration, son domaine numérique et son
binaire restent compatibles avec le snapshot courant.

Les mutations d'audit utilisent un build ou une copie isolée. Elles ne
corrompent pas `build/`, ne réécrivent pas de dataset publié et ne modifient pas
une baseline. Une campagne longue, un rebaseline, une publication ou une
suppression de preuves exige une autorisation explicite.

## Unités de revue

Inventorier exhaustivement les capacités, puis regrouper seulement les wrappers
qui partagent réellement :

- le même moteur ;
- la même transition ou formule ;
- la même politique produit et le même calendrier ;
- la même stratégie de préparation et de réduction ;
- les mêmes types, précision et frontières numériques.

Une différence de dynamique, mesure, schéma, payoff, calendrier, exercice,
RNG, solveur ou topologie CUDA crée une unité de revue distincte. Lire mille
wrappers générés identiques n'apporte pas mille preuves ; ignorer une branche
mathématique distincte au titre de la représentativité n'apporte aucune preuve.

## Couverture, preuve et verdict

Chaque axe et chaque famille examinée reçoit séparément :

| Champ | Valeurs |
|---|---|
| Couverture | `complète`, `partielle`, `non examinée`, `non applicable` |
| Verdict | `conforme`, `non conforme`, `indéterminé` |
| Preuve | chemins, commandes, inputs, résultats et limites |

Une couverture complète peut conclure `non conforme`. Une preuve locale peut
établir une non-conformité malgré une couverture partielle. Une absence
d'erreur sur un échantillon ne donne jamais une couverture complète.

Classer la force des preuves :

1. **indépendante** : autre dérivation, autre implémentation reconnue ou identité
   mathématique qui ne réemploie pas le code testé ;
2. **croisée** : deux chemins de calcul réellement distincts ;
3. **contractuelle** : invariant, domaine ou propriété structurale directement
   vérifié ;
4. **régression** : résultat comparé à une sortie antérieure compatible ;
5. **diagnostique** : observation utile qui ne suffit pas à conclure.

Deux façades du même cœur numérique ne sont pas deux références indépendantes.
Un test bit à bit prouve une identité d'exécution, pas l'exactitude financière
du cœur partagé.

## Parcours d'un passage complet

1. Fixer le snapshot, le mandat, les exclusions et les expériences autorisées.
2. Construire les inventaires de capacités, sources, recettes, targets et
   artefacts sans prendre l'un d'eux comme vérité unique.
3. Auditer la véracité de bout en bout avant d'évaluer l'architecture ou la
   vitesse.
4. Auditer factorisation et navigation sur les mêmes familles représentatives.
5. Auditer performance, CMake et hygiène avec leurs preuves propres.
6. Rechercher une explication contractuelle et un contre-exemple avant de créer
   chaque constat.
7. Mettre à jour le tableau de bord et les constats, y compris lorsqu'aucune
   anomalie n'est trouvée.

---

# Axe I — Véracité des quantités calculées

**Question :** pour chaque entrée annoncée, la sortie correspond-elle au modèle,
au produit, à la méthode et à la dérivée déclarés ?

## 1. Inventaire mathématique

Construire une matrice au minimum selon :

```text
engine × modèle × produit × construction × ordre de dérivation × particularité
```

Les particularités comprennent notamment :

- transition exacte ou schéma temporel ;
- terminal, dépendant du chemin ou exercice anticipé ;
- closed form scalaire ou coopérative ;
- MC markovien, LSM, rough FFT ou rough N-facteurs ;
- diffusion continue, sauts à activité finie ou changement de temps ;
- courbe ajustée, mesure risque-neutre ou mesure forward ;
- gradient, Hessienne diagonale ou dérivées mixtes.

Confronter manifeste typé, bindings, fichiers physiques, codegen, catalogue et
graphe CMake. Toute capacité annoncée sans chemin exécutable, ou tout chemin
exécutable absent de l'inventaire, doit être expliqué.

## 2. Modèles et dynamiques

Pour chaque dynamique distincte, vérifier :

- définition du processus, unités, mesure et numéraire ;
- drift, diffusion, corrélations, compensateurs et actualisation ;
- loi initiale, états simulés et états observables ;
- domaine des paramètres et traitement des frontières ;
- transition exacte, discrétisation, ordre et biais attendu ;
- cas dégénérés et limites connues ;
- martingale ou premier moment lorsque le contrat l'exige ;
- cohérence entre analytics, pricing, sampling et gradients.

Pour les modèles à sauts, examiner séparément compte, temps, marques,
compensateur, consommation aléatoire et cas d'intensité nulle ou forte. Une loi
ayant la bonne marginale mais un mauvais couplage peut produire un prix correct
et une dérivée incorrecte.

Pour les modèles rough, distinguer modèle continu, approximation hybride,
convolution FFT et lift N-facteurs. Vérifier noyau, singularité, poids,
padding, longueur FFT, reconstruction et approximation temporelle. Une dérivée
du modèle approché ne doit pas être présentée comme dérivée exacte du modèle
continu sans qualification.

## 3. Produits, calendriers et exercice

Vérifier :

- paramètres, conventions de côté, unités et domaines ;
- payoff terminal, observations, états de chemin et cashflows ;
- calendrier, jours/années, conventions de grille et maturité terminale ;
- discounting, courbe, numéraire et mesure cohérents ;
- barrières, coupons, mémoire, autocall, absorption et cas limites ;
- mapping call/put, payer/receiver et signes ;
- politique Longstaff--Schwartz, bases, régressions et décision d'exercice.

Pour un calendrier gelé, documenter précisément ce qui reste fixe et ce que le
bump de maturité modifie. Pour l'exercice gelé des sensibilités, distinguer la
quantité effectivement estimée d'un repricing avec politique recalibrée.

## 4. Moteurs de pricing

### Closed form

Retracer paramètres, formule, solveur éventuel, conditions d'existence et
branche numérique. Comparer à une dérivation ou référence indépendante sur cas
ordinaires, frontières et limites. Pour Jamshidian, contrôler racine,
monotonicité, décomposition, échec du solveur et coopération du bloc.

### Monte Carlo markovien

Retracer préparation, innovations, transition, payoff, réduction, erreur
standard et finalisation. Vérifier que le nombre de chemins, la grille et le
batching annoncés sont ceux exécutés. Contrôler biais temporel, convergence en
chemins, premiers moments et invariance aux découpages promis.

### Exercice anticipé

Retracer simulation forward, états de régression, sélection des candidats,
solveur, continuation, décision et réduction. FP64, rang, conditionnement et
causes d'échec doivent être visibles. Une politique gelée, un refit et une
politique centrale représentent des estimateurs différents.

Pour chaque option américaine et swaption bermudéenne, auditer séparément les
deux replays de sensibilité :

- **`frozen_exercise_time`** : le nœud bumpé garde l'indice d'exercice du
  chemin central ; son état, payoff et facteur d'actualisation sont ceux du
  nœud. Contrôler aussi la décision initiale éventuelle à `t0` et le cas
  terminal ; aucune régression bumpée n'est évaluée.
- **`frozen_regression_policy`** : conserver coefficients, statuts du solveur,
  bases et normalisations du fit central ; avancer chaque nœud avec ses propres
  états et payoffs, puis prendre la première date où l'exercice est préféré.
  Contrôler les candidats, les échecs de régression, les égalités au seuil,
  l'exercice à `t0` et la dernière date ; aucun refit bumpé n'est effectué.

Vérifier sur les deux stratégies CUDA le même estimateur pathwise, les mêmes
innovations lorsque le couplage est annoncé, les prix centraux, les moments et
leurs erreurs. Tester des chemins construits où le bump change la date
d'exercice, où il ne la change pas, et où le central exerce à `t0`. Ajouter une
référence de décision et de payoff indépendante du cœur GPU, puis distinguer
les deux replays du repricing avec politique refittée et de la dérivée de la
valeur optimale. Qualifier séparément le biais de gel et l'effet des seuils
discontinus, notamment pour les Hessiennes.

Pour le prix LSM central, distinguer les chemins utilisés pour ajuster la
régression de ceux utilisés pour valoriser la politique. Si les mêmes chemins
servent aux deux opérations, vérifier et annoncer la portée de l'erreur
standard publiée : la dispersion des cashflows ne mesure pas à elle seule
l'incertitude liée à l'apprentissage de la politique.

### Rough

Retracer préparation des coefficients, génération des innovations, FFT,
convolution, reconstruction du processus, payoff et réduction. Les FFT ou
coefficients réutilisés entre nœuds doivent être mathématiquement identiques
pour ces nœuds.

## 5. Prix, gradients et Hessiennes

Suivre explicitement la chaîne :

```text
SensitivityRequest
→ SensitivitySpec
→ lecture du paramètre central
→ règle de bump et domaine
→ SensitivityNodes
→ SensitivityStencil
→ valeurs des nœuds sous innovations couplées
→ reconstruction
→ valeur et erreur statistique publiées
```

Auditer **séparément les deux topologies de calcul** : la voie `mono`, où
chaque tâche prépare ses nœuds, les valorise et reconstruit sa sensibilité ; la
voie `node_graph`, où les nœuds utiles sont partagés entre sensibilités et où
leurs valeurs sont reconstruites ensuite. Relever les lancements réels du code
pour chaque moteur : préparation, évaluation, accumulation des moments et
finalisation peuvent être fusionnées ou séparées selon la famille. Le nom
« trois kernels » ne suffit pas à prouver trois phases séparées.

Pour chacune, contrôler le mapping ligne/sensibilité/nœud/chemin, les écritures
uniques, la déduplication, la validité des stencils et le traitement des lignes
invalides. Comparer les valeurs **pathwise** des nœuds, puis leurs moments et
erreurs avant la sortie agrégée ; vérifier tailles de chunks, shards et
frontières de batch. Une parité de sorties entre voies qui partagent le même
stencil ou payoff ne prouve pas leur justesse commune.

Vérifier :

- sélection arbitraire de paramètres modèle, produit et courbe ;
- absence de dérivée implicite pour un paramètre non demandé ;
- bumps absolus, relatifs et déplacés ;
- central, nœuds symétriques et stencils unilatéraux aux frontières ;
- poids d'ordre un, diagonal et mixte ;
- déduplication des nœuds du graphe ;
- reconstruction d'une requête sparse sans calculs étrangers ;
- erreurs standards et covariance cohérentes avec le couplage ;
- mêmes innovations lorsque le CRN est promis ;
- absence de réutilisation d'une FFT ou d'une trajectoire lorsque le paramètre
  bumpé change effectivement cette primitive.

Cas obligatoires :

- prix seul ;
- prix + gradient ;
- prix + gradient + Hessienne diagonale ;
- sélection mixte sparse et complète lorsqu'elle est supportée ;
- maturité `T` pour les produits où elle est déclarée ;
- paramètres de saut et de marques lorsqu'ils sont déclarés ;
- paramètres proches des frontières ;
- calendriers discrets et exercice anticipé avec leurs exclusions explicites.

Lorsque le contrat le promet, comparer bit à bit le central entre prix seul,
mono-kernel et graphe de nœuds, puis gradient et diagonale entre stratégies.
Cette parité ne remplace pas une référence de dérivée.

Qualifier les différences finies en séparant :

- biais du stencil ;
- biais de discrétisation du pricer ;
- bruit Monte Carlo ;
- cancellation FP32 ;
- effet du domaine et du clipping ;
- biais de politique gelée.

Utiliser selon le cas dérivée analytique, différentiation indépendante,
identité financière, convergence multi-bumps et plusieurs seeds. Choisir le
bump avant de regarder le résultat de production, ou conserver tous les essais
et la règle de sélection.

## 6. RNG et reproductibilité

Vérifier le mapping Philox complet : seed, ligne, trajectoire, pas, groupe et
domaine. Prouver :

- injectivité sur le domaine annoncé ;
- absence de collision et dépassement ;
- stabilité aux batchs et géométries lorsqu'elle est promise ;
- isolation des consommations variables seulement lorsqu'elle est nécessaire ;
- priorité au flux unique lorsque l'ordre fixe suffit ;
- couplage correct des nœuds et indépendance correcte des répétitions ;
- provenance versionnée des changements de mapping.

Un même seed textuel ne prouve pas l'emploi des mêmes innovations.

## 7. Robustesse numérique

Examiner valeurs non finies, cancellation, overflow/underflow, solveurs,
conditionnement, fonctions spéciales et accumulation. Les échecs doivent être
classés et propagés ; ne pas transformer silencieusement une erreur en prix
plausible.

FP32 est le format naturel des états et simulations massives lorsque le budget
le permet. FP64 reste justifié pour moments, réductions sensibles, régressions,
solveurs et références. Auditer aussi bien le FP64 inutile que la réduction de
précision dangereuse.

## Preuve minimale de l'axe

- matrice de capacités et de particularités couverte ;
- trace complète de représentants justifiés ;
- références indépendantes ou identités adaptées à chaque cœur mathématique ;
- cas ordinaires, frontières, stress et dégénérescences ;
- convergence et erreurs séparées pour les méthodes approximatives ;
- tests de parité uniquement là où le contrat la promet ;
- exclusions et quantités non qualifiées explicitement nommées.

---

# Axe II — Factorisation

**Question :** les briques correspondent-elles aux invariants du domaine, sans
duplication métier ni abstraction artificielle ?

## 1. Frontières attendues

| Zone | Responsabilité |
|---|---|
| `src/common` | primitives numériques, CUDA et moteurs réutilisables sans modèle concret |
| `src/model/**/<model>` | paramètres, domaine, dynamics, analytics et sampling du modèle |
| `src/product/<product>` | paramètres, calendrier, payoff, observations et règles produit |
| `src/model/**/product` | composition mince du modèle et du produit |
| `src/curve/<curve>` | paramètres et analytics de courbe |
| `learning` | briques d'apprentissage réutilisables, sans campagnes particulières |
| `tools` | opérations offline réutilisables sur les bibliothèques de `src` |
| `catalog` | identité permanente d'une base et générateur minimal |
| `work` | campagnes, expériences et builds locaux supprimables |

Les dépendances doivent aller des compositions vers les briques, jamais d'une
primitive commune vers un modèle, produit, recette ou expérience concret.

## 2. Partage entre prix et dérivées

Prix, gradients et Hessiennes doivent partager lorsque cela est vrai :

- paramètres préparés et domaines ;
- dynamics et transitions ;
- génération des innovations ;
- payoff, observations, cashflows et calendrier ;
- discounting et réduction statistique ;
- contrats de résultat et provenance.

La couche de dérivation possède :

- sélection des paramètres ;
- `SensitivitySpec` et règles de bump ;
- nœuds et stencils ;
- couplage CRN ;
- reconstruction et sorties de dérivées.

Elle ne recopie pas dynamics ou payoff. Inversement, le moteur prix seul ne
doit pas supporter artificiellement toutes les structures de Hessienne.
Closed form, MC mono, graphe multi-kernels, LSM et rough peuvent garder des
pipelines différents tout en partageant les mêmes objets métier.

La philosophie de préparation doit rester reconnaissable : modèle central,
produit central, courbe éventuelle, spécification demandée, besoin du central,
contexte, puis ligne prête à calculer. Toute représentation matérialisée en
masse doit être justifiée par un besoin de calcul ou de mémoire.

## 3. Granularité des briques

Rechercher :

- logique métier dupliquée entre `.cu`, headers, templates et générateurs ;
- tables de capacités parallèles ;
- politiques presque identiques séparées par modèle ;
- adaptateurs qui connaissent trop de détails du moteur ;
- helpers génériques dont un seul modèle comprend réellement le contrat ;
- concepts qui valident moins que ce que leur consommateur exige ;
- templates utilisés seulement pour éviter une fonction ordinaire ;
- paramètres booléens qui cachent plusieurs stratégies ;
- façades et alias sans consommateur ou sans réduction de complexité.

Une bonne brique possède un invariant exprimable, plusieurs consommateurs
plausibles ou une frontière publique claire. Le nombre minimal de lignes ou de
fichiers n'est pas un critère.

## 4. Codegen et catalogue

Le manifeste typé possède les capacités et compositions. Le renderer transforme
ces déclarations ; les templates possèdent les corps répétés ; CMake consomme le
résultat. Vérifier qu'aucune liste manuelle concurrente ne décide des modèles,
produits, ordres de dérivation, RNG ou recettes.

Les variantes disponibles en bibliothèque ou codegen n'ont pas toutes besoin
d'une recette permanente. Le catalogue doit rester une sélection lisible et
justifiée des datasets réellement destinés à être produits.

## 5. Exercices d'extension

Mesurer le coût conceptuel et les fichiers touchés pour :

1. ajouter un paramètre modèle dérivable ;
2. ajouter un paramètre produit ou courbe ;
3. ajouter un produit terminal ;
4. ajouter un produit dépendant du chemin ;
5. ajouter un modèle MC ;
6. ajouter une formule fermée ou coopérative ;
7. ajouter une stratégie qui exige de nouvelles innovations ou FFT.

Chaque exercice doit aboutir à un propriétaire évident. Toucher de nombreuses
listes sans lien métier indique une factorisation insuffisante. Modifier un
moteur universel pour une exception locale indique une abstraction trop large.

## Preuve minimale de l'axe

- graphe des responsabilités et dépendances ;
- recherche de duplication sémantique, pas seulement textuelle ;
- comparaison prix/gradient/Hessienne sur plusieurs moteurs ;
- inspection des sources de vérité codegen/CMake/catalogue ;
- exercices d'extension documentés ;
- coût en lisibilité et performance considéré avant toute recommandation.

---

# Axe III — Arborescence et lisibilité

**Question :** un contributeur qui découvre le projet trouve-t-il rapidement la
bonne responsabilité et comprend-il les symboles qu'il ouvre ?

## 1. Parcours de découverte obligatoires

Chronométrer ou décrire les étapes nécessaires pour :

1. trouver la dynamique et le domaine d'un modèle ;
2. trouver paramètres, calendrier et payoff d'un produit ;
3. comprendre comment un couple modèle-produit est assemblé ;
4. suivre un prix depuis une recette jusqu'au kernel ;
5. suivre un gradient depuis la sélection jusqu'à la reconstruction ;
6. trouver la cible CMake et l'exécutable d'un générateur ;
7. trouver comment lancer, reprendre et suivre une génération ;
8. trouver les limites numériques et les preuves existantes.

Un parcours qui exige une mémoire historique, une recherche par ancien nom ou
l'ouverture de plusieurs façades ambiguës révèle un défaut de navigation.

## 2. Arborescence

Vérifier :

- une responsabilité claire par racine et par dossier ;
- taxonomie cohérente entre `src`, `catalog` et `datasets` ;
- `learning` réservé au code réutilisable et `work/experiments` aux campagnes ;
- symétrie utile entre modèles comparables, sans dossiers vides artificiels ;
- produits absents des dossiers de primitives communes ;
- code runtime absent des recettes et expériences ;
- tests rangés par contrat ou domaine, sans racine fourre-tout ;
- profondeur justifiée et absence de chaînes `common/core/utils` vagues ;
- README local lorsqu'un dossier constitue une vraie porte d'entrée.

Toute racine supplémentaire, notamment site compagnon ou collection d'articles,
doit avoir un propriétaire, une politique de distribution et une raison de
cohabiter avec le dépôt principal.

## 3. Naming

Employer des noms précis et cohérents :

- `snake_case` pour fichiers, variables et fonctions ;
- `PascalCase` pour types et concepts ;
- suffixes révélant rôle et phase : `parameters`, `dynamics`, `analytics`,
  `pricing_policy`, `device_preparation`, `workspace`, `result` ;
- unités visibles : `_days`, `_years`, `_bytes`, `_count`, `_offset` ;
- cardinalités et ownership explicites ;
- mêmes termes pour les mêmes objets entre prix et dérivées ;
- aucun nom permanent tel que `new`, `final`, `misc`, `helper2` ou `workbench`
  sans signification de domaine.

Le chemin complet peut fournir le contexte ; répéter tous ses mots dans chaque
symbole nuit aussi à la lecture. Les notations mathématiques courtes restent
acceptables lorsqu'elles sont définies localement.

## 4. Taille et regroupement des fichiers

Un fichier doit porter une responsabilité explicable. Fusionner des fichiers
lorsqu'ils :

- évoluent toujours ensemble ;
- ne possèdent pas de contrat autonome ;
- ne font que transférer ou renommer un symbole ;
- obligent le lecteur à naviguer sans séparer compilation ou interface.

Séparer lorsqu'ils :

- ont des consommateurs ou cycles de changement différents ;
- isolent interface publique, implémentation device ou unité compilée ;
- évitent une dépendance lourde ou une instanciation massive ;
- représentent des algorithmes ou stratégies distincts.

Ni une limite arbitraire de lignes ni une préférence esthétique ne suffit.

## Documentation et parcours de découverte

Contrôler flux de contrôle, portée des variables, conversions, invariants,
commentaires et messages d'erreur. Les commentaires expliquent conventions,
hypothèses et choix ; ils ne paraphrasent pas le code.

Le README racine doit conduire à un premier résultat, l'index documentaire doit
orienter par besoin, et chaque contrat doit avoir une autorité unique. Rechercher
liens cassés, commandes obsolètes, anciens noms, inventaires manuels concurrents
et documents qui racontent l'historique au lieu d'expliquer l'état présent.

## Preuve minimale de l'axe

- parcours de découverte exécutés ;
- inventaire des racines et propriétaires ;
- cohérence chemins/namespaces/symboles ;
- candidats de fusion ou séparation justifiés par responsabilité ;
- liens, exemples et commandes vérifiés ;
- absence de recommandation fondée sur le seul goût personnel.

---

# Axe IV — Performance

**Question :** le coût complet est-il adapté aux workloads utiles sans compromis
numérique caché ?

Le [protocole de performance](../performance-regression-protocol.md) possède les
règles détaillées de campagne, statistiques, admission et rebaseline. Le
présent axe définit ce qu'un audit doit examiner ; il ne recopie pas ce
protocole.

## 1. Ordre d'analyse

Toujours examiner dans cet ordre :

1. complexité algorithmique et travail répété ;
2. découpage du pipeline et réutilisation ;
3. topologie thread/warp/bloc ;
4. disposition et mouvements de mémoire ;
5. ressources compilées et précision ;
6. temps par phase et temps de bout en bout.

Un tuning local ne compense pas un algorithme qui recalcule les mêmes nœuds,
FFT, trajectoires, régressions ou payoffs sans nécessité.

## 2. Pipeline et stratégie CUDA

Pour chaque stratégie distincte, relever :

- unité de travail par thread, groupe, warp et bloc ;
- distribution trajectoires, nœuds, sensibilités et lignes ;
- kernels de préparation, simulation, régression, moments et finalisation ;
- synchronisations et dépendances entre phases ;
- batching sur lignes et chunking sur trajectoires ;
- travail dupliqué entre blocs ou relances ;
- saturation pour petites et grandes tailles ;
- comportement quand le nombre de paramètres ou de nœuds augmente.

Prix seul, mono-kernel gradient, graphe multi-kernels, LSM, closed form
coopérative et rough FFT ne sont pas supposés partager une topologie optimale.

## 3. Mémoire et accès

Examiner :

- coalescence par warp et ordre des dimensions ;
- AoS/SoA et alignement ;
- réutilisation register/shared/global ;
- conflits de banques, caches et divergence ;
- valeurs vivantes et lifetime des temporaires ;
- workspace persistant, mémoire transitoire et outputs ;
- allocations, copies host/device et sérialisation ;
- pic mémoire réel et marge de VRAM ;
- layout des nœuds et accès de reconstruction.

Une réduction de registres qui augmente fortement le trafic global n'est pas
une amélioration automatique.

## 4. Ressources compilées

Pour chaque kernel effectivement lancé, attacher au binaire et à la géométrie :

- registres par thread ;
- stack et mémoire locale ;
- spills loads/stores ;
- shared statique et dynamique ;
- occupation théorique et mesurée ;
- taille de code et instanciations ;
- instructions FP32/FP64, divisions et fonctions spéciales.

Croiser diagnostics compilateur, attributs runtime et profilage. Un registre
élevé ou une occupation faible déclenche une investigation ; il ne prouve pas
une régression. `__launch_bounds__`, limites de registres et spécialisations
matérielles restent configurables, documentés et mesurés sur le profil concerné.

## 5. Précision et performance

Localiser tout FP64 exécuté dans les boucles massives. Pour chaque usage,
identifier nécessité numérique, coût et alternative. Conserver FP64 lorsqu'il
protège moments, réductions, régressions ou solveurs. Refuser `--use_fast_math`
ou toute approximation qui change un contrat sans budget et preuve explicites.

Comparer les stratégies avec les mêmes modèles, produits, inputs, chemins,
grilles, facteurs, seeds et sorties. Une FFT plus courte, moins de facteurs ou
un autre bump représente une autre approximation.

## 6. Mesures

Conserver séparément :

- temps GPU par phase ;
- enveloppe CUDA multi-kernels ;
- appel public et synchronisation ;
- pipeline avec préparation et transferts ;
- génération complète avec sérialisation et checkpoints.

Warmups, répétitions, conditions matérielles, médiane, p95, variabilité et
échecs restent conformes au protocole. Une mesure sur batterie ou sous
throttling peut être diagnostique, jamais une qualification de production.

## 7. Familles obligatoires

Lorsque présentes dans le mandat, couvrir séparément :

- closed form scalaire et coopérative ;
- MC terminal exact et à pas fixes ;
- produits dépendant du chemin ;
- LSM equity et fixed income ;
- modèles à sauts et consommations variables ;
- rough FFT et N-facteurs ;
- samples à un et plusieurs chemins par paramètre ;
- prix, gradient/diagonale et graphe de Hessienne complète.

## Preuve minimale de l'axe

- inventaire des stratégies et phases ;
- ressources de chaque kernel actif ;
- profils mémoire et accès ;
- comparaisons numériques recevables ;
- chronos complets et par phase ;
- conclusion limitée au GPU, toolchain et workload mesurés.

---

# Axe V — CMake

**Question :** le graphe de build représente-t-il exactement les capacités et
dépendances du projet, sans compilation ou target parasite ?

## 1. Configuration et portabilité

Vérifier une configuration propre pour chaque profil annoncé :

- versions C++ et CUDA réellement supportées ;
- architectures CUDA et profils de tuning ;
- présence et absence des dépendances optionnelles ;
- tests activés/désactivés selon le contrat ;
- diagnostics clairs pour une configuration non supportée.

Distinguer compilation, runtime testé, qualification numérique et performance.
Un preset SM89 n'est pas une garantie universelle ni une interdiction des autres
GPU.

## 2. Ownership des targets

Chaque target possède :

- un nom prévisible ;
- un module CMake propriétaire ;
- des sources exactes ;
- des dépendances minimales ;
- un output identifiable ;
- des agrégats et labels cohérents.

Contrôler `PUBLIC`, `PRIVATE` et `INTERFACE`, includes, définitions et options.
Une source compilée plusieurs fois doit correspondre à une variante volontaire.
Aucune unité autonome, instanciation ou symbole requis ne doit dépendre d'un
include textuel accidentel.

## 3. Manifeste, codegen et catalogue

Les targets générées dérivent du manifeste et des recettes. Vérifier dans les
deux sens :

- chaque recette possède une cible ;
- chaque cible de générateur possède une recette ;
- chaque binding déclaré possède les unités nécessaires ;
- aucun ancien chemin ou target ne survit à une régénération ;
- les agrégats ne construisent que leur famille ;
- les expériences locales restent hors graphe par défaut.

Les regex de chemin peuvent router une cible ; elles ne doivent pas devenir une
seconde définition des capacités métier.

## 4. Incrémentalité

Dans un build isolé, vérifier :

- build propre ;
- second build sans travail ;
- modification d'un modèle ;
- modification d'un produit ;
- modification d'une courbe ;
- modification d'une primitive partagée ;
- modification d'un template ;
- modification du manifeste ou d'une recette.

Pour chaque mutation, comparer targets attendues et targets réellement
reconstruites, temps, taille et relink. Les fichiers lus à la configuration
doivent déclencher la régénération CMake appropriée.

## 5. Coût du build

Examiner explosion des instanciations, unités trop larges, headers lourds,
compilation répétée, taille des objets et liens. Le cache de compilation peut
être mesuré, mais son absence ne doit pas masquer un graphe incorrect.

## Preuve minimale de l'axe

- configurations propres annoncées ;
- inventaire sources/targets/outputs ;
- bijection manifeste-recettes-targets ;
- matrice d'incrémentalité ;
- targets orphelines et doublons recherchés ;
- coût et diagnostics du build documentés.

---

# Axe VI — Hygiène et artefacts

**Question :** chaque élément du dépôt et des espaces locaux a-t-il un
propriétaire, un consommateur et une politique de conservation ?

## 1. Catégories à distinguer

| Catégorie | Exigence |
|---|---|
| Source suivie | nécessaire, possédée et reproductible |
| Sortie générée suivie | source de vérité et règle de régénération explicites |
| Dataset/recette | taxonomie, provenance et statut cohérents |
| Build local | recréable, identifiable et non confondu avec une preuve |
| Campagne sous `work` | interrompable, reprenable et supprimable |
| Preuve sous `artifacts` | provenance, hash et durée de conservation |
| Cache | ignoré et supprimable sans perte de contrat |

Ne pas supprimer une preuve comme un cache. Ne pas conserver un cache comme une
preuve.

## 2. Pollution du dépôt suivi

Rechercher notamment :

- exécutables, objets, bibliothèques, core dumps et caches Python ;
- logs, fichiers temporaires, sorties de profiler et notebooks checkpoint ;
- datasets générés au mauvais endroit ;
- anciennes taxonomies, recettes ou métadonnées ;
- secrets, chemins absolus personnels et configurations d'IDE imposées ;
- fichiers générés modifiés à la main ou divergents du codegen ;
- documents, templates et fixtures sans propriétaire.

Une exception suivie doit être intentionnelle, documentée et testée.

## 3. Code et fichiers orphelins

Croiser plusieurs preuves :

- manifeste et inventaires physiques ;
- graphe CMake et `compile_commands.json` ;
- includes et liens ;
- imports Python ;
- templates appelés par les renderers ;
- recettes et générateurs ;
- documentation et tests.

Une absence trouvée par recherche textuelle seule ne prouve pas qu'un template,
une spécialisation ou une entrée dynamique est morte. Inversement, la seule
présence d'un include ne justifie pas un fichier sans contrat.

Identifier :

- headers, sources, scripts, templates et fonctions sans consommateur ;
- dossiers vides ou façades sans responsabilité ;
- tests jamais enregistrés ;
- targets sans appelant ;
- anciens launchers, recettes et compatibilités sans usage actif ;
- copies et variantes historiques qui devraient vivre dans une preuve.

## 4. Builds et exécutables périmés

Inventorier `build/`, autres `build-*`, sous-builds de campagnes et exécutables.
Pour le build principal :

- le cache correspond-il au source et au profil actuels ?
- le target existe-t-il encore dans le graphe courant ?
- `ninja -n` ou l'équivalent annonce-t-il une reconstruction ?
- l'exécutable est-il celui que la recette ou la commande va lancer ?
- son binaire et ses inputs sont-ils hashés dans une campagne en cours ?

La date du fichier seule ne prouve pas sa fraîcheur. Un exécutable de campagne
gelée peut rester volontairement ancien ; il appartient alors à cette campagne,
pas au build actif.

## 5. Racines locales

Le [plan des artefacts locaux](../local-artifacts.md) définit le rôle de
`build`, `work`, `datasets` et `artifacts`. Vérifier :

- absence de builds concurrents non documentés ;
- campagnes terminées, interrompues ou abandonnées clairement identifiables ;
- checkpoints associés à leur recette et leur source gelée ;
- aucun helper expérimental devenu dépendance cachée de `src`, `learning`,
  `tools` ou `catalog` ;
- tailles et rétention compatibles avec leur utilité.

L'audit propose une liste de suppression avec justification. Il ne supprime pas
un fichier local ou historique sans vérifier qu'il n'est ni une preuve ni un
checkpoint nécessaire.

## Preuve minimale de l'axe

- inventaire suivi/ignoré/local avec tailles et propriétaires ;
- régénération comparée des outputs suivis ;
- recherche d'orphelins croisant build, imports, includes et manifeste ;
- fraîcheur des exécutables actifs vérifiée par le graphe ;
- liste séparée des suppressions sûres, preuves à conserver et cas indéterminés.

---

# Moyens de preuve transversaux

## Tests

Pour chaque test cité, préciser ce qu'il démontre et ce qu'il ne démontre pas.
Rechercher :

- assertions qui recalculent le résultat avec le même code ;
- tests sans branche négative ;
- tolérances choisies après observation ;
- fixtures non représentatives ;
- tests jamais enregistrés dans CTest ou la suite Python ;
- skips silencieux et succès sans assertions utiles ;
- tests bit à bit employés à tort comme référence mathématique.

Une suite doit couvrir les contrats distincts, pas reproduire mécaniquement
l'arborescence de `src`.

## Codegen

Effectuer une régénération propre dans un répertoire isolé et comparer au dépôt.
Contrôler aussi les suppressions attendues : un générateur qui écrit les
nouveaux fichiers sans détecter les anciens chemins n'est pas idempotent au
niveau du dépôt.

Tester manifestes invalides, collisions, capacités absentes, ordres de
dérivation et associations modèle-produit-courbe. Les templates générés restent
revus par famille d'algorithme.

## Sanitizers et analyse statique

Employer selon le mandat :

- warnings compilateur et erreurs de link ;
- ASan/UBSan pour les chemins host ;
- Compute Sanitizer `memcheck`, `racecheck`, `initcheck`, `synccheck` pour des
  représentants justifiés ;
- analyse des ressources compilées et symboles ;
- schémas et validateurs de métadonnées.

Un sanitizer propre sur un seul kernel ne certifie pas les autres stratégies.
Une erreur de test ou d'instrumentation doit être distinguée d'une erreur du
runtime, avec preuve.

## Références historiques

`artifacts`, rapports de performance et constats fermés peuvent éviter des
campagnes redondantes. Avant réemploi, vérifier snapshot, hashes, toolchain,
inputs, stratégie et domaine. Une preuve incompatible reste historique.

---

# Constats et restitution

## Registres

- [status.md](status.md) : tableau de bord du **dernier passage**, remplacé à
  chaque nouvel audit complet ou ciblé ;
- [response.md](response.md) : constats ouverts uniquement ;
- [unresolved-closures.md](unresolved-closures.md) : constats clos
  administrativement par exclusion de sorties ou retrait de capacité,
  dont le problème numérique demeure ;
- [closed.md](closed.md) : constats réglés par correction vérifiée, contrat
  explicite, décision de périmètre ou compromis accepté. Les limites connues
  et les mesures défavorables restent dans chaque entrée.

Les longues preuves résident dans leurs rapports ou `artifacts`; les registres
les résument et les lient. `status.md` ne doit plus accumuler tous les passages
historiques.

## Format d'un constat

Chaque constat contient :

- identifiant stable et titre précis ;
- axe principal et axes secondaires ;
- état, date et propriétaire ;
- sévérité, priorité et confiance ;
- contrat et localisation exacte ;
- signature reproductible et preuve ;
- conséquence concrète et portée ;
- explications alternatives examinées ;
- correction minimale proposée ;
- critère de clôture vérifiable ;
- limites et condition de réouverture.

Lire `closed.md` et `unresolved-closures.md`, puis rechercher les signatures
avant d'ouvrir un identifiant. Une correction vérifiée, un contrat qui règle
le constat dans son périmètre, ou un compromis accepté transfère le constat
vers `closed.md`, avec les preuves défavorables et les limites ; l'acceptation
ne change pas un gate général. Si une sortie est exclue ou une capacité
retirée parce que son calcul reste incorrect ou incertain, le constat va dans
`unresolved-closures.md`. Après
résolution et vérification, **déplacer** son entrée vers `closed.md` et la
supprimer de `unresolved-closures.md` ; ne jamais la dupliquer. Une régression
reprend son identifiant historique. Un manque de preuve va d'abord
dans la couverture ; il devient constat seulement lorsqu'il contredit une
obligation ou crée un risque concret.

## Sévérité

| Niveau | Sens |
|---|---|
| critique | prix, dérivée, données ou sécurité matériellement faux sur le chemin principal |
| haute | capacité importante incorrecte, architecture bloquante ou risque fort prouvé |
| moyenne | défaut borné de maintenance, performance, portabilité ou preuve |
| basse | défaut local réel avec impact limité |

La priorité tient compte de la sévérité, du chantier en cours et du coût de
correction. La confiance distingue `prouvée`, `élevée`, `moyenne` et
`hypothèse`.

## Tableau de bord attendu

`status.md` contient au minimum :

1. snapshot et mandat ;
2. matériel, toolchains et exclusions ;
3. matrice couverture/verdict des six axes ;
4. familles et représentants examinés ;
5. commandes et preuves durables ;
6. constats ouverts, fermés, qualifiés avec limite active ou reclassés
   pendant le passage ;
7. limites qui interdisent une conclusion globale ;
8. conclusion concise.

Une conformité globale n'est possible que si tous les axes applicables sont
couverts complètement et conformes. Sinon, le projet reçoit un verdict global
`non conforme` ou `indéterminé`, avec les conformités locales explicitement
préservées.

## Discipline de conclusion

- ne pas imposer un nombre minimal de constats ;
- distinguer défaut, risque à mesurer, dette historique et choix à conserver ;
- ne pas proposer une refactorisation sans bénéfice de contrat ou de lecture ;
- ne pas publier un gain sans coût complet et recevabilité numérique ;
- ne pas fermer un constat sur la seule compilation ;
- ne pas transformer une limitation connue en bug sans contre-exemple ;
- conserver les résultats négatifs et les essais refusés.

Le résultat attendu n'est pas un dépôt qui paraît uniforme. C'est un dépôt dont
les quantités, les responsabilités, les coûts et les preuves sont explicites et
vérifiables.
