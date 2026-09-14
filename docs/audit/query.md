# Référentiel des audits du dépôt C++/CUDA

Version du référentiel : **9.2 — 2026-09-13**.

La v9.2 précise l'organisation du build et des preuves locales. Elle ne
modifie pas les critères numériques, de tests, de portabilité et de performance
ci-dessous, ni les verdicts consignés lors des audits précédents.

## Mission

Vérifier qu'un contributeur peut comprendre le projet, retrouver les
responsabilités et modifier un modèle ou un produit sans ambiguïté ; que les
calculs implémentent les contrats mathématiques et financiers annoncés ; et
que le partage du code simplifie réellement sa compréhension et son évolution,
sans coût numérique ou informatique injustifié.

La simplicité se juge du point de vue du lecteur et du contributeur, pas au
nombre minimal de fichiers ou de lignes. La justesse précède l'optimisation.
La factorisation partage des invariants réels ; elle ne cherche ni un moteur
universel ni le maximum de templates. La performance reste une exigence forte,
étayée par des mesures sur les chemins effectivement utilisés.

Ce document définit les questions et les preuves de l'audit, pas ses résultats,
un plan d'implémentation détaillé ni une autorisation de refactorisation.

## Sommaire

- [Mandat et méthode](#mandat-et-méthode)
- [Exploration, structure et documentation](#exploration-structure-et-documentation)
- [Justesse des modèles et des produits](#justesse-des-modèles-et-des-produits)
- [Homogénéité et factorisation](#homogénéité-et-factorisation)
- [Qualité et organisation des tests](#qualité-et-organisation-des-tests)
- [Codegen et génération des datasets](#codegen-et-génération-des-datasets)
- [Robustesse numérique et reproductibilité](#robustesse-numérique-et-reproductibilité)
- [Exécution CUDA et sécurité mémoire](#exécution-cuda-et-sécurité-mémoire)
- [CMake, dépendances et portabilité](#cmake-dépendances-et-portabilité)
- [Performance](#performance)

## Mandat et méthode

### Périmètre et autorisations

L'audit principal couvre `src`, `tools`, `tests`, `catalog`, `cmake`, les
CMake racine et la documentation maintenue. Il contrôle aussi le build
principal `build/` lorsqu'il existe, les chemins et schémas locaux de
`datasets`, et les preuves pertinentes sous `artifacts/` lorsqu'elles sont
disponibles. Un résultat historique n'est pas une exécution du code courant.
Pricing et sampling modèle-seul sont deux parcours de premier rang.

`validation/**` et `docs/validation/**` restent hors périmètre, y compris pour
leur structure, leurs noms et leurs abstractions. La certification indépendante
des prix, l'inventaire Premia/QuantLib et leurs caches appartiennent au
[référentiel de validation](../validation/query.md). Les références locales des
tests de composant restent dans l'audit principal. Un consommateur externe peut
être lu pour comprendre une interface ; cela n'autorise ni son audit interne
ni l'exécution implicite de ses validations.

Les registres `docs/audit/**` sont des preuves et instruments de suivi, pas des
documents à réécrire au titre du nettoyage documentaire. Le présent référentiel
ne change que sur demande explicite. Pendant un audit ordinaire, seules les
mises à jour de `status.md`, `response.md` et `closed.md` sont attendues.

Avant toute exécution, préciser le mandat, les axes couverts, le matériel
disponible et les expériences envisagées. Ne pas modifier code, recette,
baseline, budget, dataset publié ou réglage matériel sous prétexte de vérifier
une hypothèse. Le `build/` partagé ne sert pas aux mutations de l'audit : les
builds propres ou expérimentaux, sorties de tests, codegen et fixtures mutées
utilisent des emplacements temporaires isolés. Conserver les preuves utiles
sous `artifacts/` avec leur provenance, sans installer un second build
principal permanent. Une commande documentaire qui génère ou
publie n'est pas autorisée par sa seule présence dans un README.

Respecter le worktree partagé et coordonner l'accès au GPU. Une campagne
longue, une génération complète ou un rebaseline ne démarre pas implicitement
avec l'audit. Une impossibilité d'exécution limite la conclusion ; elle ne
supprime pas les vérifications statiques utiles.

### Parcours de l'auditeur indépendant

1. Lire les [entrées du projet](../../README.md), la
   [carte documentaire](../README.md), les contrats des composants examinés et
   les registres. Fixer le snapshot avant de tirer une conclusion.
2. Inventorier les capacités, chemins, propriétaires et dépendances. Confronter
   manifeste, fichiers physiques, codegen et graphe CMake dans les deux sens :
   aucun de ces inventaires ne prouve seul que les autres sont corrects.
3. Exécuter les parcours de découverte, puis suivre les calculs de bout en bout.
   Examiner profondément chaque implémentation mathématique distincte et
   chaque moteur commun ; contrôler les compositions et leurs exceptions.
4. Examiner ce que prouvent réellement les tests. Exécuter les contrôles
   ciblés et les mesures autorisées ; réutiliser les preuves antérieures
   seulement si leur domaine et leur provenance restent applicables.
5. Pour chaque anomalie, chercher une explication contractuelle et un
   contre-exemple à sa propre conclusion. Proposer la correction minimale,
   ou conclure qu'aucun changement n'est justifié.

Les chemins et capacités sont inventoriés exhaustivement. Les sorties générées
partagent une revue de leur propriétaire, des templates et du resolver, puis
un contrôle exhaustif de génération et des compositions représentatives.
Lire mille wrappers identiques ne fournit pas mille preuves indépendantes.
Inversement, une branche mathématique ou une exception manuelle distincte ne
peut être déclarée revue par analogie.

Documenter pourquoi chaque représentant couvre une famille et ce qu'il ne
couvre pas : moteur, transition, payoff, calendrier, état, mémoire et précision.
Étendre l'examen lorsqu'une différence invalide cette représentativité.

Si le travail est réparti entre spécialistes, fixer leurs frontières et leurs
livrables. Un coordinateur réconcilie les preuves, revoit les interfaces entre
axes et déduplique les causes. Il ne transforme pas plusieurs avis concordants
fondés sur la même source en validations indépendantes.

### Provenance, couverture et verdict

Le snapshot identifie révision, branche, état initial/final du worktree, index,
liste des fichiers non suivis et empreintes du contenu examiné. Sur un worktree
sale, le commit seul ne décrit pas le code. Conserver un diff et les contenus
non suivis nécessaires dans une archive reconstructible, ainsi que les hashes
des inputs, commandes, binaires, toolchain et résultats utilisés. Une preuve
temporaire perdue n'est plus une preuve durable. `artifacts/` peut contenir
d'anciens tests, logs, binaires, campagnes et snapshots ; vérifier leur
révision, configuration, domaine et intégrité avant de les réutiliser. Leur
présence ne prouve ni que le build actuel est à jour ni que le test repasse.

Pour chaque axe ou famille, `status.md` sépare obligatoirement :

| Dimension | Valeurs | Signification |
|---|---|---|
| Couverture | `complète`, `partielle`, `non examinée`, `non applicable` | Quelles questions et preuves ont effectivement été examinées ? |
| Verdict | `conforme`, `non conforme`, `indéterminé` | Que permettent de conclure ces preuves sur le périmètre indiqué ? |

Un audit **complet et non conforme** est possible : tous les contrôles ont été
menés et certains ont révélé des défauts. Un audit partiel peut aussi prouver
une non-conformité locale. Une erreur trouvée ne rend pas automatiquement
l'examen incomplet ; une absence d'erreur ne rend pas l'examen complet.

La couverture complète exige les examens applicables annoncés par cette
query sur le mandat déclaré, avec leurs résultats positifs ou négatifs.
Une preuve requise absente laisse la couverture correspondante partielle et
interdit le verdict positif qu'elle devait soutenir. Une exclusion ou un
échantillon réduit ne certifie jamais la partie omise. `Non applicable`
requiert un motif contractuel, pas simplement un outil indisponible.

Les anciens tableaux `P0-*` restent des enregistrements historiques. La v9
emploie ces deux dimensions ; les conditions de recevabilité des preuves
(snapshot identifiable, inventaire explicite, comparaison valide) ne sont
ni des niveaux de sévérité ni un verdict de conformité du code.

### Constats et restitution

Lire [closed.md](closed.md) avant de créer ou rouvrir un constat. L'identité
vient de la cause et du périmètre, pas du titre ou de la correction envisagée.
Une signature qui réapparaît reprend son identifiant. Un changement de version
ou une case non mesurée d'un ancien mandat ne rouvre pas automatiquement une
décision close.

[response.md](response.md) contient tous les constats non résolus, y compris
les reports explicites. Chaque entrée conserve :

- identifiant stable, titre précis, état, date et propriétaire ;
- sévérité, priorité et confiance, distinguant fait prouvé et hypothèse ;
- contrat concerné, localisation exacte et preuve reproductible ;
- conséquence concrète, portée et éventuelles explications réfutées ;
- correction minimale proposée, sans implémentation imposée sans preuve ;
- critère de clôture vérifiable et liens aux axes secondaires.

Une ambiguïté qui gêne navigation, compréhension ou extension est un impact
réel, même sans bug numérique. Un goût personnel, une ligne longue, un nombre
de templates ou un registre élevé ne suffisent pas seuls. Un manque de preuve
va d'abord dans la couverture ; il devient constat seulement si un risque,
une obligation contractuelle ou une affirmation non étayée sont identifiés.
Ne pas imposer un nombre minimal de constats.

Un constat transversal a un propriétaire principal, pas une copie par section.
Les préfixes historiques sont conservés ; la justesse mathématique peut relever
de `NUM-*`, la qualité documentaire de `DOC-*`, sans renumérotation générale.
Les constats corrigés, réfutés, fusionnés ou devenus inapplicables rejoignent
`closed.md` avec signature initiale, nature de clôture, preuves, limites et
condition de réouverture. Lors d'une remédiation autorisée, les règles durables vont au contrat propriétaire.

Après chaque passage, même sans constat, mettre à jour `status.md` :
snapshot, mandat, couverture/verdict, exclusions, commandes et preuves,
mouvements d'identifiants et limites. Ne pas réécrire les passages historiques.
La restitution distingue défauts, risques à mesurer et choix à conserver.
Une hypothèse peut être rejetée ; sa clôture n'exige pas une modification.

## Exploration, structure et documentation

**Question :** un lecteur sans mémoire du projet trouve-t-il la bonne
responsabilité et comprend-il ce qu'il ouvre ?

### Arborescence et propriétaires

La qualité attendue est la même dans `src`, `tools`, `tests` et leurs
sous-dossiers. Une zone « interne », « personnelle » ou rarement visitée ne
bénéficie d'aucune exemption de rangement, nommage ou lisibilité.

| Domaine | Responsabilité |
|---|---|
| `src/common` | Primitives runtime, CUDA, numériques et moteurs réellement réutilisables |
| `src/model/<asset_class>/<family?>/<model>` | Paramètres, loaders, dynamics, analytics et sampling du modèle |
| `src/model/**/product/[<curve>/]` | Bindings minces `<product>.cuh/.cu`, sans infrastructure modèle |
| `src/curve/<curve>` | Paramètres et analytics de courbe |
| `src/product/<product>` | Paramètres, calendriers, observations et sémantique produit |
| `src/generative` | Code génératif indépendant d'une méthode financière |
| `tools/<responsibility>` | Codegen, orchestration, génération et publication offline |
| `catalog` et `datasets` | Recettes/métadonnées et artefacts, selon la taxonomie de `src` |
| `tests/<domain-or-contract>` | Tests du domaine ou contrat, fixtures et supports identifiables |
| `cmake` | Graphe de build et enregistrement des capacités |
| `build/` | Unique build CMake principal persistant, avec exécutables et intermédiaires recréables |
| `artifacts/` | Preuves, sorties historiques et caches d'outils séparés du build actif |
| `docs` | Entrées, contrats, procédures et références, avec autorité explicite |

Vérifier chaque chemin, rôle et consommateur, notamment :

- domaines stables plutôt qu'accumulation historique ; profondeur utile, pas
  dossier `core/utils/common` sans frontière ni prolifération de micro-dossiers ;
- racine modèle sans produits dispersés, dossier `product/` sans dynamics,
  préparation modèle ou workspace générique ;
- `common` organisé par invariant/domaine, sans modèle ou produit concret
  caché dans une primitive prétendument neutre ;
- produits à ossature prévisible, avec variantes justifiées par leur contrat ;
- séparation templates, renderer, runners, diagnostics, profils et artefacts ;
- pas de collection permanente de `build-*` concurrents ni de preuves rangées
  parmi les objets du build principal ; les builds isolés requis par un audit
  ou un autre profil restent explicitement nommés et distincts ;
- tests groupés par propriétaire ou contrat : ni racine fourre-tout, ni
  miroir mécanique de `src` qui disperserait les tests transversaux ;
- mêmes responsabilités disposées de façon analogue entre modèles ; aucune
  symétrie artificielle pour des capacités mathématiques différentes.

`src` décide de l'identité et du préfixe des entités ; `catalog` et
`datasets` en héritent sans seconde taxonomie. Vérifier la bijection :

```text
src/model/<asset_class>/<family?>/<model>
catalog/model/<asset_class>/<family?>/<model>/<kind>/...
datasets/model/<asset_class>/<family?>/<model>/<kind>/...

src/curve/<curve>       -> catalog/curve/<curve>       -> datasets/curve/<curve>
src/product/<product>   -> catalog/product/<product>   -> datasets/product/<product>
```

`equity/markovian`, `equity/rough`, les identités de courbe/produit et les
variantes pertinentes ne disparaissent pas dans les chemins dérivés. Les
suffixes de recette, dataset et side commencent après le préfixe d'entité.
Contrôler chemins déclarés, chemins générés et artefacts locaux disponibles.
Ne pas exiger que `tests` ou `tools` héritent de cette taxonomie : ils sont
organisés par leurs propres responsabilités.

### Noms et lisibilité du code

Le **chemin complet**, le nom et le contexte doivent désambiguïser le rôle,
sans répéter tous les qualificatifs dans chaque symbole. Vérifier :

- `snake_case` pour le code, fichiers et variables ; `PascalCase` pour types
  et concepts ; constantes selon une convention explicite ; documents nommés
  en `kebab-case`, avec `README.md` comme entrée de dossier ;
- noms canoniques cohérents entre homologues : `parameters`, `dataset`,
  `dynamics`, `analytics`, `schedule`, `pricing_policy`, `sample`,
  `workspace`, `concepts`, lorsque ces rôles existent ;
- qualificatif de méthode/phase lorsqu'il lève une vraie ambiguïté, notamment
  FFT, N-facteurs, préparation, simulation, régression et publication ;
- noms publics, arguments et variables révélant unités et cardinalités :
  jours/années/pas, prix/trajectoires, indice/offset, octets/éléments ;
- cohérence des verbes, côtés, signes et ordre des arguments entre API,
  implémentations, tests, templates et recettes ;
- booléens lisibles comme prédicats et modes explicités par des types ;
- absence de noms permanents fondés sur l'ancienneté (`new`, `additional`,
  `final`, numérotation sans sens) ou d'un helper vague sans propriétaire ;
- variables compréhensibles, portée courte, conversions explicites et flux
  de contrôle lisible ; noms longs et couches supplémentaires ne sont pas
  une amélioration automatique.

Les notations mathématiques canoniques (`A`, `B`, `log_A`, par exemple)
sont admises lorsqu'elles sont définies sans ambiguïté. Le namespace et le
dossier doivent être compatibles, pas nécessairement isomorphes : un namespace
public plat volontaire n'est pas un défaut sans collision ou confusion prouvée.
Un alias public stable peut être utile ; une chaîne d'alias masquant le type
effectif doit être examinée.

### Fichiers et commentaires

Chaque fichier handwritten d'implémentation ou d'automatisation sous `src`,
`tools`, `tests` et `cmake` porte en tête une description courte et spécifique
de son contenu et de son utilité. La règle s'applique aussi aux helpers et
probes de test maintenus, pas seulement aux headers publics.

- Employer un commentaire C++/CMake ou une docstring Python, avant le code,
  en respectant les préambules imposés par le format.
- Distinguer le contrat public de `dynamics.cuh` des définitions device de
  `dynamics_impl.cuh`, ainsi que préparation host et exécution device.
- Ne pas remplacer une explication par la répétition du nom ou « utilities ».
- Classer les exceptions : données déclaratives, recettes minimales et
  outputs générés, notamment les bindings sous `product/`. Ces derniers
  identifient leur origine ; une bannière générée ne masque pas du code manuel.
- Commenter les conventions, invariants, domaines, choix numériques et
  contraintes non évidents ; ne pas paraphraser chaque instruction.
- Signaler hypothèse, unité, mesure ou ordre d'événements au point où leur
  oubli peut changer le calcul. Vérifier ces commentaires contre le code.

Un fichier porte une responsabilité explicable. Séparer les parties qui
évoluent indépendamment, sans fabriquer des micro-fichiers de transit.
Headers publics, définitions incluses `*_impl.cuh` et unités `.cu/.cpp`
autonomes ont des rôles distincts ; le header public expose seulement la surface
nécessaire aux appelants. Pas d'inclusion textuelle d'un `.cu`.

### Documentation et parcours de découverte

Toute documentation maintenue, y compris les README, s'adresse à un lecteur
néophyte qui découvre le projet. Elle utilise des phrases courtes, claires et
descriptives. Elle explique les termes techniques et les acronymes à leur
première utilisation. Elle annonce les prérequis nécessaires.

Elle aide le lecteur à comprendre le rôle des composants et leurs relations.
Elle indique les chemins à suivre pour explorer le projet. Les exemples
simples précèdent les variantes. La concision ne doit pas supprimer les
explications nécessaires ni supposer une connaissance implicite du projet.

Le [README racine](../../README.md) expose but, capacités, limites, prérequis,
un premier résultat observable et les principaux parcours. L'[index docs](../README.md)
oriente par besoin. Les README locaux précisent leur périmètre sans recopier
les contrats ou les listes générées.
Le [guide CMake](../cmake-build-workflow.md) explique configuration, cible,
compilation, exécutable et incrémentalité à partir d'un exemple simple ; le
[plan des artefacts](../local-artifacts.md) distingue `build/` des preuves
historiques et des campagnes.

Vérifier exhaustivement les documents maintenus hors registres :

- nom, sujet, public et autorité identifiables ; `*-contract`, `*-workflow`,
  `*-protocol`, `*-reference` ou `*-index` selon le rôle ;
- rédaction accessible à un néophyte selon les règles ci-dessus ; vocabulaire
  cohérent et aucun historique nécessaire à la compréhension ;
- propriétaire unique des règles ; liens utiles plutôt que contrats copiés,
  inventaires manuels concurrents ou fragmentation en micro-pages ;
- absence de pages orphelines, liens/ancres cassés, chemins ignorés présentés
  comme distribués, commandes obsolètes ou promesses non implémentées ;
- concordance des exemples avec les symboles, options, schémas et capacités
  réels ; distinction claire entre build, test, génération et publication.

Effectuer sans recherche globale initiale les parcours suivants, puis employer
la recherche pour diagnostiquer les difficultés :

1. Depuis la racine, construire et exécuter un premier contrôle sûr.
2. Pour un modèle exact, à schéma, FFT, N-facteurs et fixed income, retrouver
   dynamics, analytics applicables, sample et produits.
3. Depuis un produit, atteindre binding, moteur, recette, paramètres, test,
   fixture et commande correspondante.
4. Depuis un test ou un outil, retrouver propriétaire, entrées, sorties,
   effets de bord, helpers et invocation.
5. Retrouver les templates pricing/sample par méthode et le chemin d'extension.
6. Retrouver génération/reprise, frontière de certification et adaptation à
   un autre GPU sans interpréter un profil local comme universel.
7. Depuis le preset CMake, retrouver `build/`, une cible de générateur, ses
   sources et dépendances, puis distinguer la compilation de son exécution et
   les preuves historiques conservées sous `artifacts/`.

Consigner détours, fichiers trompeurs, pages inutiles et informations devinées ;
un temps de navigation n'est pas à lui seul un verdict. Un checker vert ne
prouve ni lisibilité ni absence d'angle mort.

**Preuve de couverture :** inventaire des racines, fichiers et documents,
exceptions classées, symboles publics et targets revus, recherches systématiques
sur les noms privés, références/consommateurs contrôlés et parcours exécutés.
Les problèmes découverts alimentent le verdict, pas une réduction artificielle
de la couverture.

## Justesse des modèles et des produits

**Question :** le calcul exécuté correspond-il au modèle, au produit et aux
conventions annoncés, avant même de discuter précision ou vitesse ?

Les documents et le manifeste déclarent un contrat ; ils ne prouvent pas sa
validité mathématique. Confronter les équations, l'implémentation et des
identités/références distinctes. Une documentation qui décrit fidèlement une
erreur de code ne la rend pas correcte.

### Traçabilité de bout en bout

Pour chaque implémentation distincte et ses compositions pertinentes, suivre :

```text
paramètres et unités -> dynamique et mesure -> calendrier
                    -> observations -> payoff/exercice
                    -> actualisation/numéraire -> résultat et métadonnées
```

Localiser les fonctions réellement appelées, pas seulement les déclarations.
Tracer aussi le sampling : paramètres/calendrier -> loi simulée -> observables
et layout publiés. Identifier toute transformation entre entrée, état préparé
et valeur calculée, ainsi que ses hypothèses.

### Dynamique et mesure

Vérifier notamment :

- équations, drifts, diffusions, compensateurs de sauts, corrélations et lois
  de tirage ; conventions de volatilité, variance, intensité et temps ;
- nature exacte ou approchée de chaque transition et domaine où elle est valide ;
  un endpoint exact ne rend pas exacte l'observation continue du chemin ;
- schémas, traitement des frontières, corrélations et stabilité des états ;
  aucun clamp, rejet de ligne ou projection ne change silencieusement la loi ;
- état markovien suffisant ; mémoire Volterra, noyau/covariance et lift
  N-facteurs conformes au processus visé, approximation annoncée ;
- transitions gaussiennes conjointes facteurs/intégrale et leurs covariances,
  lorsqu'elles sont utilisées pour l'actualisation ;
- mesure de simulation et numéraire explicites ; drifts, paramètres de
  transition, payoff normalisé et conversion initiale cohérents ;
- actualisation appliquée exactement une fois, au bon horizon et avec la
  bonne variable ; ne pas confondre facteur réalisé et obligation conditionnelle ;
- réemploi des analytics sous un changement de mesure seulement si leur
  interprétation conditionnelle et leurs arguments restent valides.

Un changement de numéraire n'est pas une norme à imposer partout. Comparer
avec les capacités existantes : une transition jointe exacte ou une formule
fermée peut rendre inutile une nouvelle infrastructure. La méthode et son
approximation doivent être visibles dans la composition, sans contaminer le
sampling risque-neutre.

### Produit, calendrier et exercice

Contrôler paramètres, domaines et sémantique :

- call/put, payer/receiver, notionnel, strike, coupon, rebate, barrières et
  convention de paiement ; unités et signes cohérents ;
- dates d'observation, fixing, exercice, règlement et cashflows distinctes ;
  inclusion des endpoints, stubs, accruals et ordre d'événements simultanés ;
- monitoring annoncé conforme au code : discret, continu approché ou corrigé,
  et traitement pertinent des sauts ; pas d'hypothèse brownienne implicite ;
- état du payoff suffisant pour moyenne, extrema, mémoire coupon, autocall
  et extinction ; état initial, arrêt et paiement final corrects ;
- calendrier contractuel sans sous-pas artificiels pour une transition exacte ;
  schémas à pas fixe dérivés de `time_grid.steps_per_year`, pas recalculés
  implicitement à partir de chaque maturité ;
- American sur grille décrit comme approximation bermudéenne ; backward,
  candidats, cible de régression et décision cohérents avec la mesure ;
- variables de continuation disponibles à la date d'exercice, sans information
  future ; distinction entre erreur de régression et biais d'estimation LSM.

Les formules fermées et leurs décompositions satisfont leurs hypothèses
(modèle, signe, monotonie, calendrier, cashflows). Ne pas remplacer une méthode
de pricing par une autre, ni étendre vers le semi-analytique, sans mandat.

### Composition, samples et résultat

Vérifier que binding, side, schedule, provider, engine et données préparées
correspondent à la capacité déclarée. Un concept compilable ne prouve pas cette
sémantique. Examiner les différences de composition même si le moteur est partagé.

Pour les samples, distinguer un chemin par paramètre et plusieurs chemins
conditionnels par paramètre ; distribution des paramètres, répétitions,
maturités et observables doivent correspondre au layout annoncé. Pricing et
sampling peuvent partager une dynamique sans partager mesure ou sortie.

Contrôler aligned/cartesian, associations modèle-produit-courbe, cardinalités,
ordre des lignes et métadonnées. Un JSON valide contenant le prix d'une autre
ligne ou un mauvais observable est incorrect.

**Preuve de couverture :** chaîne de calcul et hypothèses localisées pour
chaque implémentation distincte, branches de composition contrôlées, identités,
cas déterministes, moments ou références locales adaptés. Distinguer erreur
de modèle, de payoff, de mesure, de calendrier et erreur arithmétique. Le
détail des budgets numériques et de l'indépendance des tests relève des axes
suivants ; une certification externe des datasets n'est pas requise ici.

## Homogénéité et factorisation

**Question :** partage-t-on le bon invariant, au niveau le plus simple qui
rend les consommateurs lisibles ?

L'homogénéité concerne les rôles et contrats homologues. La factorisation
concerne l'implémentation partagée. Une API semblable ne suffit pas à justifier
un moteur commun ; une différence mathématique ne justifie pas une convention
de nommage arbitrairement différente.

### Contrats minimaux et frontières

S'appuyer sur les contrats [dynamics](../cuda/model-dynamics-contract.md),
[analytics](../cuda/model-analytics-contract.md) et
[composition](../cuda/pricing-policy-composition.md), en vérifiant leur
concordance avec l'implémentation :

- séparer paramètres bruts, types préparés, état mutable et observables ;
  ne pas imposer un type vide à une famille qui n'en a pas besoin ;
- mêmes noms, unités, types de retour, qualificateurs CUDA, ordre des arguments
  et conventions de propriété pour une responsabilité commune ;
- capacités optionnelles explicites et concepts minimaux, chaque exigence
  correspondant à un appel ou invariant réel ;
- concepts positifs/négatifs et relations de types vérifiés au propriétaire,
  avec diagnostic lisible plutôt qu'erreur profonde d'instanciation ;
- analytics modèle sans produit concret ; schedule sans dynamique/payoff ;
  handler sans moteur ; régresseur sans modèle, courbe ou produit ;
- `ContinuationState` limité aux données nécessaires à la continuation ;
  types device compacts, trivialement copiables et budgets motivés ;
- aucun accès d'un moteur aux champs privés de préparation d'une policy ;
  l'interface promise doit suffire à une autre implémentation conforme ;
- composition modèle-produit mince, sans seconde formule ou orchestrateur ;
  spécialisations de side publiques et explicites conformément au contrat.

Confronter le graphe réel à la direction :

```text
common <- model / curve / product <- tools <- catalog
                                 <- tests
```

Les tests des outils peuvent dépendre de `tools`. Le runtime ne dépend jamais
de `tools`, `catalog` ou `validation`. Une formule runtime réutilisée offline
garde son propriétaire dans `src`. Ne pas déplacer une responsabilité produit
dans `common` uniquement pour éviter un include.

### Pyramide des invariants partagés

Conserver les niveaux distincts, sans exiger une classe ou un dossier pour
chaque flèche :

```text
transition exacte --+
schéma à pas fixe ---+--> markovien général --------+
                                                   +--> commun rough-markovien
rough FFT -----------+                              |
rough N-facteurs ----+--> rough général ------------+

Black-Scholes closed form --+
fixed income closed form ---+--> exécution closed form commune

equity + fixed income --> infrastructure cross-asset strictement neutre
```

Examiner les propriétaires et consommateurs à chaque niveau :

- **Exact :** préparation/transition sur dates utiles, lois spécifiques
  préservées, aucune sous-grille artificielle.
- **Schéma :** préparation de `dt`, boucle, observation et handlers communs ;
  Euler/QE/autre algorithme reste dans la dynamique.
- **Markovien général :** état initial, composition schedule/dynamics/produit,
  Monte Carlo, samples compatibles, indexation, RNG et réduction ; pas de
  branche par modèle dans la boucle chaude ni de données inutiles communes.
- **FFT :** noyau, poids, covariance, padding, convolution et reconstruction
  partagés ; état et transformation propres au modèle restent au modèle.
  `KernelPolicy` et `PathPolicy` ont des contrats séparés, leurs paramètres
  concordent, le moteur possède buffers/plans/dispatch et traite le noyau
  préparé comme opaque. Pricing et samples ne dupliquent pas ce moteur.
- **N-facteurs :** noyau exponentiel et préparation identifiables, nombre de
  facteurs explicite, exécution markovienne réemployée lorsqu'elle convient ;
  erreur de lift et convergence ne disparaissent pas derrière le partage.
- **Rough général :** seulement les invariants communs effectivement prouvés ;
  ni fausse dynamique markovienne imposée au FFT, ni dépendance FFT du lift.
- **Rough-markovien :** produits/schedules compatibles, RNG, résultats,
  réductions, diagnostics et exécution partageables ; pas de variant device
  géant, branche de famille ou pointeur de fonction dans le chemin chaud.
- **Closed form :** distinguer primitives mathématiques et exécution.
  Black-Scholes et fixed income partagent ce qui conserve domaines, unités et
  conventions ; CIR, gaussien, courbes et Jamshidian gardent leurs spécificités.
  Scalaire et coopératif peuvent partager une formule sans partager un kernel.
- **Cross-asset :** ressources CUDA, indexation, RNG, moments, workspaces et
  moteurs neutres ; spot, courbe, accrual, swap et signes produit ne remontent
  pas dans une infrastructure qui n'en a pas besoin.

### Duplication et sur-factorisation

Pour chaque abstraction commune ou nouvelle proposition, expliciter :
invariant, consommateurs réels, partie spécifique, duplication supprimée,
indirections/adaptateurs ajoutés et conséquence d'une extension ordinaire.

Rechercher autant les couches superflues que les copies :

- template dont un paramètre ne varie jamais ou ne porte aucun invariant ;
- concept redondant, façade de transit, chaîne d'alias opaque, traits globaux
  et combinaisons de flags dont les contraintes sont introuvables ;
- généricité pour un consommateur hypothétique ; exceptions en cascade dans
  le moteur partagé ; nombre d'instanciations sans usage réel ;
- duplication sémantique cachée par des noms différents, particulièrement
  entre pricing/samples, equity/taux et implémentations handwritten/générées ;
- préparation, dispatch ou sérialisation réimplémentés par chaque recette.

Deux consommateurs réels constituent un argument de réutilisation, pas un
critère universel pour toute abstraction. Une frontière d'encapsulation, un
type d'unité ou un propriétaire RAII peut être utile à un seul consommateur.
Sa raison doit être explicite ; cela ne justifie pas un framework extensible.

Évaluer le coût cognitif : pour comprendre ou modifier un calcul, combien de
couches et paramètres faut-il suivre, et lesquels portent une décision utile ?
Aucun quota arbitraire ne remplace cette démonstration. La bonne proposition
peut être de conserver une petite duplication ou de supprimer une couche.

Pour un chemin sensible, comparer instanciations, build propre/incrémental,
taille machine, types préparés, valeurs vivantes, registres, spills, mémoire et
temps. Réutiliser des preuves compatibles ou démontrer l'absence d'effet ;
mesurer avant de revendiquer un gain. Une branche dépendant des données peut
être nécessaire : viser les dispatchs architecturaux inutiles, pas interdire
les décisions mathématiques du payoff ou du schéma.

**Preuve de couverture :** contrats et consommateurs des moteurs communs
recensés, recherche des duplications dans chaque implémentation distincte,
exceptions examinées et décision motivée par abstraction sensible. Une preuve
de coût absente limite la conclusion sur le coût, pas la possibilité de
diagnostiquer une frontière manifestement incorrecte.

## Qualité et organisation des tests

**Question :** quel défaut chaque test détecterait-il et pourquoi son oracle
ne reproduirait-il pas ce même défaut ?

### Organisation et lisibilité

Les règles de la première section s'appliquent intégralement. Vérifier :

- tests regroupés par domaine ou contrat, fichiers nommant le comportement
  ou propriétaire et nature CPU/CUDA lorsqu'utile ;
- noms de tests, exécutables, CTest et diagnostics cohérents et retrouvables ;
- fixtures, références, helpers, benchmarks et rapports distincts ;
- en-tête précisant contrat testé et méthode de vérification ; données de
  fixture avec unités, origine et raison du cas ;
- helpers partagés seulement lorsqu'ils clarifient plusieurs tests réels ;
  aucun `common.py` ou header fourre-tout sans frontière ;
- prototypes et sorties de campagne hors sources maintenues ; les preuves
  historiques utiles restent compactes et accessibles ;
- absence de seconde infrastructure de production cachée dans les tests.

Une implémentation CPU indépendante, simple et bornée, est un oracle légitime.
Elle n'a pas à réutiliser les helpers numériques qu'elle est censée vérifier.
À l'inverse, ne pas entretenir un second pricer de production sans nécessité.

### Force de la preuve

Cartographier contrat -> test -> oracle -> assertions -> commande effective :

- vérifier que le test appelle bien le chemin de production, avec la bonne
  spécialisation et les données annoncées ;
- vérifier les assertions dans le build réellement utilisé, notamment
  `NDEBUG`, macros de tests, codes retour et remontée des erreurs CUDA ;
- distinguer test de compilation, smoke, replay, parité entre géométries,
  référence indépendante et certification financière ;
- ne pas qualifier la justesse par la seule finitude, un replay bitwise ou
  une référence recalculée par la fonction candidate ;
- identifier les dépendances partagées entre candidat et référence : passer
  en FP64 ou sur CPU n'élimine pas une erreur de formule commune ;
- choisir cas déterministes, identités, moments, limites et références locales
  qui rendent visibles les erreurs de signe, de mesure, de calendrier et
  de branche, pas seulement le cas nominal ;
- vérifier que le test peut échouer pour la cause visée. En cas de doute,
  utiliser une fixture négative ou une perturbation locale isolée ; aucune
  campagne de mutation exhaustive n'est imposée ;
- contrôler skips, tests non enregistrés, exceptions avalées, sorties non
  examinées et assertions tautologiques ;
- conserver budget d'erreur et protocole statistique fixés indépendamment du
  résultat ; documenter variance, graines et risque de faux échec, sans
  relancer jusqu'au succès.

### Couverture utile et maintenance

Couvrir les frontières propres à chaque contrat : vide/unitaire/dernier lot,
domaines valides et invalides, core/stress, zéro ou très petite maturité,
paramètres limites, schedules irréguliers, côtés, layouts, offset, réutilisation
et mode d'échec. La liste applicable vient du contrat, pas d'un produit cartésien
automatique de tous les paramètres.

Pour les concepts, avoir des cas positifs et négatifs au bon niveau. Pour
les moteurs partagés, distinguer tests du moteur, de la policy et de leur
composition. Pour codegen/planners/writers/checkers, tester aussi les erreurs
de mapping, les sorties manquantes et les reprises, pas seulement le succès.

Un test de régression historique conserve le cas minimal et la raison du
défaut ; il ne fige pas inutilement le layout privé du moteur. Les tests doivent
être reproductibles, isolés des datasets publiés, sans téléchargement,
publication ou import de référence externe implicite.

Les dépendances GPU, mathDx et longues durées sont déclarées ; les suites
ordinaires et la validation indépendante ont des sélections explicites et
disjointes. Un test lent n'est pas supprimé pour rendre la suite verte :
il reçoit un périmètre d'exécution adapté.

**Preuve de couverture :** inventaire source/CTest confronté, contrats critiques
associés à leurs oracles, assertions et diagnostics revus, tests exécutés
identifiés par binaire et configuration. Le nombre de tests passés ne remplace
aucun de ces liens.

## Codegen et génération des datasets

**Question :** une extension ordinaire demande-t-elle seulement le contenu
mathématique/produit nouveau et sa déclaration, ou impose-t-elle des copies ?

Suivre le [guide codegen](../../tools/codegen/pricing_bindings/README.md) et le
[workflow d'extension](../catalog-extension-and-validation-workflow.md), puis
vérifier la chaîne réelle :

```text
capacité déclarée -> resolver -> template -> renderer -> binding
                 -> target -> recette -> artefact/métadonnées -> contrôle
```

### Source de vérité et templates

- Inventaire typé des modèles, produits, courbes, méthodes, sides, samples
  et dépendances ; état explicite disponible, différé, non supporté ou absent.
  L'existence d'un fichier ne définit pas une capacité.
- Ownership explicite de chaque output : généré, handwritten ou exception
  justifiée. Chaque exception a un propriétaire et un test.
- Même source pour resolver, CMake, recettes et inventaires dérivés ; aucune
  table parallèle reconstruisant noms, chemins, capacités ou dépendances.
- Préfixes `src/catalog/datasets` conservés ; variantes de courbe, méthode
  et produit explicites, aucun alias historique caché dans un renderer.
- Templates trouvables par méthode et rôle : markovien, rough N-facteurs,
  Volterra FFT, closed form equity/fixed income, exercice anticipé, sampling.
  Binding `.cuh/.cu` et générateur de catalogue sont distingués.
- Formules et fonctions C++ complètes visibles dans leurs templates, pas dans
  des chaînes opaques du renderer. Fragments partagés limités à un rôle réel,
  sans puzzle de micro-templates pour une fonction simple.
- Le générateur de catalogue actuel est `generator.cpp` ; un changement de
  langage est une décision, pas une exigence implicite de l'audit.
- Profils numériques, time grid, graines, géométries et limites de batching
  ont un propriétaire explicite ; pas de choix financier masqué par défaut.

### Coût réel d'une extension

Pour un modèle, identifier paramètres, dynamique/analytics, préparation,
observables, domaines et tests qui doivent être écrits à la main. Pour un
produit, identifier paramètres, calendrier, observations, payoff, continuation
si nécessaire, domaines et tests. Le reste doit être dérivable lorsqu'il ne
porte pas de nouvelle décision sémantique.

Un couple compatible ne doit pas recopier moteur, launcher, ressources CUDA,
publication, recette entière ou liste CMake. Les formules closed form propres
et les exceptions nécessaires restent visibles ; le codegen ne les devine pas.

Sur un ajout modèle, produit et composition représentatif, faire un parcours
d'extension isolé ou un dry-run suffisamment concret : fichiers et décisions
manuels, entrées de manifeste, outputs, commandes et retrait sans orphelins.
Mesurer la localité de l'ajout, pas seulement le volume de code généré.
Ne pas inventer une extension permanente pour satisfaire cet exercice.

### Pricing, samples et recettes

Vérifier séparément les deux chaînes. Pour le sample : capacité modèle ->
engine -> `sample.cuh/.cu` -> recettes -> observables et artefacts. Les deux
layouts contractuels de
[model-sample-dataset-generation.md](../model-sample-dataset-generation.md)
doivent exister lorsqu'annoncés, avec leurs propres dimensions et graines ;
un nombre de trajectoires de pricing n'est pas une shape de samples.

Pour tous les datasets :

- domaines, contraintes et transformations conformes au
  [contrat des paramètres](../model-and-product-parameter-dataset-generation.md),
  avec politique ordonnée 90/10 et provenance core/stress ;
- préparation, génération de paramètres/calendriers, simulation, assemblage
  et JSON/YAML séparés et partagés aux bons niveaux ;
- schema, unités, ordre des champs/lignes, dimensions, source rows et domaines
  Philox cohérents entre déclaration, loader et publication ;
- métadonnées structurelles générées depuis leur propriétaire ; observations
  de timing issues du run réel, pas d'une recette supposée ;
- aucune certification indépendante inventée : prix non certifiés explicitement
  `pending / verified: false`, selon le workflow propriétaire ;
- batching, contrôles mémoire, erreurs, staging et reprise sans écrasement
  prématuré, recalcul inutile ou artefact partiellement présenté comme complet.

Suivre le [workflow de génération](../dataset-generation-workflow.md) sans
publier pour les besoins de l'audit. Vérifier par tests isolés les interruptions,
reprises, intégrité et écritures atomiques à leur granularité réelle.

### Régénération et contrôles

Le zéro-diff est nécessaire, pas suffisant. Comparer l'ensemble attendu et
l'ensemble physique dans les deux sens ; tester output absent, extra, orphelin,
rename, mapping erroné et exception retirée. Vérifier les bindings pricing et
samples, recettes, fragments CMake et champs YAML effectivement dérivables.

Exécuter la génération hors du tree, les tests de resolver/templates/checkers
et les builds/smokes pertinents avec/sans dépendances optionnelles. Les outputs
générés ne peuvent porter de corrections manuelles invisibles à leur source.

**Preuve de couverture :** capacité-vers-output résolue pour tout le manifeste,
exceptions relues, génération bidirectionnelle, erreurs du checker vérifiées et
trois parcours d'extension documentés. Une compilation seule ne prouve pas le
choix sémantique du binding.

## Robustesse numérique et reproductibilité

**Question :** les erreurs, approximations et arrondis restent-ils maîtrisés
sur le domaine annoncé, avec une reproductibilité clairement définie ?

Pour chaque famille de primitives, relier domaine core/stress/limites,
stockage/calcul/accumulation, référence, budget, mode d'échec et tests. Séparer
erreur de formule, discrétisation, approximation rough, bruit Monte Carlo,
régression/exercice et arithmétique.

### Précision, FP64 et domaines

Inventorier tout calcul FP64 atteignable depuis du code device, y compris
préparation, kernels auxiliaires, réductions et fonctions
`__host__ __device__`. Pour chacun : fréquence d'appel, volume d'opérations,
justification numérique, alternative FP32/mixte/host, ressources compilées
et coût mesuré. Une justification mathématique sans comparaison numérique et
performance ne suffit pas.

- Examiner les instanciations et promotions effectives, pas seulement une
  recherche textuelle de `double`. Distinguer préparation par paramètre,
  calcul par prix, par trajectoire, par date et par itération.
- FP32 n'est pas correct par définition ; FP64 n'est pas gratuit par
  définition. Les accumulations/algèbres linéaires sensibles existantes restent
  préservées tant qu'une alternative n'est pas qualifiée.
- Inclure les multiplications avant accumulation, coefficients, décisions,
  fonctions spéciales et chemins auxiliaires, pas seulement le stockage.
- Une préparation device n'est pas une exemption : quantifier sa fréquence.
  Mutualiser les preuves d'une même primitive dans le même domaine, sans
  extrapoler à un consommateur de fréquence ou d'échelle différente.
- Confronter alternatives au budget de sortie et au coût du pipeline ; un
  microbenchmark plus rapide ne justifie pas une perte de précision ni un
  transfert host plus coûteux.

Contrôler domaines de log, racines, divisions, puissances, exponentielles,
CDF/inverses ; cancellation, overflow/underflow, sous-normaux, petits temps,
faible volatilité/mean reversion, corrélations limites et intensités extrêmes.
Préférer les expressions stables justifiées (`log1p`, `expm1`, limites,
compensation), sans fast math ni changement de loi silencieux.

### Solveurs, approximations et budgets

- Racines/inversions : bracket, monotonie applicable, résidu, stagnation,
  itérations bornées et certification avant publication.
- Régressions/factorisations : conditionnement, normalisation, ridge/pivot,
  erreur des statistiques et du solveur, propagation à la décision.
- LSM : distinguer aucun candidat, sous-détermination et échec numérique ;
  traiter systèmes singuliers/quasi singuliers et coefficients non finis.
  Un faible résidu backward ne prouve pas la stabilité du prix.
- Convergence : moments et limites exactes, raffinement du schéma, résolution
  FFT, nombre de facteurs et grille d'exercice selon l'approximation employée.
- Budgets absolus/relatifs/ULP/statistiques adaptés aux échelles et définis
  avant le résultat candidat. Une calibration empirique utilise un jeu
  distinct ; aucune tolérance élargie pour faire passer le cas observé.
- Cas core et stress conservés, pas de filtrage après échec, changement
  opportuniste de référence, fallback ou clamp non contractuel.

Les références locales peuvent être une identité, une limite analytique ou un
calcul indépendant de précision supérieure. Leur degré d'indépendance est
documenté dans les tests ; il ne s'agit pas de la certification externe des
datasets.

### RNG et propagation des erreurs

Préserver le mapping Philox `(path_index: uint64, local_group_index: uint64)`
sous la clé de ligne, sans réservation aplatie dépendant du batching.
Vérifier domaines disjoints paramètres/calendriers/dynamics/variantes, offsets
globaux, rejets/sauts et frontières des paquets conditionnels.

Préciser pour chaque comparaison : bitwise contractuel, tolérance arithmétique
ou comparaison statistique. Ne pas demander le bitwise entre algorithmes quand
il n'est pas contractuel, ni relâcher une invariance promise parce que les lois
semblent identiques. Examiner ordre des réductions, atomiques, streams et
différences CPU/GPU ou inter-architectures.

Les erreurs de domaine, non-convergence, allocation, lancement et sortie
doivent être attribuables et empêcher une publication invalide. Une valeur
intermédiaire sentinelle documentée (par exemple un log-spot absorbé) se
distingue d'une sortie non finie accidentelle ; finitude seule n'est pas justesse.

**Preuve de couverture :** primitives et calculs FP64 device classés, domaines
et modes d'échec examinés, tests de limites/convergence/RNG associés aux contrats,
références et coûts qualifiés sur le périmètre déclaré. Toute preuve requise
absente reste visible, sans convertir « non mesuré » en « sans coût ».

## Exécution CUDA et sécurité mémoire

**Question :** les entrées, ressources et synchronisations garantissent-elles
une exécution correcte, y compris aux frontières ?

Appliquer le [contrat de lancement et diagnostics](../cuda/launch-validation-and-kernel-diagnostics.md).
Revoir tous les launchers et workspaces actifs :

- additions/multiplications, conversions, strides, offsets, alignements et
  cardinalités contrôlés avant allocation/lancement, y compris overflow ;
- pointeurs, vues ragged, pools, derniers blocs partiels et offsets de batch ;
- durée de vie des buffers, plans, événements et préparations jusqu'à la
  dernière opération asynchrone ; RAII et exceptions host ;
- initialisation des champs, flags, coefficients et réductions sur toutes les
  branches, dont absence de candidat et workspace réutilisé ;
- data races, atomiques, fences, synchronisations et portée inter-blocs/streams ;
- propagation du stream, pas de default stream ou synchronisation globale
  accidentelle ; comportement concurrent conforme à l'API ;
- erreurs asynchrones remontées au bon propriétaire et nettoyage correct ;
- budgets de RAM/VRAM et pic réellement vivant, contexte/pools distingués ;
  une mesure de RAM libre ne représente pas nécessairement la RAM disponible ;
- rejet explicite d'un prix qui ne tient pas, sans découpage qui change
  l'algorithme ; mapping et résultats préservés aux frontières de lots.

ASan/UBSan couvrent les loaders, planners, tailles et orchestrations host
pertinents. Cartographier les quatre modes Compute Sanitizer (`memcheck`,
`racecheck`, `initcheck`, `synccheck`) aux cas CUDA suivants :

1. closed form scalaire et coopératif, lorsque disponibles ;
2. MC exact et à schéma, schedules réguliers/explicites, aligned/cartesian ;
3. LSM equity multi-états et taux deux facteurs, workspace réutilisé ;
4. FFT pricing et sampling, deux layouts de samples ;
5. N-facteurs pricing et sampling, plus grand nombre de facteurs publié ;
6. petite taille, dernier lot partiel, frontière de groupe conditionnel ;
7. stream non-default, répétitions et concurrence effectivement supportée.

Un test peut couvrir plusieurs cases. Chaque case garde commande, binaire,
logs et résultat par mode. Un mode inapplicable est justifié ; un mode non
exécuté est une limite, pas un faux succès ou un défaut automatique du code.

**Preuve de couverture :** revue des launchers/workspaces et matrice host/CUDA
renseignées, y compris frontières et modes d'échec. Un sanitizer vert ne
remplace pas la vérification des contrats de durée de vie et de concurrence.

## CMake, dépendances et portabilité

**Question :** le build principal est-il unique, lisible, minimal et
prévisible, avec un coût de configuration/compilation justifié et sans
dépendance accidentelle au GPU de référence ?

### Graphe et incrémentalité

- `build/` est le seul build principal persistant. Le preset, les scripts, les
  notebooks et les commandes maintenues pointent vers lui ; les sources et
  règles restent dans Git, les exécutables et objets recréables dans le build
  ignoré par Git. Aucun `build-dev` ou `build-*` parallèle ne devient une
  seconde référence implicite. Un autre profil ou un essai d'audit peut créer
  un build isolé et nommé, sans déplacer ni corrompre `build/`.
- `artifacts/` reçoit les preuves à garder hors du build : anciens tests,
  sorties, logs, mesures, snapshots et binaires figés lorsque leur conservation
  est justifiée. Identifier ce qui est historique, ce qui est encore utilisé
  et les liens vers les registres ; ne pas compter un vieux binaire comme
  exécutable actif ni perdre les preuves en nettoyant un cache CMake. Comme
  ce dossier est ignoré par Git, les preuves nécessaires sont inventoriées,
  hashées et exportables selon le mandat.
- Racine CMake centrée sur configuration et orchestration, modules de domaine
  nommés ; règles courtes, lisibles et localisées ; pas de fonctions, options,
  alias, couches ou façades sans consommateur.
- Un propriétaire logique explicite par source et target. Une recompilation
  volontaire d'un même source pour un test A/B ou une variante est justifiée,
  pas confondue avec un doublon accidentel ou interdite par principe.
- Pour une cible représentative, retrouver sans ambiguïté source, bibliothèques
  liées, options, fichier produit et commande d'exécution. Une inclusion de
  header n'est pas prise pour une règle de lien ; un agrégat ne masque pas la
  cible individuelle. Le nom du target et de l'exécutable est découvrable
  depuis le graphe réellement configuré.
- Toutes les unités autonomes et instanciations enregistrées, aucun symbole
  oublié au link, aucune double définition ou ancienne implémentation active.
- `PUBLIC/PRIVATE/INTERFACE`, includes et définitions limités aux usages réels ;
  pas de loaders/publication réexportés par un simple launcher.
- Capacités dérivées du manifeste, pas de liste CMake concurrente ou de regex
  de chemin servant de définition sémantique ; globs contrôlés dans les deux
  sens s'ils participent à l'inventaire.
- Dépendances optionnelles, notamment mathDx/cuFFTDx, réservées aux targets
  qui les utilisent ; paramètres host-only conservés sans mathDx.
- Presets, labels et agrégats cohérents : ce que l'on construit correspond à
  ce que le preset de test exécute, sans lancer la validation indépendante.
  Les builds ciblés ne compilent pas inutilement toutes les recettes.
- Headers/template definitions visibles aux points nécessaires, sans exposition
  ou instanciation massive injustifiée ; erreur de compilation lisible.
- Build propre, no-op et modifications représentatives d'un modèle, produit,
  courbe, primitive, template et manifeste : recompilations attendues/réelles,
  temps et tailles d'artefacts. Mutations en copie isolée. Examiner les cibles
  orphelines, les doublons et l'espace retenu ; la réduction du coût doit être
  mesurée, sans sacrifier la justesse du graphe ou sa lisibilité.
- Cache de compilation évalué selon sa disponibilité ; son absence n'est pas
  un défaut si le build et les dépendances restent corrects.

### Compatibilité et profils matériels

Pour chaque engine et architecture annoncée, distinguer configuration,
compilation, exécution testée, qualification numérique et performance mesurée.
Les preuves offline ne certifient ni le runtime ni le débit.

Vérifier les architectures/minimums annoncés, dépendances, mono-architecture,
fatbin/PTX lorsqu'ils sont supportés et configurations avec/sans options.
Un preset SM89 local peut coexister avec un build portable. Une limite matérielle
ou de bibliothèque produit un diagnostic précis, pas un refus arbitraire de
tout GPU différent du GPU personnel.

Séparer invariant numérique, dimension de workload et paramètre de tuning.
Géométrie, chunks, streams, EPT/FFTs-par-bloc et seuils de stratégie sont
retunables ; facteurs rough et nombre de pas affectent aussi l'approximation,
ils ne sont pas des réglages de vitesse libres à précision supposée constante.

Les valeurs RTX 4090 Laptop/SM89 sont des références étiquetées, pas des optima
universels. Elles sont centralisées et ajustables par leur propriétaire sans
modifier modèle, payoff ou mapping RNG. La documentation invite l'utilisateur
à vérifier ressources, tests et performances sur son GPU avant de publier un
autre profil. Pas d'auto-tuning opaque ou de nouveau dispatch par architecture
sans besoin mesuré.

**Preuve de couverture :** chemin preset -> `build/` -> source -> target ->
dépendances -> exécutable confronté au graphe réel, racine du projet inspectée,
traces historiques `artifacts/` classées et non confondues avec les tests
actuels ; builds propres/no-op et mutations applicables documentés avec leurs
coûts, niveaux de support distincts. Une architecture sans matériel peut être
déclarée compilable avec runtime non examiné ; elle n'est pas artificiellement
rendue incompatible.

## Performance

**Question :** les stratégies, ressources et coûts de génération sont-ils
adaptés aux charges utiles, sans compromis numérique caché ?

L'exigence porte à la fois sur le coût du calcul et sur la qualité de la preuve.
Registres, spills, stratégies de kernels, mémoire et coût des abstractions sont
des critères centraux. Une amélioration supposée, une occupation théorique ou
un unique chrono ne suffit pas.

Conserver quatre sous-audits distincts : **CUDA générique, samples, exercice
anticipé et rough**, avec protocole commun et rattachement des preuves sans
duplication des mesures partagées.

### Périmètre de campagne et preuves réutilisables

Le [protocole de performance](../performance-regression-protocol.md) possède
l'exécution, les budgets et le rebaseline. `tests/performance` possède
benchmarks, fixtures, manifests et baselines ; `tools/performance` possède
runners, checkers et profilage. `validation/**` ne porte aucun gate principal.

Distinguer explicitement :

- **audit** : revue des stratégies, ressources, tests et preuves disponibles ;
- **qualification ciblée** : expérience répondant à une question et un domaine
  préalablement fixés ;
- **retuning** : comparaison de candidats puis confirmation indépendante ;
- **régression officielle** : protocole complet contre une baseline compatible.

Le mandat fixe familles, workloads, tailles et variantes avant les mesures.
Les grilles chiffrées des campagnes et exemples du protocole ne deviennent pas
une obligation de refaire toute leur matrice à chaque audit. Cela ne réduit
aucune exigence statistique ou numérique pour une conclusion effectivement
revendiquée. Une qualification ciblée ne remplace pas la baseline officielle.

Inventorier exhaustivement les familles et stratégies actives ; choisir les
représentants selon leurs coûts et contrats, pas seulement le nom du modèle.
Terminal, monitoring, exact, schéma, FFT, N-facteurs, LSM equity et parcours
fixed income ne sont pas interchangeables. Documenter les familles non couvertes
et les exceptions qui demandent une mesure propre.

Des preuves anciennes peuvent être réutilisées si source pertinente, inputs,
domaine, spécialisation, toolchain et environnement restent compatibles.
Un changement du code compilé ou de son domaine exige de réexaminer cette
compatibilité. Ni l'ancienneté seule ni la proximité des chiffres ne tranche.

### Stratégies et ressources compilées

Pour chaque kernel/spécialisation représentative, examiner :

- unité de travail : thread, warp ou bloc par trajectoire/prix, plusieurs prix
  successifs, grille dense/persistante et logique de saturation ;
- toutes les phases et leur orchestration, y compris préparation, kernels
  auxiliaires, réductions et finalisation, pas seulement le kernel dominant ;
- nombre de threads/blocs, blocs par prix, nombre de prix résidents, chunks
  de trajectoires et batching mémoire : dimensions distinctes ;
- valeurs vivantes, registres/thread, effets de l'inlining, unrolling,
  templates, types préparés et lifetime des temporaires ;
- spill loads/stores, stack frame, allocations locales, shared
  statique/dynamique et accès locaux réellement exécutés ;
- occupation théorique et mesurée, SM/warps/blocs actifs, barrières et stalls ;
- trafic global, coalescence, AoS/SoA, banques shared, caches et divergence
  liée aux dates, rejets, sauts ou sorties anticipées ;
- mélange d'instructions FP32/FP64, indexation 32/64 bits, divisions/modulos,
  fonctions spéciales et coût de génération aléatoire ;
- taille SASS/cubin, instanciations, instruction cache et coût de compilation ;
- allocations, copies, workspace, synchronisations et travail CPU.

Rattacher chaque diagnostic au symbole **effectivement lancé**, au binaire,
à sa géométrie et au workload. Croiser rapports `ptxas`, attributs runtime,
`cuobjdump`/SASS et profilage pertinent. `localSizeBytes == 0` n'est pas à lui
seul une preuve de zéro spill ; la taille de l'exécutable n'est pas la taille
machine du kernel. Un compteur indisponible reste inconnu.

Une hausse de registres ou un spill mérite une investigation, pas une
condamnation automatique ; une occupation plus élevée ne prouve pas un gain.
Établir l'effet sur le chemin réel et vérifier les budgets. Réciproquement,
un bon chrono ne permet pas de masquer une ressource requise non mesurée.

### Mesures, statistiques et décision

Le manifeste de workloads est unique pour les identités, inputs, shapes, side,
calendrier, méthode, nombre de trajectoires/pas/facteurs et métriques requises.
Chaque profil d'environnement porte GPU, architecture, toolchain, flags,
binaires, tuning, ressources et baseline. Un autre GPU ne redéfinit pas
silencieusement l'identité ou le budget numérique du workload.

Avant la campagne, fixer :

- provenance du code, binaires et données ; GPU, toolchain et environnement ;
- baseline compatible et frontières exactes des chronos ;
- warmups exclus, répétitions, regroupement des appels trop courts et ordre
  des candidats ; aucune compilation ou autre charge GPU concurrente ;
- agrégation, seuil de gain utile, budgets numériques/ressources/médiane/p95/CV,
  règle d'exclusion et conditions d'arrêt/reprise.

Distinguer quatre frontières lorsqu'elles s'appliquent :

| Temps | Frontière |
|---|---|
| GPU | Intervalle CUDA déclaré ; un kernel ou une enveloppe multi-kernels, explicitement nommé |
| API publique | Appel et synchronisation, avec horloge host brute conservée |
| Pipeline | Travail appelant complet, préparation externe, allocations/copies et récupération, hors sérialisation |
| Publication | Sérialisation, streaming, écriture et vérification de l'artefact |

L'enveloppe CUDA multi-kernels ne remplace pas le profil par phase. Si le schéma
historique emploie `public_api = max(raw_host_clock, GPU)`, conserver aussi
les deux observations brutes et expliquer leur désaccord, sans prétendre
avoir mesuré une nouvelle horloge. Ne pas additionner des intervalles imbriqués.
Une somme de phases n'est pas un chronométrage de processus complet ; un temps
GPU n'est pas une estimation complète de génération.

Exiger les ressources de **chaque phase active**. Ventiler la mémoire entre
inputs persistants, workspace appelant, transitoires de l'appel, sorties et
marge. Mesurer le pic vivant et distinguer contexte/driver, pools et processus
tiers ; `total-free` après l'appel ne prouve ni le pic ni son propriétaire.

Les valeurs brutes, essais refusés et motifs restent conservés. Interdire
best-of-N, minimum parmi les périodes, recomposition par clé, relances jusqu'à
un CV favorable et choix de tolérance après observation. Le screening sert à
présélectionner, jamais à qualifier seul ; les candidats retenus exigent une
confirmation fraîche sans choisir opportunément sa période.

Les budgets du manifeste sont bloquants et justifiés. Une métrique obligatoire
absente, inconnue, dupliquée ou rattachée au mauvais binaire ne passe pas.
Un échec numérique exclut la comparaison de tuning, même si les timings sont
stables. Un gain inférieur au bruit est indéterminé, pas une optimisation.

Chaque hypothèse reçoit `accept`, `reject`, `inconclusive` ou `unavailable`,
avec portée. `Inconclusive` et `unavailable` n'autorisent aucune qualification
positive. Ne pas confondre ce résultat expérimental avec la couverture de la
revue qui l'a établi.

### Environnement et protections

Observer avant, pendant et après les expériences : GPU, secteur/batterie,
clocks/power, température, régulation et concurrence. Les seuils et budgets
d'admission dépendent du profil ; ceux du laptop SM89 ne sont pas universels.

La politique thermique applicative est `thermal_policy: telemetry_only` :
aucun seuil d'arrêt, attente de refroidissement ou boucle de convergence
thermique ne bloque expériences longues ou génération de prix/samples.
Les variations de limite de puissance dans les pilotes exploratoires sont
journalisées et peuvent rendre les timings inéligibles de façon persistante ;
elles ne tuent ni le calcul ni le job suivant. Conserver les anciennes preuves
avec leurs conditions réelles.

Cela ne supprime ni télémétrie, ni contrôles numériques/mémoire, ni watchdog,
concurrence ou protections natives. L'admission officielle des campagnes suit
son protocole. Ne pas modifier firmware, régulation du GPU ou autorisations
externes. Absence de veto et faible CV ne prouvent pas à eux seuls la
comparabilité des conditions.

### CUDA générique

Auditer closed form et MC ordinaires : préparation, simulation/évaluation,
réduction, launcher et pipeline. Examiner invariants déplaçables hors boucle,
inlining/noinline, fonctions stables, indexation, calendriers hétérogènes,
divergence et coût fixe.

Comparer thread/warp/bloc par résultat ou trajectoire seulement lorsque la
question le justifie. Conserver une stratégie simple si le gain d'une variante
ne couvre pas sa complexité. Ne pas imposer un kernel unique aux formules
scalaires et coopératives lorsque leurs besoins diffèrent.

À nombres de prix/trajectoires fixés, préférer une grille lisible de candidats
admissibles (par exemple 128/256/512 threads), puis élargir si un bénéfice est
plausible. Les contraintes des réductions/FFT priment ; tous les multiples de
warp ne sont pas nécessairement valides. Le nombre de blocs vient du travail,
des ressources et du batching, pas d'une exigence esthétique de puissance de 2.

### Samples

Auditer séparément les deux layouts de production : un chemin par paramètre
et plusieurs chemins conditionnels par paramètre. Mesurer amortissement de
la préparation, génération des paramètres/calendriers, simulation, copies et
streaming, avec débit en samples/s et mémoire host/device.

Comparer les stratégies grid-stride et parameter-block lorsqu'applicables,
saturation, frontières de groupes conditionnels, réutilisation et coûts FFT/
N-facteurs. Pas de plafond lié implicitement au nombre de SM du GPU personnel.
Un smoke-test ne qualifie pas le débit ; une shape réduite ne remplace la shape
production que si la préservation du régime de saturation/mémoire est démontrée.

### Exercice anticipé

Auditer LSM equity simple/multi-états et fixed income un/deux facteurs, y compris
les parcours à actualisation jointe et ceux à numéraire transformé lorsqu'actifs.
Pas d'héritage automatique d'une conclusion MC terminale.

Mesurer simulation forward, calcul des états/payoffs/cashflows, statistiques,
résolution, décision et réduction ; identifier les coûts de préparation des
cashflows et les lancements par date. Examiner SoA, bases, temporaires,
registres, FP64 sensible, shared, atomiques et synchronisations.

Confronter threads, blocs/prix, prix résidents, nombre de dates et découpage
VRAM. Ne jamais partitionner les trajectoires d'un prix en régressions
indépendantes pour faire tenir le workspace. Absence de candidat, réutilisation,
frontières de batches et invariance contractuelle restent contrôlées.
Un tri, une permutation, fusion de kernels ou passage CPU exige une preuve de
coût complet, pas un gain local qui ignore ses copies et synchronisations.

### Rough

Pour FFT, examiner padding/longueurs, EPT, FFTs/bloc, shared opt-in, layouts
complexes, réutilisation des coefficients/plans/spectres, génération, convolution,
reconstruction, payoff, réduction, chunking et streams. Pricing et sampling
partagent les briques sans être supposés avoir le même profil.

Pour N-facteurs, mesurer préparation host, valeurs vivantes, tableaux
register/shared/global, croissance avec les facteurs, exécution markovienne
réemployée et mémoire. Ne pas attribuer au GPU un coût d'ajustement CPU.

Comparer à précision, calendrier et méthode comparables. Un autre nombre de
facteurs ou résolution FFT change aussi l'erreur. Les alternatives directe/FFT
ou les décisions historiques sont réexaminées si leur domaine change ou si
une preuve les remet en cause, pas réimplémentées à chaque audit.

### Mise à l'échelle et projection des générations

Distinguer dimensions de calcul, géométrie et découpage mémoire. Une grille
bornée peut traiter successivement plusieurs prix ; un chunk FFT de trajectoires
n'est pas le nombre total de trajectoires du prix. Les métadonnées doivent
permettre de retrouver les réglages effectivement utilisés.

Le mandat de scaling précise l'axe étudié. À précision de production fixée
(par exemple **2²⁰ = 1 048 576 trajectoires par prix**), mesurer le débit par
prix sur les tailles utiles, sans imposer de refaire les paliers de trajectoires.
Si l'axe trajectoires ou une approximation devient la question, le déclarer
dans une campagne distincte.

Pour extrapoler de 1 000 à un million de prix, comparer d'abord des tailles
intermédiaires avec leur planning de production. Publier débit et rapports
normalisés, distinguer GPU et génération complète, et prolonger le palier si
le régime n'est pas stabilisé. Le facteur ×1 000 est une hypothèse à vérifier,
pas une valeur à faire passer.

Conserver distributions de paramètres, maturités, calendriers et core/stress
comparables. Une tuile répétée peut isoler le débit device, mais ne représente
pas un million de paramètres indépendants : tracer cache hits et coût de
préparation distincte. Contrôler offsets/graines stables, mémoire bornée et
reprise/publication à grand volume. Une mesure à 10 000 prix ne certifie pas un
temps observé à un million ; une projection nomme hypothèses et incertitude.

Les études par famille évitent les campagnes redondantes, mais ne qualifient
pas un produit/moteur dont le coût ou le contrat diffère. Un jeu closed form
n'a aucun axe de trajectoires ; les samples gardent leurs propres shapes.

### Acceptation, rebaseline et couverture

Une optimisation retenue conserve contrat financier, budgets numériques,
mapping RNG et invariances promises. Sans gain utile confirmé ou simplification
justifiée sans régression, conserver le code existant. Interdire
`--use_fast_math` ; ne pas ajouter `__launch_bounds__` sans design approuvé,
adapté aux architectures et mesuré sur chaque cible concernée.

Un rebaseline est explicite, versionné et revu : prédécesseur conservé et hashé,
environnement, bruts, diff exhaustif, motif de chaque delta et approbation.
Tester un candidat contre la baseline qu'il vient lui-même de remplacer ne
prouve pas l'absence de régression. Une première initialisation ne prétend pas
mesurer un delta. Aucun budget n'est relâché silencieusement.

**Preuve de couverture :** inventaire des quatre familles et stratégies actives,
workloads examinés et exclus identifiés, phases/ressources attachées aux binaires,
frontières temporelles et statistiques contrôlées, hypothèses tranchées ou
explicitement indéterminées. La couverture complète de la revue ne transforme
pas des mesures indisponibles en qualification complète des performances.
