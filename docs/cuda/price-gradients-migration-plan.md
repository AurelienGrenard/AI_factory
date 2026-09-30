# Plan de migration vers les prix et gradients sélectionnés

**État :** plan de chantier, non preuve de couverture. Les capacités actives
sont celles du [manifeste typé](../../tools/codegen/pricing_bindings/capability_manifest.py)
et des contrats [equity](equity-price-gradients-contract.md) et
[fixed income](fixed-income-price-gradients-contract.md). Aucun binding,
dataset ou kernel supplémentaire n'est livré par ce plan.

## Cible et ordre de réalisation

La cible est une seule **vue de calcul des sensibilités sélectionnées** pour
les modèles equity et fixed income. Le pilote est le terminal Heston déjà
préparé sur le device : l'hôte transmet modèles, produits et `K`
`SensitivitySpec` résolues ; pour une ligne et une coordonnée, la préparation
compose les nœuds bumpés, le stencil représenté et le besoin du central dans
un `SensitivityTask`. Le moteur évalue les payoffs nécessaires, puis
`reconstruct_sensitivity` applique les coefficients du stencil. Une sélection
vide calcule uniquement le prix. Le [contrat equity](equity-price-gradients-contract.md#européennes-terminales-préparées-sur-le-device)
définit les types et le comportement déjà implémentés.

Il s'agit d'unifier **la représentation et les responsabilités**, pas
d'imposer un kernel unique : Monte Carlo terminal, formule fermée scalaire,
Jamshidian coopératif, Longstaff--Schwartz (LSM) et Volterra FFT ont des
géométries et des workspaces différents. Le central est évalué une seule fois
par prix. Les autres tâches peuvent le rejouer pour leurs innovations ou
leurs nœuds si la méthode l'exige, mais n'écrivent jamais un second prix.

L'ordre des portes de sortie est :

1. consolider cette vue sur les voies gradients déjà actives ;
2. inventorier, migrer et qualifier **tout le markovien equity puis fixed
   income**, y compris les produits de chemin et l'exercice anticipé
   effectivement pris en charge par les prix/prix-delta ;
3. achever dans ce périmètre les sensibilités de saut et de temps admissibles,
   avec les dépendances [Philox](philox-domain-migration-plan.md) et
   [sauts](jump-price-gradient-migration-plan.md) ;
4. seulement après une porte de qualification markovienne, étendre aux
   approximations rough N-facteurs puis aux moteurs Volterra FFT ;
5. migrer les derniers consommateurs et supprimer les artefacts actifs
   `price_delta` devenus inutiles.

Le manifeste, et non une liste recopiée dans ce document, détermine les
couples modèle, courbe éventuelle, produit, moteur et recettes à contrôler à
chaque porte. Une capacité absente ou exclue reste indiquée comme telle ; une
façade compilée ne suffit pas à déclarer sa sensibilité numériquement qualifiée.

Les trois plans n'ont pas le même prérequis :

| Travail | Peut commencer avec le mapping courant ? | Dépendance avant qualification |
|---|---|---|
| Vue préparée et sensibilités markoviennes à tirages fixes | oui | parité prix/gradient et tests de chaque modèle |
| Sensibilités d'intensité et de maturité des modèles à sauts | pilote isolé seulement | couplage des comptes/marques du plan sauts et adresses Philox stables si le flux unique ne suffit pas |
| Rough N-facteurs et FFT | après la porte markovienne | CRN et préparations propres aux nœuds rough ; mapping Philox versionné seulement pour les sources qui doivent changer |
| Retrait de `price_delta` | non | couverture et comparaison de tous ses consommateurs actifs |

Il n'est donc pas nécessaire de convertir préventivement chaque moteur à
plusieurs domaines Philox pour migrer un gradient à tirages fixes. Inversement,
une API de gradients commune ne qualifie pas à elle seule le couplage des
sauts ou le bump de `H`.

## Contrat commun à préserver

- `SensitivitySpec` identifie le paramètre et la règle de bump. L'hôte résout
  une fois les noms en identifiants typés ; un kernel chaud ne compare pas des
  chaînes. Le modèle, la courbe éventuelle et le produit possèdent l'accès au
  paramètre, son domaine et la préparation qui dépend de sa valeur.
- `SensitivityRequest` demande un sous-ensemble d'ordre un, diagonal et
  mixte, ou la Hessienne complète. `SensitivityStencil<C>` conserve les
  points FP32 axiaux et leurs poids ; `SensitivityGraphPlan` mutualise le
  central, les axes et les quatre coins de chaque paire sélectionnée.
  `SensitivityTask` regroupe nœuds, stencil et `CentralRequirement`.
  `reconstruct_sensitivity` et la reconstruction tensorielle possèdent les
  règles communes de reconstruction.
- La préparation part des lignes centrales modèle/produit/courbe et des `K`
  spécifications. La matérialisation hôte de `N*(1+2K)` lignes bumpées n'est
  pas la cible finale, y compris pour LSM et Jamshidian. Une préparation
  lourde partagée, telle que l'ajustement d'un noyau N-facteurs, peut rester
  hors du kernel chaud, à condition de ne pas réintroduire cette matrice de
  scénarios ou de cacher son coût dans les timings. Les `N*K` stencils de
  sortie restent utiles comme preuve des points utilisés.
- Les entrées du pricer ne dépendent pas de `K` quand la sélection est vide.
  Sur les tâches Monte Carlo compatibles, la coordonnée sélectionnée est
  distribuée entre blocs et les chemins au sein des blocs. Le premier bloc de
  sensibilité possède le prix ; tous les blocs lisent la même clé de ligne et
  le même indice de chemin. Les formules fermées et LSM gardent leur
  distribution propre si elle est mesurée et justifiée.
- Le prix central ne doit pas dépendre du nombre, de l'ordre ou du sous-ensemble
  de sensibilités. Le bruit Monte Carlo d'une dérivée provient des différences
  **couplées chemin par chemin** ; les moments sont réduits en FP64. Aux
  frontières, les stencils unilatéraux autorisés utilisent les valeurs
  représentées, sans clamp silencieux. Un échec de domaine est explicite.
- La durée numérique, les dates contractuelles et les paramètres continus ne
  sont pas interchangeables. Un bump de maturité d'un moteur à pas respecte
  la grille entière ; une date ou un nombre de paiements discret n'est pas
  automatiquement une coordonnée différentiable.

Les responsabilités restent lisibles dans l'arborescence :

| Responsabilité | Propriétaire |
|---|---|
| Sélection, tâche, stencil, reconstruction, sorties et validation commune | `src/common/price_gradients/` |
| Distribution Monte Carlo mono, politique terminale partagée et replay des innovations | `src/common/monte_carlo/price_gradients/` |
| Évaluation, reconstruction et finalisation du graphe de nœuds terminal | `src/common/monte_carlo/price_gradients/terminal_node_graph/` |
| Formules scalaires et coopératives | `src/common/closed_form/price_gradients/` |
| Stratégie LSM centrale et replay gelé | `src/common/longstaff_schwartz/price_gradients/` |
| Workspace, convolution et réduction rough FFT, lors de leur ajout | `src/common/volterra/price_gradients/` |
| Paramètre, domaine, nœuds et dynamique couplée spécifiques | `src/model/equity/{markovian,rough}/<model>/price_gradients/` ou `src/model/fixed_income/<model>/price_gradients/` |
| Paramètres et observations du produit | `src/product/<product>/price_gradients/` |
| Façades, recettes et matrice de capacités | `src/model/**/product/`, `tools/codegen/pricing_bindings/`, `catalog/` |

Une extraction vers `src/common` n'a lieu que lorsqu'au moins deux familles
partagent effectivement la même logique. Ne pas déplacer les équations d'un
modèle dans un moteur générique pour rendre un binding plus court. Les
adaptateurs et fichiers sont ajoutés par propriétaire au fur et à mesure ; ce
tableau n'impose aucun dossier vide.

## Étapes et critères de passage

### 0. Figer l'inventaire et les références

Construire depuis le manifeste et les sources une matrice **prix →
prix-delta → prix-gradients** par modèle, courbe, produit et moteur. Pour
chaque ligne, noter les coordonnées continues admissibles, les limites
explicites, le calendrier, le mapping Philox, la préparation du modèle, le
type de payoff et le coût de la pipeline. Conserver des fixtures communes
avec mêmes modèles, produits, `dt`, seed, nombre de chemins, géométrie et
toolchain. Capturer prix, delta spot, erreurs standards, stencils, temps de
bout en bout, registres, mémoire locale/partagée et taille de workspace.

**Sortie :** chaque consommateur `price_delta` et chaque voie gradients
existante a un propriétaire, une référence et une décision de migration. Les
limites non qualifiées sont visibles dans la matrice, sans être déclarées
supportées par défaut.

### 1. Consolider la vue préparée

**État au 22 septembre 2026 : achevé pour les bindings actifs.** Les terminaux
equity, la formule fermée Black--Scholes, le replay LSM Heston et Jamshidian
CIR emploient les mêmes `SensitivitySpec`, `SensitivityRequest`,
`SensitivityTask`, `SensitivityStencil` et `SensitivityResult`. LSM et CIR ne
matérialisent plus `N*(1+2K)` scénarios côté hôte. La régression LSM centrale
et le solveur Jamshidian coopératif restent leurs briques canoniques ; seule la
préparation des nœuds a été déplacée au point d'usage. Les anciens kernels et
politiques fondés sur la matrice hôte ont été retirés.

Cette étape reste une condition à vérifier lors de chaque nouveau binding :
les précomputations invariantes au niveau modèle, courbe ou calendrier peuvent
rester hors du kernel chaud, mais ne doivent pas recréer implicitement la
matrice de scénarios.

Pour les terminaux Monte Carlo, la stratégie mono et le graphe de nœuds à
trois phases partagent `SensitivitySpec`, `SensitivityTask`,
`SensitivityStencil`, `TerminalNodePolicy`, les entrées préparées et les
sorties. La stratégie choisie ne change que l'ordonnancement et son workspace.
La stratégie mono garde ses nœuds et moments dans le bloc ; le graphe écrit les
observations de nœuds, reconstruit les dérivées, puis finalise les moments. La
parité bitwise sur prix, erreurs, gradient et Hessienne diagonale est le garde
de non-régression entre les deux pour les demandes qu'elles savent toutes deux
exprimer. Le choix automatique de stratégie reste une étape séparée fondée sur
les ressources compilées et une qualification par couple et matériel.

**Sortie :** sélection vide, une sensibilité, sous-ensemble et permutation
passent sur les voies concernées ; le prix central est écrit une fois, les
stencils et résultats correspondent aux points effectivement évalués, et la
pipeline complète n'a pas de régression de coût non expliquée. Aucun moteur
déclaré migré ne conserve une matrice hôte `N*(1+2K)` de scénarios.

### 2. Couvrir les modèles et produits markoviens

**État au 26 septembre 2026 : porte d’intégration fermée.** Le manifeste
contient 301 bindings `price_gradients`, tous compilés : 261 equity et 40 fixed
income. La couverture equity se décompose en 55 bindings Monte Carlo
terminaux, 189 bindings Monte Carlo de chemin, huit bindings Black--Scholes en
formule fermée et neuf bindings américains LSM. La couverture fixed income se
décompose en 20 formules fermées scalaires, sept swaptions européennes
Jamshidian coopératives, trois swaptions européennes Monte Carlo exactes et dix
swaptions bermudéennes LSM.

Les moteurs stochastiques d’ordre un plus deux diagonal exposent les deux
ordonnancements `mono` et `node_graph` sur le même plan compact. Les
requêtes contenant des termes mixtes utilisent `mixed_node_graph`, qui
évalue chaque nœud partagé une fois avant reconstruction. Les formules
fermées conservent leur géométrie naturelle : un thread par ligne pour les
formules scalaires, un bloc par ligne pour Jamshidian. Les pipelines LSM
calculent une seule politique centrale, puis évaluent les nœuds sous exercice
gelé. Les produits de chemin calculent simultanément l’état et les cashflows
de chaque nœud à partir des mêmes innovations primitives.

La coordonnée virtuelle `product.maturity_years` est intégrée à toutes les
voies markoviennes européennes : terminales, produits de chemin, formules
fermées equity, formules scalaires fixed income, Jamshidian et swaptions
européennes Monte Carlo. La préparation garde le calendrier central fixe et
ne déplace que sa dernière date ; les préfixes réutilisent les mêmes
innovations. American et Bermudan restent explicitement exclus. Les recettes
codegen sélectionnent `T` avec un bump absolu `1/504` et dimensionnent leurs
capacités de graphe en conséquence.

Le catalogue contient 2 664 recettes permanentes : 2 184 equity et 480 fixed
income. Chaque famille fournit l’ordre un, l’ordre un plus Hessienne diagonale
et la Hessienne complète, ainsi que les constructions alignée et cartésienne
lorsque la recette de prix source les expose. Les 301 bibliothèques de binding ont été
reconstruites sans erreur ; les 160 nouveaux générateurs scalaires fixed income
ont aussi été compilés exhaustivement. Une régénération propre du codegen ne
produit aucune divergence.

Les exclusions structurelles restent explicites : maturité et dates
d’exercice des américaines/bermudéennes, cardinalités calendaires discrètes et
moteurs rough. La présence d’une coordonnée continue dans le
binding prouve son intégration logicielle, pas la qualité universelle de son
bump. Les campagnes multi-seeds, l’étude du biais de stencil et du freeze LSM,
ainsi que la qualification de performance par matériel restent la porte
numérique suivante.

**Sortie atteinte :** aucun couple markovien déjà priceable et déclaré
admissible par le manifeste n’attend une brique de préparation, un launcher ou
un générateur `price_gradients`. Les futurs ajouts passent par les propriétaires
modèle/produit/courbe et réutilisent les moteurs communs ; ils ne recopient ni
transition, ni observation, ni réduction.

### 3. Qualifier sauts et temps sur le markovien

**État d’intégration au 26 septembre 2026 : achevé sur les coordonnées
admissibles.** Merton, Kou et Bates disposent de comptes emboîtés et de marques
centrales rejouables pour intensité, paramètres de marque et maturité
européenne. Variance-Gamma et NIG déclarent leur couplage propre de
subordonnateur. Les produits terminaux, de chemin et les replays LSM utilisent
les mêmes adapters, avec le mapping Philox V2 et ses domaines seulement lorsque
la consommation variable l’exige.

La maturité européenne est exposée quand l’adapter définit son couplage :
pont/extension brownien pour les transitions directes et poursuite d’une grille
commune pour les schémas à pas. La maturité d’une américaine ou d’une
bermudéenne, les dates d’exercice et les cardinalités calendaires restent
exclues. Elles demandent une convention financière distincte et ne se réduisent
pas à modifier un scalaire `T`.

Le [plan de sauts](jump-price-gradient-migration-plan.md) et le
[plan Philox](philox-domain-migration-plan.md) restent les propriétaires de la
qualification. Les tests d’intégration prouvent le central partagé, le rejeu,
la parité des deux ordonnancements et la sûreté mémoire sur des représentants.
Ils ne suffisent pas à certifier une intensité, un paramètre de marque ou une
maturité pour toutes les largeurs de bump et toutes les seeds.

**Sortie logicielle atteinte :** les coordonnées déclarées possèdent une loi,
un compensateur, un couplage et un stencil implémentés. **Sortie numérique
ouverte :** NUM-031 et NUM-032 conservent les campagnes de collisions/bornes,
de lois, de bumps, de forte intensité, de registres et de temps complet avant
qualification de publication.

### 4. Fermer la porte markovienne avant le rough

Exécuter la matrice de tests ci-dessous sur equity **et** fixed income,
pour prix seul, `K=1`, sélection partielle et sélection complète. Qualifier
sur plusieurs seeds les bumps et les frontières, en distinguant bruit Monte
Carlo, biais de stencil, discrétisation temporelle, erreur FP32 et, pour LSM,
biais de politique gelée. Les références indépendantes de prix ne certifient
pas à elles seules les gradients. Consigner les limites par coordonnée et
par produit avant de passer au rough.

**Sortie :** couverture, preuves numériques, ressources et temps complets
sont publiés dans les contrats/rapports appropriés. Une seule fixture Heston
rapide ne ferme pas cette porte. Le passage au rough requiert la clôture des
lignes markoviennes prévues, hormis les coordonnées explicitement hors domaine
(par exemple dates discrètes ou maturité d'exercice anticipé) ; un échec de
qualification n'est pas reclassé en exclusion pour franchir la porte.

### 5. Étendre aux deux familles rough

Commencer par les **lifts N-facteurs** : ils réutilisent le moteur de chemins
markovien après un fit hôte coûteux. Une perturbation de `H` modifie les poids
et taux du fit ; chaque nœud `H`, y compris les points unilatéraux, doit
recevoir sa propre préparation, avec les **mêmes innovations primitives**.
Mesurer fit, transferts, kernels et coût complet, ainsi que l'erreur de
factorisation et de grille. Ne pas assimiler une dérivée du modèle approché
à celle du modèle rough limite sans étude de convergence.

Poursuivre avec les **moteurs Volterra FFT**. Conserver la pipeline existante
de préparation du noyau, convolution par chunks de chemins, évolution du
modèle, observation produit et réduction. Si une coordonnée laisse inchangés
le noyau rough et la grille, une convolution donnée peut alimenter les nœuds
bumpés de cette coordonnée ; seules la dynamique et les observations qui en
dépendent sont recalculées. Si `H` change, les poids, le spectre, la variance
et la partie locale du schéma hybride changent : le central et les extrémités
requièrent **plusieurs convolutions** du même chemin brownien, et plus encore
si un stencil unilatéral ajoute un point. Leurs tirages primitifs doivent
rester identiques. Étudier la réutilisation de la transformée directe du
brownien entre noyaux, mais ne présumer ni sa rentabilité ni une seule
convolution par prix. Un bump de maturité peut aussi changer grille, longueur
FFT ou horizon ; sa stratégie CRN doit être définie et mesurée séparément.

Le workspace et le planner comptent spectra, convolutions, moments et
stencils selon le nombre de nœuds réellement actifs et la taille des chunks.
Choisir la répartition blocs/threads après mesure des registres, spills,
accès contigus par warp, VRAM et temps complets. Préparer Heston rough et
rough Bergomi comme pilotes distincts, puis parcourir la matrice rough du
manifeste sans présumer que les six modèles partagent le même coût.

**Sortie :** CRN démontré pour `H` et les autres coordonnées annoncées, erreurs
séparant bump et approximation rough, prix central unique, géométrie et
mémoire qualifiées ; chaque couple rough couvert dispose de tests et d'une
provenance versionnée.

### 6. Retirer `price_delta`

Une fois ses consommateurs actifs, y compris rough, couverts et comparés,
faire pointer launchers, codegen, recettes, tests, runners et documentation
vers `price_gradients` avec la sélection `{model.spot}` et le bump historique
approprié. Vérifier la parité bit à bit lorsque **mapping RNG, points
représentés, ordre de réduction, géométrie, GPU et toolchain** sont identiques ;
sinon produire une comparaison numérique et expliquer la version qui change.
Supprimer ensuite kernels, adaptateurs, templates, cibles et descriptions
`price_delta` sans consommateurs. Les artefacts historiques publiés gardent
leur recette originale, leur qualification et leurs sources gelées dans la
preuve de campagne selon le
[contrat de réutilisation des datasets](../dataset-provenance-contract.md) ;
ils n'imposent pas de conserver des recettes `price_delta` comme capacités
actives du catalogue. Les archives d'audit ne sont pas réécrites pour simuler
une migration rétroactive. Le constat [DELTA-001](../audit/response.md#delta-001--déployer-et-qualifier-la-voie-prix-delta-sans-dupliquer-les-moteurs-métier)
reste ouvert tant que ses propres critères ne sont pas remplis.

## Preuves requises à chaque migration

- **Fonctionnel :** `K=0`, `K=1`, plusieurs coordonnées, permutations,
  coordonnées modèle/produit/courbe, frontières et refus de domaine ; stencils
  exacts, prix central écrit une fois, reprise après checkpoint et aligné vs
  cartésien. La sélection `{model.spot}` est comparée à `price_delta` dans
  les conditions de parité ci-dessus.
- **Stochastique et numérique :** mêmes innovations `(row,path)` pour tous
  les nœuds qui doivent être couplés ; tests de lois et références
  indépendantes adaptés au modèle ; plusieurs seeds et largeurs de bump ;
  convergence en chemins, en `dt` ou en facteurs quand applicable ; erreur
  standard de la différence couplée, biais de stencil et FP32 rapportés
  séparément. Les discontinuités de payoff, les sauts et l'exercice ont leurs
  campagnes propres.
- **CUDA et performance :** tests hôte/codegen, CUDA et Compute Sanitizer
  pertinents ; `git diff --check` et régénération codegen zéro-diff. Mesurer
  sur le même GPU et la même toolchain les kernels **et le mur complet**
  incluant préparation, allocations, copies, fit éventuel, convolution et
  réduction. Examiner registres, mémoire locale, spills, shared, occupation,
  accès mémoire et nombre d'appels Philox. `B=1` et 256 threads restent la
  référence MC terminale mesurée, pas une règle imposée aux autres moteurs.
  Suivre le [protocole de performance](../performance-regression-protocol.md).
- **Catalogue et provenance :** manifeste et bindings générés cohérents avec
  les capacités testées ; recettes déclarant sélection, politique de bump,
  calendrier, méthode, RNG et version. Une modification du central, du
  mapping ou de la méthode crée une nouvelle identité de dataset ; une reprise
  vérifie les empreintes de ses sources et ne mélange pas deux versions.

Le [contrat de qualification des bumps](price-gradient-bump-qualification-policy.md)
ne couvre actuellement qu'une partie des calls européens BS/Heston/CEV ; ses
verdicts ne s'étendent ni aux autres modèles ni aux produits rough. Chaque
nouvelle surface élargit explicitement sa politique et conserve ses échecs.
