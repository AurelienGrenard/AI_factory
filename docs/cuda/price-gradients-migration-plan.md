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
- `SensitivityRequest` demande l'ordre un, l'ordre deux diagonal ou les deux.
  `SensitivityStencil<C>` conserve les points FP32 effectivement représentés
  et leurs poids ; `SensitivityNodes<Node,C>` conserve les jeux de paramètres
  correspondants. `SensitivityTask` regroupe ces objets et
  `CentralRequirement`. `reconstruct_sensitivity` possède la règle commune
  de reconstruction. Les dérivées mixtes demandent une extension séparée ;
  elles ne sont pas promises par la présence d'un stencil diagonal.
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
| Distribution Monte Carlo, réduction et replay des innovations | `src/common/monte_carlo/price_gradients/` |
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

**Sortie :** sélection vide, une sensibilité, sous-ensemble et permutation
passent sur les voies concernées ; le prix central est écrit une fois, les
stencils et résultats correspondent aux points effectivement évalués, et la
pipeline complète n'a pas de régression de coût non expliquée. Aucun moteur
déclaré migré ne conserve une matrice hôte `N*(1+2K)` de scénarios.

### 2. Couvrir les modèles et produits markoviens

**Avancement au 23 septembre 2026 :** les payoffs terminaux européens générés
(`european_option`, `asset_or_nothing_option`, `digital_option`) exposent les
ordres un et deux diagonal pour Black--Scholes, CEV, Heston, Heston 3/2, SABR,
Schöbel--Zhu, Stein--Stein, Merton, Kou, Bates, Variance-Gamma et NIG selon les
bindings de prix existants. Les transitions à sauts finis partagent le replay
événementiel commun ; VG et NIG partagent leurs primitives de subordination.

Le moteur américain gelé couvre désormais tous les modèles equity disposant
d'un binding LSM dans ce périmètre : Black--Scholes, CEV, Heston,
Schöbel--Zhu, Merton, Kou, Bates, Variance-Gamma et NIG. Il distingue deux
replays communs : pas fixe sur les intervalles numériques canoniques et
transition exacte une fois par intervalle contractuel. Le prix et la politique
d'exercice centraux sont calculés une seule fois. La maturité et les dates
d'exercice américaines restent exclues. La swaption européenne CIR/Jamshidian et les dix compositions Bermudan
fixed income sont maintenant intégrées. Les Bermudans couvrent CIR, CIR++,
G2, G2++, Hull--White, OU et Vasicek, avec Nelson--Siegel ou Svensson lorsque
le modèle exige une courbe. Leur central LSM est calculé une fois, puis les
nœuds sont rejoués sous politique d'exercice gelée. Les modèles, courbes et
produits partagent leurs domaines entre loaders et préparation GPU. Les
produits de chemin equity et les autres produits de taux restent hors de cette
tranche.

Ajouter par futur modèle seulement l'accès typé aux paramètres, le domaine,
la préparation et l'adaptation des transitions. Ajouter par produit la
préparation du payoff et de son calendrier. Réutiliser les moteurs et
observers existants pour terminaux, produits de chemin, formules fermées,
Monte Carlo exact ou à pas, et LSM gelé. Les paramètres de CIR tirés via
Poisson--Gamma demandent toujours une preuve spécifique de couplage ; le flux
historique ne vaut pas validation implicite.

Pour l'exercice anticipé, enregistrer une seule politique centrale et rejouer
les trajectoires bumpées contre ses décisions gelées. Comparer sur des cas
bornés au recalibrage de la stratégie afin de quantifier le biais du freeze ;
ne pas confondre ce résultat avec une dérivée de la politique réentraînée.

**Sortie :** la matrice markovienne est parcourue, les capacités demandées
sont intégrées ou assorties d'une exclusion justifiée par le contrat du
modèle/produit. Chaque capacité encore manquante reste une ligne ouverte de
la matrice et empêche de déclarer le périmètre markovien achevé. Aucune formule,
transition, observation ou réduction n'est recopiée dans un binding.

### 3. Achever sauts et temps sur le markovien

Les sensibilités de paramètres ordinaires et de maturité déjà présentes
restent les témoins de compatibilité. Étendre la maturité européenne là où le
couplage des chemins est défini : pont/extension brownien pour les transitions
terminales directes, continuité des chemins sur les grilles à pas, cohérence
des observations et de l'actualisation. Le saut de date ne se réduit pas à
changer une valeur `T` dans un payoff. La maturité d'une américaine ou d'une
bermudéenne, ainsi que les dates contractuelles discrètes, restent exclues de
ce lot jusqu'à une politique d'exercice et de calendrier propre.

Appliquer le [plan de sauts](jump-price-gradient-migration-plan.md) à Merton,
Bates, Kou et aux autres modèles admissibles, avec son central événementiel,
ses comptes couplés et ses marques rejouables. Le
[plan Philox](philox-domain-migration-plan.md) fournit, seulement si
nécessaire, des domaines internes indépendants pour éviter qu'une consommation
variable de tirages décale le brownien, le pas suivant ou une autre source.
Le flux unique reste prioritaire lorsque son ordre fixe donne déjà le CRN.
Versionner le mapping et les recettes : les prix centraux d'une **nouvelle**
version doivent coïncider bit à bit entre prix seuls et prix-gradients, sans
exiger une parité bit à bit avec l'ancien tirage agrégé des sauts.

**Sortie :** les coordonnées déclarées ont une loi, un compensateur, un
couplage et un bump qualifiés. Les autres restent explicitement absentes.
NUM-031 et NUM-032 conservent leurs critères de clôture propres ; ce plan ne
les remplace pas.

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
