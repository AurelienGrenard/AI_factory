# Prix, gradients et Hessiennes fixed income

## Périmètre actif

Quatre formes d’exécution fixed income partagent le même contrat de
sensibilités sélectionnées :

1. 26 bindings de formule fermée scalaire pour caplets/floorlets et options
   call/put sur zéro-coupon, sur les treize compositions modèle/courbe ;
2. neuf bindings de swaption européenne par Jamshidian coopératif ;
3. quatre bindings de swaption européenne Monte Carlo exacte pour G2 et G2++ ;
4. treize bindings de swaption bermudéenne Longstaff--Schwartz à exercice central
   gelé.

Ces 52 bindings publient le prix central, les dérivées premières, la
diagonale et les dérivées mixtes sélectionnées. Les formules scalaires
attribuent une ligne à un thread, Jamshidian une ligne à un bloc coopératif.
Les voies stochastiques exposent `mono` et `node_graph` pour les demandes
sans terme mixte, puis `mixed_node_graph` pour une Hessienne sélectionnée ou
complète.

Les treize compositions modèle/courbe sont CIR, CIR++
Flat/Nelson--Siegel/Svensson, G2, G2++ Flat/Nelson--Siegel/Svensson,
Hull--White Flat/Nelson--Siegel/Svensson, Ornstein--Uhlenbeck et Vasicek. Les
modèles ajustés ajoutent les coordonnées de leur courbe. Les domaines
appartiennent au modèle, à la courbe ou au produit et sont partagés par le
loader et la préparation des nœuds.

Les champs calendaires entiers restent discrets : dates d’exercice, intervalles
de paiement, nombres de paiements et d’exercices ne sont pas des coordonnées
différentiables. Les bindings européens exposent toutefois la coordonnée
virtuelle continue `product.maturity_years`, avec un bump absolu minimal
`dt = 1/504` par défaut. Elle déplace uniquement la dernière date
contractuelle : date de paiement du rate option, maturité du zéro-coupon pour
l'option sur bond, dernier paiement de la swaption européenne. Le fixing,
l'expiry/exercice, les paiements antérieurs, les accruals antérieurs et la
cardinalité du calendrier restent fixes. Pour une swaption, l'accrual du dernier
coupon devient le stub `T_node - t_{m-1}`. Le nœud doit rester strictement
postérieur à la date fixe précédente ; sinon la préparation emploie le repli unilatéral d'ordre
deux demandé ou refuse le stencil.

Le central conserve le type et le chemin de calcul du pricer historique. Une
vue gradient-only ne substitue la dernière date que pour un nœud réellement
bumpé, ce qui préserve ses bits. Pour Bermudan, la maturité d’exercice anticipé
reste exclue. Les paramètres continus disponibles sont ceux que chaque binding
résout explicitement ; une recette peut en sélectionner un sous-ensemble sans
les déduire de leur valeur.

## Plan compact et préparation des sensibilités

L'hôte conserve uniquement :

```text
models[M]
curves[C]                    # modèles ajustés uniquement
products[P]
sensitivities[K] = {paramètre résolu, BumpConfiguration}
construction = aligned | cartesian
SensitivityRequest = first | second | first_and_second | selected/full_hessian
```

Il ne matérialise pas `N*(1+2*K)` scénarios. Le mapping canonique retrouve le
modèle, la courbe éventuelle et le produit de chaque ligne. Les noms sont
résolus une fois sur l'hôte en identifiants typés ; aucun kernel ne compare de
chaînes.

Pour chaque couple `(ligne, sensibilité)`, le thread 0 du bloc appelle
`build_sensitivity_task`. Le couple formé par les paramètres centraux et le
`SensitivitySpec` construit :

- trois nœuds pour l'ordre un et les stencils diagonaux centrés ;
- quatre nœuds pour une diagonale unilatérale d'ordre deux ;
- quatre coins tensoriels par paire mixte sélectionnée, en réutilisant le
  central et les nœuds axiaux déjà nécessaires ;
- le `SensitivityStencil` avec les valeurs FP32 et les poids effectivement
  utilisés ;
- le besoin éventuel du nœud central pour reconstruire la dérivée.

Une frontière produit un stencil unilatéral d'ordre deux. Aucun clamp
silencieux n'est admis. Les `N*K` stencils sont des sorties de preuve ; les
entrées restent en `O(M+C+P+K)`.

## Formules fermées scalaires

Les caplets/floorlets et options sur zéro-coupon emploient un thread par ligne.
Ce thread charge le modèle, la courbe éventuelle et le produit centraux, puis
construit et évalue successivement les nœuds utiles du graphe sélectionné. Il
n’existe ni workspace Monte Carlo ni flux Philox. Le même launcher couvre
ordre un, diagonale et termes mixtes ; seul le `SensitivityRequest` choisit
les sorties reconstruites.

## Swaptions européennes Monte Carlo exactes

G2 et G2++ utilisent leurs transitions exactes état/intégrale avec innovations
communes. La voie `mono` garde les nœuds et moments dans le bloc pour les
demandes sans dérivée mixte. Les voies `node_graph` et
`mixed_node_graph` séparent évaluation des nœuds, accumulation des moments
et finalisation, avec la même préparation et les mêmes stencils. Le graphe
mixte mutualise central, axes et coins entre toutes les sorties demandées.

## Jamshidian coopératif

Les neuf compositions analytiques utilisent les mêmes tâches, stencils et
reconstruction que les formules scalaires. Un bloc possède une ligne et
coopère sur la recherche de la racine et la somme des options sur
zéro-coupon. Le modèle et la courbe éventuelle restent propriétaires de leurs
analytics ; le moteur commun ne connaît aucun modèle concret.

Le central est écrit une fois. Le bloc réutilise son workspace partagé pour
chaque nœud. `reconstruct_sensitivity` et la reconstruction tensorielle
restent les logiques communes. Le graphe de sélection décrit les nœuds utiles,
mais la famille analytique les évalue dans son unique kernel coopératif ; elle
n'utilise pas les trois phases Monte Carlo.

Les mesures historiques du 22 septembre 2026 sur SM89, pour 1 000 lignes,
sept sensibilités et au plus douze paiements, donnent 1,114 ms pour l’ordre un
et 1,841 ms pour ordre un plus diagonale sur la voie coopérative. Elles ne
qualifient ni les autres familles fixed income ni un autre GPU.

## Longstaff--Schwartz à exercice gelé

La pipeline centrale reste celle du prix seul : simulation forward,
régressions backward, décisions d’exercice puis réduction. Elle enregistre
l’exercice central de chaque chemin et calcule le prix une seule fois.

Si `K>0`, deux ordonnancements sont disponibles :

- `mono` rejoue les nœuds d’une sensibilité dans
  `frozen_sensitivity_moments_kernel`, puis les réduit dans
  `finalize_frozen_sensitivities_kernel` ;
- `node_graph` évalue chaque nœud utile dans `frozen_node_evaluation`,
  reconstruit les moments transitoires dans
  `frozen_node_moment_accumulation`, puis finalise dans
  `finalize_frozen_node_sensitivities` ;
- `mixed_node_graph` applique la même décomposition au graphe complet,
  mutualise les nœuds partagés et reconstruit les termes mixtes sous la même
  politique centrale gelée.

Les deux stratégies partagent la préparation compacte, les stencils, la trace
d’exercice et les policies de dynamique/payoff. CIR/CIR++ utilisent leur
mesure terminal-forward ; G2, G2++, Hull--White, OU et Vasicek utilisent la
voie jointe état/intégrale. Cette différence appartient aux policies, pas aux
kernels communs.

Le cas `K=0` délègue au prix seul. Le planner compte la trace d’exercice, les
nœuds, les moments et les sorties dans la VRAM avant de choisir le nombre de
lignes résidentes. Le graphe conserve une réduction complète par ligne quand
son workspace est admissible et refuse une géométrie qui changerait
silencieusement l’ordre de sommation.

## Génération et provenance

Le codegen produit 208 recettes fixed income permanentes : 52 pour les
produits de taux scalaires, 52 pour les options sur zéro-coupon, 52 pour les
swaptions européennes et 52 pour les bermudéennes. Chaque famille couvre ses
deux côtés et les constructions alignée et cartésienne. Le catalogue permanent
publie l’ordre un avec la diagonale de la Hessienne. Les APIs et templates de
codegen conservent les sélections d’ordre un et la Hessienne mixte pour les
générations ponctuelles, sans multiplier les recettes permanentes.

Les recettes analytiques conservent l’URL `/v1/` de leur méthode déterministe.
Les recettes stochastiques emploient `/v2/`, une seed et le générateur
`philox`. Toutes déclarent les datasets modèle/courbe/produit,
les bumps sélectionnés et la recette de prix source. La génération écrit un
checkpoint durable après chaque lot et reprend au premier préfixe incomplet.

## Responsabilités

| Responsabilité | Emplacement |
|---|---|
| Specs, tâches, stencils et reconstruction | `src/common/price_gradients/` |
| Plan compact modèle/courbe/produit et vue de dernière date | `src/common/fixed_income/price_gradients/` |
| Workspace et kernels LSM gelés communs | `src/common/longstaff_schwartz/price_gradients/` |
| Composition Bermudan et replay du payoff | `src/product/bermudan_swaption/price_gradients/` |
| Accès et domaines modèle | `src/model/fixed_income/<model>/{parameter_domain.hpp,price_gradients/}` |
| Accès et domaines courbe | `src/curve/<curve>/{parameter_domain.hpp,price_gradients/}` |
| Bindings publics | `src/model/fixed_income/<model>/product/.../<product>_price_gradients.cuh/.cu` |
| Génération, checkpoint et artefact | `tools/pricing/price_gradients/`, `tools/datasets/price_gradients/` |
| Templates, manifeste et recettes | `tools/codegen/pricing_bindings/`, `catalog/model/fixed_income/` |

## Preuves et limites

Les 52 bibliothèques fixed income et les 208 recettes permanentes sont dérivées
du même manifeste de capacités. La régénération complète avec comparaison au
dépôt, les contrôles du catalogue et la compilation des générateurs constituent
le contrôle d’exhaustivité structurelle. Les templates de Hessienne complète
restent vérifiés par leurs représentants dédiés sans dupliquer ces variantes
dans le catalogue permanent.

Les tests CUDA permanents couvrent les formules fermées, Jamshidian, G2/G2++
Monte Carlo et les familles Bermudan CIR terminal-forward, Vasicek un facteur,
G2 deux facteurs et CIR++ ajusté. Ils contrôlent prix central, ordres un et
deux, paramètres modèle/courbe/produit, sélections réordonnées et parité des
stratégies sur leur périmètre commun. La suite globale `price_gradients`
passe 27/27 tests. Les quatre modes Compute Sanitizer passent sur un
représentant Bermudan.

Sur SM89, l’inspection historique du kernel mono G2 Bermudan donne 168
registres/thread et 32 octets de stack à l’ordre un, puis 242 registres/thread
et 464 octets de stack à l’ordre un plus deux diagonal. Cette pression justifie
le graphe de nœuds pour les demandes plus larges, mais ne constitue pas une
qualification de performance universelle.

Deux exécutions temporaires de générateurs scalaires valident aussi la chaîne
complète. Vasicek/caplet aligné et CIR++/Nelson--Siegel/option zéro-coupon
aligné produisent chacun 1 000 lignes dont les prix centraux sont exactement
égaux aux bases de prix. Une fixture cartésienne `2 x 2 x 2` vérifie le chemin
à trois axes, la même parité centrale et la finitude des gradients/Hessiennes.
La recette cartésienne construite avec les trois fichiers permanents de 1 000
lignes représente un milliard de sorties ; elle dépasse la VRAM de la machine
de test et n’est pas une campagne qualifiée sur ce matériel.

Ces contrôles qualifient l’intégration, la compilation et la reproductibilité.
Ils ne certifient pas encore les bumps, le biais de politique gelée, la
convergence LSM ni la performance de production. Une campagne numérique
multi-seeds et des mesures complètes sur secteur restent nécessaires avant de
publier une base de Hessiennes comme qualifiée.
