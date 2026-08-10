# Pipeline de validation independante des prix

Ce document definit la decouverte et la selection des moteurs de reference, le
traitement des echecs et les metadonnees obligatoires des bases de prix. La
certification independante porte exclusivement sur les 900 lignes `core` et,
lorsque le backend le permet, se route ligne par ligne. Les 100 lignes `stress`
restent dans le dataset mais servent uniquement aux controles internes de
robustesse. L'existence d'une methode etablit sa disponibilite potentielle;
seule son execution permet ensuite d'etablir qu'elle valide effectivement une
ligne core.

## Regimes

Les bases ordonnees selon la convention 90/10 conservent deux roles distincts:

- `core`: les 900 premieres lignes, representant le domaine usuel et constituant
  l'unique perimetre de certification independante;
- `stress`: les 100 dernieres lignes, volontairement plus larges, utilisees
  uniquement pour tester la robustesse numerique interne.

Le statut public `validated` est determine uniquement par le core. Une
certification core reussie doit etre decrite sans ambiguite par la phrase:

> Dataset valide independamment sur son domaine core. Les lignes stress testent
> la robustesse numerique et ne sont pas couvertes par la certification externe.

Il ne faut jamais presenter cette situation comme une validation `1000/1000`:
la couverture externe est `900/900 core`. Les lignes stress ne sont ni
supprimees ni assouplies pour faire passer un test. La pipeline standard ne
lance aucun pricer Premia ou QuantLib sur elles. Elle controle seulement les
invariants internes applicables: valeurs et erreurs standards finies,
non-negativite et bornes de payoff, parites exactes, martingalite, coherence de
l'erreur Monte-Carlo et convergence ciblee lorsque le produit l'exige. Une
comparaison externe stress peut etre executee ponctuellement pour la recherche,
mais elle reste hors certification, hors statut YAML et hors decision de
publication.

## Inventaire Premia obligatoire

Avant toute validation longue, construire l'inventaire exhaustif des moteurs
Premia du couple exact `(modele, produit)`. Cette recherche precede le choix de
la methode et ne doit pas s'arreter au premier pricer trouve. Elle doit:

1. parcourir tous les menus/classes d'actifs Premia, car un moteur peut etre
   enregistre hors du menu intuitif du modele;
2. enumerer les options et toutes leurs methodes depuis la bibliotheque, puis
   confronter cet inventaire aux sources et a la documentation Premia;
3. chercher le contrat direct, mais aussi les reductions exactes composees de
   moteurs Premia, telles qu'une parite ou une somme de digitals;
4. conserver le nom natif, le domaine de parametres, les conventions du contrat
   et les reglages numeriques de chaque candidat;
5. conclure `Premia indisponible` seulement apres avoir documente qu'aucun
   candidat compatible n'existe dans l'inventaire complet.

La formule fermee, l'approximation, l'arbre, les differences finies et le
Monte-Carlo sont tous des candidats recevables. Leur famille numerique ne
decide pas de leur disponibilite. Une difference entre monitoring continu et
discret ne retire pas non plus le couple de l'inventaire: elle est traitee par
le critere de comparaison et l'explication du biais.

Lorsque plusieurs methodes existent, effectuer un sondage representatif du
core, puis les ordonner. Le moteur principal doit etre compatible et robuste;
entre candidats comparables, retenir le plus rapide. Les moteurs suivants
restent des replis Premia actifs, et non de simples noms documentaires.

Un biais signe important sans explication contractuelle declenche
obligatoirement un diagnostic avec les autres methodes compatibles. Si ce
diagnostic montre que le moteur principal est une approximation biaisee alors
qu'un autre moteur reproduit mieux le contrat sans biais, l'ordre global des
methodes est corrige avant la validation complete. Ce changement se fonde sur
un echantillon core fixe a l'avance, la fidelite du contrat et la
robustesse, jamais sur une selection ligne par ligne du prix le plus proche.
Une fois l'ordre fixe, une divergence finie reste attachee au moteur principal;
seules ses erreurs techniques descendent vers les replis.

## Hierarchie par ligne

Pour chaque ligne core, appliquer dans cet ordre:

1. moteurs Premia compatibles, dans l'ordre determine par l'inventaire et les
   sondages;
2. pricer specialise QuantLib compatible;
3. Monte Carlo QuantLib reproduisant exactement le contrat;
4. `none` lorsqu'aucune reference independante n'est exploitable.

La disponibilite de Premia se decide sur le couple exact `(modele, produit)` et
sur l'existence d'un moteur compatible dans Premia. Elle ne depend pas de la
methode numerique choisie dans AI_factory: un payoff simule sur une grille CUDA
reste eligible a un pricer Premia continu, PDE ou Monte Carlo des lors que le
contrat financier correspondant existe. Les differences de discretisation sont
traitees par le critere, une borne prouvee et l'explication du biais; elles ne
servent jamais a declarer Premia indisponible.

La premiere reference qui produit une validation reussie est conservee. Si le
moteur Premia principal valide 897 lignes core et echoue techniquement sur 3,
ces trois lignes passent d'abord aux autres moteurs Premia compatibles, dans
l'ordre declare. QuantLib ne traite que les lignes encore sans prix comparable
apres epuisement de toute la liste Premia. La couverture core finale peut donc
utiliser plusieurs methodes Premia, puis eventuellement QuantLib, sans
affaiblir la priorite donnee a Premia.

Le routage est implemente une seule fois dans `validation/hierarchy.py`. Chaque
fichier modele-produit declare la liste ordonnee complete des moteurs Premia,
puis les emplacements QuantLib specialise et QuantLib Monte Carlo; `none` reste
le resultat final implicite. Un moteur compatible fournit son adaptateur; un
moteur indisponible fournit une raison courte et n'est pas execute. Le rapport
conserve ainsi l'inventaire et l'ordre de selection, y compris les possibilites
examinees mais indisponibles. Seules les exceptions techniques descendent au
moteur suivant. Une comparaison calculee mais hors tolerance reste un echec du
moteur courant: une autre methode peut servir au diagnostic, mais ne doit pas
etre choisie retrospectivement parce que son prix est plus proche.

Chaque moteur disponible declare aussi son `pricing_method` exact. Pour Premia,
il s'agit du nom natif enregistre dans la bibliotheque, par exemple `CF_Call`,
`MC_FixedAsian_ExactMethod` ou `CF_ZBCallEuroHW2D`; les prefixes rendent la
nature de la methode directement auditable. Pour QuantLib, le champ nomme la
classe ou la fonction effectivement appelee, par exemple `BlackCalculator`,
`MCDiscreteArithmeticAPEngine` ou `G2.discountBondOption`. Une replication
statique indique toutes ses briques. Ce champ est obligatoire pour un moteur
disponible, absent pour un moteur indisponible et suit la ligne lors d'un repli.

Lorsqu'un backend de lot ne fournit pas lui-meme ses exceptions par ligne, le
routeur peut isoler les erreurs techniques par dichotomie. Les lots sans erreur
restent groupes; seules les branches fautives sont subdivisees jusqu'a obtenir
les identifiants concernes.

Une borne mathematique issue d'un contrat continu peut etre utilisee pour un
contrat discret lorsque son sens est prouve. Par exemple, une knock-out discrete
vaut au moins la knock-out continue, tandis qu'une knock-in discrete ou un
lookback discret vaut au plus son equivalent continu. Cette verification
apparait dans le critere de comparaison du rapport, jamais comme un champ YAML
`relationship` ni comme un prix exact.

## Convention temporelle

Les temps financiers du catalogue sont exprimes en annees selon `Actual/360`.
Le pont QuantLib utilise donc `ql.Actual360()` et convertit un temps `t` en
`floor(360 * t + 0.5)` jours, comme les grilles CUDA. Pour les grilles
Black-Scholes qui representent une
observation quotidienne ou une discretisation reellement requise, le pas cible
standard est `target_dt = 1 / 360`.

Trois notions restent volontairement distinctes:

- la convention de decompte `Actual/360`, qui donne le sens d'un temps annuel;
- le pas numerique `target_dt`, utilise uniquement lorsqu'une grille fine est
  necessaire;
- les dates contractuelles d'observation ou de paiement, imposees par le
  produit et jamais remplacees par une grille quotidienne artificielle.

Une transition exacte entre deux dates ne doit pas etre subdivisee pour suivre
la convention de decompte. En particulier, les autocalls et cliquets
Black-Scholes sont simules directement sur leurs dates d'observation. Les
Asian, barrieres, touch et lookback quotidiennement observes utilisent la
grille la plus proche de `1 / 360`. Les pricers analytiques terminaux n'ont pas
de `target_dt`.

## Echecs techniques et divergences

Deux situations doivent rester distinctes.

Un echec technique signifie que le backend n'a pas fourni de prix comparable:
statut d'erreur, valeur non finie, erreur standard invalide, prix pourtant fini
mais violant une borne de non-arbitrage, contrat hors domaine documente ou
methode indisponible. L'identifiant, le statut et la raison sont conserves. Cet
echec du validateur n'invalide jamais le prix CUDA. Le pipeline doit alors
essayer, pour cette seule ligne, chaque autre methode Premia compatible, puis
QuantLib specialise, puis un Monte-Carlo QuantLib independant si QuantLib sait
simuler le modele.

Une divergence signifie que le backend a produit un prix fini avec succes,
mais que notre prix ne respecte pas la tolerance, la borne ou le controle de
biais. Ce n'est pas une indisponibilite du backend. La ligne echoue et ne doit
pas etre remplacee silencieusement par une reference moins prioritaire. Les
autres methodes compatibles peuvent servir de diagnostic, mais ne doivent pas
etre choisies retrospectivement uniquement parce que leur prix est plus
proche. Si le diagnostic prouve que la premiere sortie viole une borne
financiere ou son domaine documente, elle est requalifiee en echec technique et
la hierarchie normale reprend. Il faut examiner le pricing, le budget Monte
Carlo et les conventions du cas core. Si aucune reference exploitable ne
subsiste, la ligne reste `unvalidated`: elle n'est ni acceptee ni declaree
fausse.

Les tolerances combinent erreur absolue, erreur relative et erreurs standards
independantes. Elles ne sont pas elargies pour effacer une divergence. Les
validations Monte Carlo controlent egalement le biais signe du core.
Strictement plus de 60% d'ecarts dans le meme sens declenche une alarme. Pour
les deux produits touch Black-Scholes, cette alarme provoque automatiquement
une confirmation du core complet avec 4 096 paires antithetiques QuantLib au
lieu de 1 024; seul le controle renforce est conserve dans le rapport. Si le
biais persiste, la validation echoue.

## Adjudication des lignes extremes

Cette adjudication ne concerne que les lignes core soumises a une reference
externe. Une ligne hors tolerance au premier traitement n'invalide pas
automatiquement le dataset. Elle entre dans une adjudication seulement
lorsqu'une regle generale, declaree avant l'execution, s'applique. Les cas admis
sont un repli independant apres echec technique, une borne de contrat prouvee,
une revalorisation Monte-Carlo a budget superieur ou une regle explicite de
materialite near-zero. La formulation subjective « legerement hors tolerance »
n'est jamais une regle d'acceptation.

Pour une estimation CUDA nulle, `price = 0` et `standard_error = 0` signifient
qu'aucune trajectoire n'a produit de payoff positif; ils ne prouvent pas que le
prix mathematique est nul. Les datasets Kou publies utilisent donc
`1,048,576 = 2^20` trajectoires par prix. Le diagnostic conserve le nombre de
trajectoires et, lorsqu'il est disponible, le nombre de payoffs positifs. Une
ligne rare peut etre recalculee avec un budget croissant. Si le payoff est
borne, une borne probabiliste unilaterale peut conclure; pour un payoff non
borne, un echantillon toujours nul ne suffit jamais a lui seul.

La regle Kou de materialite near-zero accepte specialement une divergence
seulement lorsque les valeurs absolues du prix CUDA et de la reference sont
toutes deux inferieures ou egales a `2e-3` fois l'echelle naturelle du prix.
Pour l'equity, cette echelle est le spot initial. Cette convention est separee
de la tolerance numerique: le rapport dit explicitement que la ligne est
acceptee comme economiquement near-zero, et non qu'elle respecte la tolerance
initiale. Un biais directionnel reste controle apres adjudication; strictement
plus de 60% des ecarts du meme signe doit etre explique ou faire echouer la
validation.

Chaque ligne acceptee apres traitement special est stockee dans le JSON avec
son `category`, son diagnostic initial, sa `resolution`, son
`acceptance_rule` et les elements d'`evidence` (prix, erreurs standards,
budget de trajectoires, moteurs employes et seuil eventuel). Le compteur
`accepted_without_special_treatment` ne les inclut jamais. Le notebook affiche
separement `accepted after special treatment`; sa conclusion demande de
consulter `validation_report.json` uniquement lorsque ce compteur est non nul.
Une ligne sans resolution objective reste dans `failed_row_ids`.

## Metadonnees YAML

Le generateur ecrit toujours un bloc initial `pending`, non verifie. Apres une
execution reelle, le validateur remplace uniquement ce bloc a partir du rapport;
aucun statut de validation n'est redige a la main. Le statut racine est
exclusivement celui des 900 lignes core.

Le YAML conserve `status`, `verified`, `scope: "core (900 rows)"`, une reference
fusionnee avec sa methode, par exemple `Premia (specialized pricer)` ou
`QuantLib (Monte Carlo)`, puis le chemin repository relatif du notebook compile
dans `notebook`. Il ne publie aucune reference externe pour le stress. Si une
seule ligne core reste divergente ou sans reference exploitable, `verified`
reste faux; l'echec technique de Premia, a lui seul, n'est jamais presente
comme une invalidation du prix CUDA.

Le YAML reste un resume public compact. Il ne contient ni plan des moteurs, ni
historique `attempts`, ni `relationship`, ni commentaire
`stress_parameter_note`. Les diagnostics detailles, les incidents et les replis
appartiennent au rapport JSON de validation.

## Compte rendu d'execution

Chaque couple modele-produit possede un validateur unifie sous
`validation/model/<asset_class>/<model>/[<curve>/]<product>.py`. Le niveau
courbe est present seulement lorsque le pricing en depend. Ce fichier applique la
hierarchie Premia, QuantLib specialise, QuantLib Monte Carlo, puis `none`, et
ecrit une seule fois `validation_report.json` a cote du notebook du dataset.
Les adaptateurs propres aux backends restent ranges sous `validation/premia`
et `validation/quantlib`.

La commande publique d'un validateur unifie suit toujours le meme contrat:

```bash
python -m validation.model.<asset_class>.<model>[.<curve>].<product> \
  DATASET VALIDATION_REPORT
```

Elle lit le vrai dataset CUDA, execute la hierarchie, ecrit atomiquement le
rapport, synchronise le bloc `validation` du YAML puis permet l'execution du
notebook de presentation. Une commande directe sous `validation/premia/` ou
`validation/quantlib/` sert uniquement a diagnostiquer un backend; elle ne
publie ni rapport canonique ni metadonnees de catalogue.

Un nouveau couple modele-produit ne reimplemente pas cette orchestration. Son
module declare les moteurs disponibles, les conversions de lignes, les
tolerances et, si necessaire, une borne de contrat ou une explication de biais.
`validation/dataset_validation.py`, `validation/hierarchy.py` et
`validation/reporting.py` restent les uniques proprietaires respectifs de
l'execution commune, du fallback et du format de rapport.

Le rapport JSON contient deux sections aux roles explicitement differents:

- `core`: certification externe des 900 lignes, avec `passed`, `failed` ou
  `not_available`, la reference principale, son `pricing_method` exact, la
  tolerance, les ecarts, le biais, les lignes speciales, les replis,
  `engine_plan` et `engine_coverage`;
- `stress`: diagnostic interne des 100 lignes, marque
  `certification: false` et `external_reference: "none"`, avec uniquement les
  invariants controles, leurs resultats et les identifiants des violations.

Chaque repli core conserve son `pricing_method`. `engine_plan` conserve toute
la hierarchie declaree, la methode exacte des moteurs disponibles et la raison
de chaque indisponibilite. `engine_coverage` conserve, pour chaque moteur
execute sur le core, sa methode exacte et les lignes demandees, calculees,
divergentes et techniquement rejetees. Aucun appel Premia ou QuantLib stress ne
doit apparaitre dans ces couvertures.

L'empreinte SHA-256 est canonique: elle couvre les prix, les references des
datasets d'entree et la configuration numerique du YAML (`summary`, grille de
temps, sorties et construction du prix). Elle exclut les timings et le bloc de
validation, qui ne changent pas le resultat numerique.

Si aucun moteur compatible n'est declare, ou si tous les moteurs disponibles
rejettent techniquement une ligne core, le rapport la compte dans
`unvalidated`, utilise `reference: "none"` lorsqu'aucune ligne n'est couverte et
n'invente aucune statistique d'erreur. Le rendu affiche explicitement que
l'absence ou l'echec des validateurs ne prouve pas que le prix CUDA est faux.
Le YAML correspondant conserve `status: "not_available"`, `verified: false`,
`scope: "core (900 rows)"` et `reference: "none"`.

Le notebook compile ne lance aucun pricer. Le rapport et son rendu affichent la
reference core, le `pricing_method` exact juste en dessous, puis la tolerance,
sans champ `criterion`; lorsqu'un biais est observe, sa cause est donnee par
`bias explanation`. Le stress est affiche dans une section distincte intitulee
`Stress robustness diagnostics`, sans reference externe et sans vocabulaire de
certification. Il charge le rapport avec
`validation.reporting.load_validation_report`, qui refuse une empreinte
obsolete, puis appelle `display_validation_report`. Tous les notebooks
obtiennent ainsi exactement la meme presentation sans dupliquer sa mise en
forme. Une regeneration complete suit donc toujours cet ordre: generateur CUDA,
validateur unifie, rapport JSON, synchronisation YAML, puis execution du
notebook de presentation.

Le validateur doit afficher pour le core:

- le nombre de lignes demandees, calculees et echouees par backend;
- les identifiants et statuts des echecs techniques;
- les identifiants des divergences et leurs erreurs maximales;
- le backend de repli utilise pour chaque ligne concernee;
- le resultat des controles ligne par ligne et du biais agrege.

Il affiche separement les controles internes du stress, sans prix Premia ou
QuantLib. Une validation `passed` exige que chaque ligne core soit couverte et
qu'aucune divergence ne soit dissimulee par un repli. `not_available` est un
resultat technique valide du pipeline, mais ne rend pas le dataset verifie.
La conclusion d'un rapport core reussi reprend obligatoirement la phrase de la
section `Regimes`; elle ne revendique jamais une couverture externe des 1 000
lignes.

## Couverture Black-Scholes

Les adaptateurs existants permettent d'appliquer cette politique aux 29 bases
Black-Scholes hors exercice anticipe:

- Premia direct ou compose pour les European, digital, asset-or-nothing, gap,
  straddle, forward-start, geometric Asian et range accrual;
- Premia continu avec borne mathematique de difference de contrat pour les
  arithmetic Asian quotidiens;
- Premia en borne directionnelle pour les quatre barrieres simples, les deux
  double knock-outs, le lookback fixe et les deux touch products;
- QuantLib Monte Carlo seulement pour Athena, Cliquet, Phoenix et Phoenix
  Memory, car Premia n'expose aucun moteur compatible pour ces quatre couples
  Black-Scholes-produit.

Premia est donc la reference principale de 25 bases Black-Scholes sur 29.
QuantLib n'est la reference principale que des trois autocalls et du cliquet.
Il reste disponible comme repli ligne par ligne lorsqu'un calcul Premia est
techniquement inexploitable; il ne remplace jamais un prix Premia comparable
simplement parce qu'une autre methode numerique donnerait un ecart plus petit.

Les 29 familles Black-Scholes hors exercice anticipe utilisent la meme ossature
unifiee sous `validation/model/equity/black_scholes/validation.py`. Chaque produit garde un
module mince et explicite; la selection des moteurs, le fallback par ligne, les
metriques, le rapport JSON et la synchronisation YAML passent par
`validation/dataset_validation.py`, commun a toutes les classes d'actifs.

Les datasets Monte Carlo Black-Scholes publies utilisent 65 536 trajectoires
par prix. Cette puissance de deux remplace l'ancien budget de 16 384 chemins;
elle reduit l'incertitude sans modifier le mapping deterministe des chemins ni
les tolerances. Les rapports produits apres chaque regeneration sont la seule
source des eventuels incidents Premia, lignes speciales et fallbacks QuantLib:
la documentation ne conserve pas de diagnostic historique susceptible de
devenir obsolete.

## Couverture equity stochastique

Merton, Kou, Heston et Bates partagent l'ossature
`validation/model/equity/stochastic_equity.py`. Chaque module de modele ne
declare que sa couverture et les noms natifs des moteurs; chacun de ses
produits conserve un module CLI mince identique. Le plan contient toujours,
dans cet ordre, Premia specialise, QuantLib specialise et QuantLib Monte Carlo,
y compris lorsqu'un emplacement est explicitement indisponible.

La couverture Premia effectivement raccordee est la suivante:

- Merton et Kou: European, straddle, digitals et leurs replications statiques,
  Asian fixe, quatre barrieres simples et lookback fixe;
- Heston: European, straddle, Asian fixe, quatre barrieres simples et American
  continu utilise comme borne des Bermudan;
- Bates: European, straddle, Asian et barrieres par Monte Carlo Alfonsi, puis
  American par Longstaff-Schwartz Premia.

Les moteurs Premia de chemin conservent leur erreur standard lorsqu'elle est
exposee. Une formule de parite compose exactement les knock-in lorsque Premia
n'expose directement que le knock-out. Une reference continue n'est pas
comparee comme une egalite lorsque l'ordre discret/continu est mathematiquement
connu. Pour Bates, les references barrieres Premia et CUDA sont deux schemas
discrets independants: elles utilisent donc une comparaison symetrique et non
une fausse borne continue.

Heston et Bates utilisent ensuite leurs adaptateurs QuantLib communs. Les
European, digitals, replications statiques et American passent par les moteurs
analytiques ou PDE; les produits de chemin passent par un generateur de chemins
QuantLib antithetique. Merton et Kou ne declarent aucun Monte Carlo QuantLib
generique: `Merton76Process` n'expose pas une evolution de chemin exploitable et
QuantLib ne fournit pas de processus Kou. Un payoff sans moteur Premia
compatible reste donc honnetement `not_available` pour ces deux modeles.

## Couverture Gaussian short rate autonome

Vasicek, Ornstein-Uhlenbeck centre et G2 utilisent la meme declaration de
produits et la meme orchestration sous
`validation/model/fixed_income/gaussian_rate.py`. Les caplets et floorlets sont
ramenes par identite exacte a un put ou call sur zero-coupon; ce n'est pas un
rapprochement avec un cap ou floor multi-reset.

Pour Vasicek et OU, les quatre familles `caplet`, `floorlet`,
`zero_coupon_bond_call` et `zero_coupon_bond_put` utilisent les formules
specialisees Vasicek de Premia, puis QuantLib en repli. Pour G2 autonome,
Premia expose Hull-White 2D mais exige une courbe initiale calibree externe et
ne represente pas directement les deux etats initiaux du contrat catalogue.
Ce mapping n'est donc pas declare compatible: les quatre familles utilisent le
pricer specialise QuantLib G2 construit sur la courbe initiale exacte induite
par les deux facteurs. Aucun de ces douze datasets analytiques n'introduit une
grille `1/360` ou un budget Monte-Carlo artificiel.

## Couverture Gaussian short rate avec courbe ajustee

Hull-White et G2++ combines aux courbes Nelson-Siegel ou Svensson utilisent
l'orchestration commune
`validation/model/fixed_income/fitted_gaussian_rate.py`. Les seize couples
modele-courbe-produit couvrent les caplets, floorlets et options call/put sur
zero-coupon. Premia est la reference principale; le pricer specialise
QuantLib construit sur les memes noeuds de courbe reste le repli ligne par
ligne.

Premia HW1D et HW2D lisent leur courbe initiale depuis un fichier. Le runner
cree un fichier temporaire propre au processus et, pour chaque ligne, y ecrit
des noeuds encadrant exactement les deux dates contractuelles utilisees par la
formule. Les discounts sont evalues directement depuis Nelson-Siegel ou
Svensson; l'encadrement local evite que l'interpolation lineaire interne de
Premia introduise une approximation visible. Le fichier est supprime apres
chaque prix.

Le mapping Hull-White est direct. Pour G2++, le runner applique le changement
de variables exact entre les deux facteurs OU centres du catalogue et la
parametrisation HW2D de Premia avant d'appeler sa formule d'option sur bond.
Les caplets et floorlets utilisent ensuite la meme identite exacte de bond
option que les modeles autonomes. Ces pricers sont analytiques: aucune grille
`1/360` ni trajectoire Monte-Carlo n'est ajoutee artificiellement.
