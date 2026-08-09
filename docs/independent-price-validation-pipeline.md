# Pipeline de validation independante des prix

Ce document definit la selection du backend de reference, le traitement des
echecs et les metadonnees obligatoires des bases de prix. La selection se fait
par regime puis, lorsque le backend le permet, par ligne. Elle ne doit jamais
etre deduite de la seule presence d'une methode dans une bibliotheque.

## Regimes

Les bases ordonnees selon la convention 90/10 sont validees separement:

- `core`: les 900 premieres lignes, representant le domaine usuel;
- `stress`: les 100 dernieres lignes, volontairement plus larges.

Un backend peut donc couvrir le core sans couvrir tout le stress. Les lignes
stress ne sont pas supprimees pour faire passer un test. Une plage peut etre
resserree seulement si le contrat du dataset est financierement peu utile ou
si le domaine documente du pricer l'exclut; la raison doit alors apparaitre
dans la recette de generation.

## Hierarchie par ligne

Pour chaque ligne, appliquer dans cet ordre:

1. pricer specialise Premia compatible;
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

La premiere reference qui produit une validation reussie est conservee. Si
Premia valide 97 lignes stress et echoue techniquement sur 3, QuantLib ne traite
que ces 3 lignes. La couverture finale peut donc etre mixte sans affaiblir la
priorite donnee a Premia.

Le routage est implemente une seule fois dans `validation/hierarchy.py`. Chaque
fichier modele-produit declare les trois moteurs de la hierarchie; `none` reste
le resultat final implicite. Un
moteur compatible fournit son adaptateur; un moteur indisponible fournit une
raison courte et n'est pas execute. Le rapport conserve ainsi le plan complet,
y compris les possibilites examinees mais indisponibles. Seules les exceptions
techniques descendent au moteur suivant. Une comparaison calculee mais hors
tolerance reste un echec du moteur courant.

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
statut d'erreur, valeur non finie, erreur standard invalide, prix violant une
borne de non-arbitrage, contrat hors domaine documente ou methode indisponible.
L'identifiant, le statut et la raison sont conserves. Le pipeline peut alors
essayer le backend suivant pour cette seule ligne.

Une divergence signifie que le backend a produit un prix fini avec succes,
mais que notre prix ne respecte pas la tolerance, la borne ou le controle de
biais. Ce n'est pas une indisponibilite du backend. La ligne echoue et ne doit
pas etre remplacee silencieusement par une reference moins prioritaire. Il faut
examiner le pricing, le budget Monte Carlo, les conventions et la pertinence du
cas stress.

Les tolerances combinent erreur absolue, erreur relative et erreurs standards
independantes. Elles ne sont pas elargies pour effacer une divergence. Les
validations Monte Carlo controlent egalement le biais signe du dataset.
Strictement plus de 60% d'ecarts dans le meme sens declenche une alarme. Pour
les deux produits touch Black-Scholes, cette alarme provoque automatiquement
une confirmation du regime complet avec 4 096 paires antithetiques QuantLib au
lieu de 1 024; seul le controle renforce est conserve dans le rapport. Si le
biais persiste, la validation echoue.

## Metadonnees YAML

Le generateur ecrit toujours un bloc initial `pending`, non verifie. Apres une
execution reelle, le validateur remplace uniquement ce bloc a partir du rapport;
aucun statut de validation n'est redige a la main.
Le YAML conserve `status`, `verified`, une reference fusionnee avec sa methode,
par exemple `Premia (specialized pricer)` ou `QuantLib (Monte Carlo)`, puis le
chemin repository relatif du notebook compile dans `notebook`. Viennent ensuite
le statut, le nombre de lignes et la reference de `core` et `stress`.

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

Le rapport JSON contient deux sections de meme schema, `core` et `stress`.
Chacune stocke l'un des trois statuts `passed`, `failed` ou `not_available`, la
reference principale avec sa methode entre parentheses, la tolerance, la
couverture des lignes ordinaires, speciales et echouees, les
ecarts de prix signe moyen, absolu moyen et absolu maximal, les comptes
`higher / lower / equal`, le diagnostic de biais, les lignes speciales et les
replis. `engine_plan` conserve toute la hierarchie declaree et la raison de
chaque indisponibilite. `engine_coverage` conserve, pour chaque moteur execute,
les lignes demandees, calculees, divergentes et techniquement rejetees.

L'empreinte SHA-256 est canonique: elle couvre les prix, les references des
datasets d'entree et la configuration numerique du YAML (`summary`, grille de
temps, sorties et construction du prix). Elle exclut les timings et le bloc de
validation, qui ne changent pas le resultat numerique.

Si aucun moteur compatible n'est declare, ou si tous les moteurs disponibles
rejettent techniquement les lignes, le rapport les compte dans `unvalidated`,
utilise `reference: "none"` et n'invente aucune statistique d'erreur. Le rendu
affiche explicitement qu'aucune validation independante Premia ou QuantLib
n'est disponible. Le YAML correspondant conserve `status: "not_available"`,
`verified: false` et `reference: "none"`.

Le notebook compile ne lance aucun pricer. Le rapport et son rendu affichent la
reference et la tolerance, sans champ `criterion`; lorsqu'un biais est observe,
sa cause est donnee par `bias explanation`. Il charge le rapport avec
`validation.reporting.load_validation_report`, qui refuse une empreinte
obsolete, puis appelle `display_validation_report`. Tous les notebooks
obtiennent ainsi exactement la meme presentation sans dupliquer sa mise en
forme. Une regeneration complete suit donc toujours cet ordre: generateur CUDA,
validateur unifie, rapport JSON, synchronisation YAML, puis execution du
notebook de presentation.

Le validateur doit afficher, pour chaque regime:

- le nombre de lignes demandees, calculees et echouees par backend;
- les identifiants et statuts des echecs techniques;
- les identifiants des divergences et leurs erreurs maximales;
- le backend de repli utilise pour chaque ligne concernee;
- le resultat des controles ligne par ligne et du biais agrege.

Une validation `passed` exige que chaque ligne soit couverte et qu'aucune
divergence ne soit dissimulee par un repli. `not_available` est un resultat
technique valide du pipeline, mais ne rend pas le dataset verifie.

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
