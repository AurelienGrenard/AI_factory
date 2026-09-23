# Plan de migration des gradients de prix des modèles à sauts

**État au 23 septembre 2026 :** première tranche terminale intégrée, qualification
globale encore ouverte dans [NUM-032](../audit/response.md#num-032--ouvrir-les-sensibilités-de-saut-avec-des-marques-centrales-rejouables).
Merton, Kou et Bates disposent du central événementiel, des sensibilités de
marque, d'intensité et de maturité européenne aux ordres un et deux diagonal.
Variance-Gamma et NIG utilisent une reparamétrisation couplée propre à leur
subordonnateur d'activité infinie. Produits de chemin, exercice gelé, campagne
de bumps, sanitizers et qualification de performance restent ouverts. Ce plan
complète le [plan d'adressage Philox](philox-domain-migration-plan.md).

Le [plan global prix-gradients](price-gradients-migration-plan.md) place cette
extension avant la porte de qualification markovienne ; le présent document
possède le couplage des événements, les marques et leur validation.

## Objectif et périmètre

Rendre sélectionnables l'intensité, les paramètres de loi des marques et, pour
les produits européens admissibles, la maturité, tout en conservant un prix
central calculé une seule fois et des innovations couplées entre scénarios.
La première qualification porte sur les processus de sauts **à activité finie**
du dépôt : Merton et Bates (marques gaussiennes) puis Kou (marques double
exponentielles). Une interface de modèle doit permettre d'ajouter une autre loi
de marques sans modifier le moteur générique. Un processus à activité infinie
ou à intensité dépendante de l'état demande un couplage spécifique déclaré par
son modèle ; le schéma Poisson homogène ci-dessous ne lui est pas attribué par
défaut.

La maturité d'un produit américain/bermudéen reste hors de cette première
extension, conformément au contrat actuel. L'exercice gelé peut être étudié
ensuite pour les autres coordonnées de saut ; il ne qualifie pas à lui seul le
biais par rapport à une stratégie réentraînée.

## Responsabilités à préserver

- **`src/common/philox.cuh` et son éventuel compagnon d'adressage :** une
  primitive Philox, une clé par ligne, des adresses de chemin, pas, type de
  tirage et événement sans collision. Une source fixe peut garder le flux
  unique ; les consommations variables sont isolées seulement si le CRN le
  demande. Aucun modèle n'est nommé dans cette couche.
- **`src/common/monte_carlo/price_gradients/` :** la distribution des nœuds
  préparés, la boucle des chemins, le prix central unique, les moments et la
  réduction restent communs. Le contrat actuel `Dynamics::draw` produit une
  innovation unique réutilisée par tous les nœuds ; ajouter un chemin de
  composition à la compilation pour les innovations **couplées mais propres à
  chaque nœud**. Garder les transitions déterministes et éviter un tableau de
  marques par chemin ou un dispatch par nom dans le kernel chaud. Mesurer si
  le calcul couplé de plusieurs nœuds augmente trop les registres pour `B=1`.
- **`src/model/equity/markovian/<model>/` :** chaque modèle possède le tirage
  du compte central, le couplage des comptes admissibles, la transformation
  d'une innovation en marque, son compensateur de martingale et l'application
  des sauts à son état. Les adapters `price_gradients/` exposent ces opérations
  au moteur commun ; aucun modèle ne recopie le payoff, le stencil ou la
  réduction MC. Une loi de marques non gaussienne ne reçoit pas un pont
  gaussien par défaut.
- **Préparation des sensibilités :** les `SensitivitySpec` existants restent
  l'entrée de sélection. La policy de chaque modèle ajoute ses coordonnées
  de saut et leurs domaines admissibles ; le builder existant prépare les
  nœuds et le stencil, avec règle unilatérale explicite à `lambda=0` ou aux
  frontières des lois de marques. Chaque nœud recalcule le compensateur et la
  préparation temporelle correspondant à ses paramètres.
- **`catalog`, codegen et provenance :** les prix seuls et les gradients de
  la nouvelle version utilisent le même central événementiel et déclarent
  leur mapping RNG, leur politique de bump, leur source de seed et leurs
  versions. Les samples qui réutilisent une dynamique modifiée déclarent aussi
  la nouvelle version. Les recettes encore actives gardent l'ancien tirage
  jusqu'à leur migration ; les datasets historiques gardent leur recette et
  leurs sources gelées comme provenance. Aucune égalité bit à bit n'est
  attendue entre les deux versions.

## Construction des trajectoires de sauts

1. **Central événementiel.** Tirer le compte central, puis l'innovation de
   **chaque marque** dans un ordre stable ; transformer et sommer à la volée.
   Le prix seul et le premier bloc de gradients exécutent exactement le même
   calcul central. Les autres blocs peuvent rejouer ses innovations pour leurs
   sensibilités, sans recalculer son payoff ni stocker les marques en VRAM.
   Merton/Bates abandonnent leur somme gaussienne en un tirage ; Kou fournit le
   précédent événementiel, mais ses marques doivent recevoir un identifiant
   stable indépendamment de la consommation Poisson précédente.
2. **Intensité Poisson homogène.** À intensité centrale `lambda0>0`, garder
   `N0` comme ancre. Pour `lambda-<lambda0`, partager les événements centraux
   par thinning de probabilité `lambda-/lambda0`. Pour `lambda+>lambda0`,
   ajouter un compte indépendant de moyenne `(lambda+-lambda0)*T`. Les
   événements centraux retenus gardent leur identifiant et leur innovation de
   marque. À `lambda0=0`, utiliser le stencil unilatéral admis et les seuls
   nouveaux événements ; aucune division par zéro. Construire **tous les
   nœuds d'un même stencil sur une seule famille emboîtée** : un uniforme
   stable par événement central décide sa présence à chaque intensité
   inférieure, et les intervalles d'intensité au-dessus du central apportent
   des événements additionnels partagés. Ainsi les points `lambda0+h` et
   `lambda0+2h` d'un stencil unilatéral ne reçoivent pas deux comptes
   indépendants. Les identifiants des nouveaux événements sont distincts de
   ceux du central et stables sous rejeu.
3. **Maturité européenne.** Garder le compte et le prix à `T0`. À `T-`, les
   événements centraux surviennent avant cette date avec probabilité
   `T-/T0` sous Poisson homogène ; à `T+`, ajouter le compte de l'intervalle
   `(T0,T+]`. Affecter à chaque événement central une même date relative
   latente, ou un uniforme équivalent, pour que toutes les dates bumpées
   sélectionnent des sous-ensembles emboîtés. Les intervalles supplémentaires
   au-delà de `T0` sont eux aussi partagés entre les nœuds d'un stencil
   unilatéral. Coupler le brownien sur **l'ensemble** de ces dates suivant le
   contrat temporel du modèle.
   Pour un payoff terminal, le compte par intervalle et l'ordre des marques
   suffisent. Un payoff qui observe l'intérieur d'un intervalle a besoin de
   dates d'événement explicites ; ne pas lui appliquer silencieusement le
   seul couplage terminal.
4. **Paramètres de marque.** Rejouer la même innovation primitive par
   événement, puis appliquer la loi de marque de chaque nœud : normale
   standard pour Merton/Bates, uniformes de signe et de magnitude pour Kou.
   Les valeurs réalisées peuvent différer quand la loi est bumpée ; c'est
   l'innovation sous-jacente qui est commune. Conserver le même événement
   central lorsque seule sa marque ou le compensateur change. Une valeur
   dégénérée du paramètre de marque ne doit pas décaler l'adresse des
   événements suivants : les primitives de cet événement restent réservées
   ou accessibles directement par son identifiant, même si sa transformation
   n'a pas besoin de toutes leurs valeurs.
5. **Extension de modèle.** Exiger une déclaration explicite de la loi du
   compte, du mécanisme de couplage, du budget de tirages par événement et des
   paramètres supportés. Si l'intensité est stochastique ou si la marque
   dépend de l'état/du temps, l'adapter construit son propre processus dominé
   ou son horloge intégrée et prouve ses marges ; le moteur commun ne suppose
   jamais `Poisson(lambda*T)` pour tous les modèles à sauts.

## Étapes de réalisation et portes de sortie

1. **Inventaire et baseline.** Répertorier prix, samples, gradients, produits,
   intervalles agrégés, ordres de tirage et contrats bit à bit pour Merton,
   Bates et Kou ; figer prix, temps complets, nombres de registres/spills et
   lois de contrôle sur faibles et fortes intensités. Comparer les chemins
   exacts et à pas ; ne pas assimiler les samples aux prix.
2. **Pilote Merton européen.** Introduire central événementiel et replay des
   marques, puis les paramètres de marque déjà exposés, l'intensité et `T`
   européen. Tester d'abord prix central vs gradients dans la nouvelle
   version, puis la qualité des différences finies. Le prix central ancien
   reste la référence de loi et de coût, pas d'égalité bit à bit interversion.
3. **Composition commune.** À partir du pilote, extraire uniquement les
   primitives réellement partagées : adressage Philox et couplage Poisson
   homogène. Stabiliser le hook compilé permettant au modèle de fournir des
   innovations par nœud ; conserver le moteur prix/gradients et les stencils
   existants. Comparer le central seul, le stencil d'ordre un à trois nœuds
   et le stencil diagonal pouvant atteindre quatre nœuds ; inclure le cas
   unilatéral et la stratégie de blocs avant de fixer un profil.
4. **Bates puis Kou.** Bates réemploie les marques gaussiennes événementielles
   sans recopier Heston QE ni les kernels de prix ; qualifier ses intervalles
   de sauts agrégés et ses dates d'observation. Kou garde ses transformations
   double exponentielles événementielles et acquiert l'adressage/replay, les
   coordonnées de marque et d'intensité. Vérifier séparément chaque
   compensateur et les frontières des paramètres.
5. **Produits et publication.** Étendre d'abord les payoffs terminaux, puis
   les produits à observation de chemin compatibles. Examiner à part les
   barrières continues, les dates de saut et le replay d'exercice gelé ; la
   maturité américaine reste exclue. Ajouter codegen, recettes, métadonnées
   et reprise de génération seulement pour les capacités qualifiées. Une
   nouvelle version de dataset est obligatoire dès que le central change.
6. **Extension ultérieure.** Pour tout nouveau modèle à sauts, fournir son
   adapter de compte/événements/marques, ses coordonnées et ses preuves de loi,
   de CRN et de coût. Les modèles hors activité finie ou sans couplage qualifié
   restent explicitement non disponibles pour ces sensibilités.

## Tests bloquants

- **RNG et central :** adresses uniques aux frontières, replay bit à bit des
  innovations des marques par événement, central prix seuls = central
  prix-gradients dans la nouvelle version, indépendamment du nombre/ordre de
  sensibilités, du batch, de la taille de bloc et d'une reprise. Les anciennes
  recettes restent bit à bit inchangées sous leur ancien mapping.
- **Lois et couplage :** cas zéro saut, saut unique, rejets PTRS, forte
  intensité, intensité nulle, fenêtres temporelles courtes ; comptes marginaux
  Poisson, comptes emboîtés sur **tous les points du stencil**, marques communes,
  moments/queues des lois de marques, compensateur et premier moment
  actualisé. Pour une loi dépendante de l'état, référence spécifique du modèle.
- **Sensibilités :** comparaison aux dérivées ou prix indépendants disponibles
  (notamment la série de Poisson Merton), plusieurs seeds, convergence en
  chemins et balayage des bumps. Rapporter séparément bruit de différence
  couplée, biais du stencil, erreur FP32 et écart de prix ; ne pas certifier
  `lambda` sur un seul choix de bump ou une seule seed. Tester sélection
  partielle, coordonnées de marques et frontières unilatérales. Si le bump
  couplé reste trop bruité aux budgets de chemins prévus, étudier alors un
  estimateur conditionnel ou par score propre au modèle avant de déclarer la
  coordonnée disponible ; ne pas masquer le bruit par une tolérance élargie.
- **CUDA et performance :** tests hôte/codegen et CUDA ciblés, Compute
  Sanitizer, diagnostics registres/spills/shared/local/SASS ; timings kernel,
  appel public, pipeline et génération complète suivant le
  [protocole](../performance-regression-protocol.md). Mesurer `B=1` et les
  largeurs utiles, 128/256 threads, taux de divergence des boucles de marques,
  faibles et fortes intensités, produits terminaux et de chemin. Tout surcoût
  de la voie événementielle face à la somme O(1) Merton/Bates reste explicite
  et bloque une bascule non qualifiée.

NUM-031 ne se ferme que sur l'adressage et l'isolation RNG réellement livrés ;
NUM-032 suit les lois, les sensibilités et les consommateurs produits. Leurs
preuves peuvent être produites dans la même campagne sans fusionner leurs
critères de clôture.

## Avancement de l'implémentation terminale

- `common/compound_poisson.cuh` possède le tirage de compte canonique et les
  domaines de source nommés. `coupled_poisson_events.cuh` rejoue les marques
  centrales, amincit les intensités inférieures et construit une unique suite
  d'arrivées exponentielles au-dessus du central. Un nœud conserve donc les
  mêmes événements si un autre point est ajouté au stencil.
- `coupled_brownian_endpoints.cuh` fournit les extrémités browniennes
  conditionnelles des transitions directes Merton/Kou. Bates conserve la
  boucle QE canonique et agrège seulement son intervalle de sauts grâce au
  hook terminal commun ; payoff et réduction restent dans le moteur partagé.
- Merton, Kou et Bates réservent une innovation primitive par marque et ne
  matérialisent aucun tableau de marques en VRAM. Le central des gradients est
  bit à bit celui du pricer et la sélection `{model.spot}` reste bit à bit
  celle de `price_delta` sous la même configuration.
- Variance-Gamma redémarre la même source Gamma pour chaque paramétrisation du
  clock, ce qui isole les nombres de rejets, puis partage la normale brownienne.
  NIG rejoue le normal, l'uniforme de sélection et la normale brownienne de la
  construction Michael--Schucany--Haas. Ces couplages donnent les lois
  marginales exactes et un CRN projectif ; ils ne sont pas présentés comme un
  pont de sous-ordonnateur entre horizons.
