# Politique de qualification des bumps de prix-gradients

## Objet et statut

La politique exécutable
`tests/price_gradients/qualification_policy_v2.json` définit avant exécution
les cas, bumps candidats, références et critères de la première campagne
commune aux calls européens Black--Scholes, Heston et CEV. Son statut initial
est `candidate` : la présence du fichier et une exécution réussie ne certifient
aucun dataset.

La campagne conserve trois niveaux distincts :

1. la convergence de la référence indépendante ;
2. le biais du stencil par rapport à une dérivée de référence plus fine ;
3. le résidu du moteur, séparé de son erreur standard Monte Carlo.

La stabilité de plusieurs bumps n'est jamais un critère suffisant. Un candidat
échoue s'il est stable autour d'une valeur biaisée, si sa référence ne converge
pas, ou si un cas requis échoue. Tous les cas restent dans le rapport, y compris
les échecs et les résultats non évaluables.

## Domaine versionné

La version 2 couvre seulement le call européen :

- Black--Scholes, formule fermée et Monte Carlo terminal exact ;
- Heston QE-M sur 504 et 1 008 pas par an ;
- CEV Milstein absorbant sur 504 et 1 008 pas par an.

Les cas couvrent régime ordinaire, stress, frontière, maturité d'un jour,
maturité longue et changement d'échelle spot/strike. Les tailles de bump sont
propres à chaque coordonnée ; les facteurs communs ne changent pas leur unité.
Le taux et le dividende sont absolus, notamment afin que zéro reste admissible.
La maturité reste un nombre entier de pas côté moteur.

Les puts, produits de chemin, exercice anticipé, modèles à sauts et modèles
rough sont hors de ce domaine. Ils doivent étendre la politique au lieu de
réutiliser silencieusement son verdict.

## Références et cache

Black--Scholes utilise les prix et dérivées analytiques FP64 côté hôte. Heston
utilise le solveur Riccati/Fourier indépendant déjà présent, avec les deux
résolutions pré-déclarées. CEV utilise la formule analytique CEV absorbée en
temps continu, exprimée par deux lois du chi-deux non centrales après la
transformation exacte qui retire le carry. Elle conserve exactement les
maturités fractionnaires ; sa valeur est croisée avec `AnalyticCEVEngine` de
QuantLib uniquement aux dates représentables sur la grille business-day/252.
Les références analytiques BS/CEV sont marquées comme telles et ne prétendent
plus démontrer une convergence par deux évaluations identiques.

Ces moteurs ne sont appelés que par la commande explicite de qualification.
Le répertoire de preuve conserve les sorties natives, références, rapport,
politique et empreintes SHA-256. Les tests ordinaires emploient seulement des
fixtures synthétiques et ne régénèrent aucune référence externe.

## Décision

Pour chaque modèle, coordonnée, cas, grille et bump, le rapport agrège les
trois seeds avec `sqrt(sum(SE_i^2)) / 3`. Il applique simultanément :

- les seuils de convergence de référence ;
- six erreurs standards sur le résidu MC ;
- une borne absolue plus relative sur l'erreur totale ;
- une borne absolue plus relative sur le biais du stencil ;
- pour Black--Scholes fermé, une borne séparée sur l'écart FP32.

Le validateur reconstruit le produit cartésien exact modèle, grille, facteur,
seed, cas et coordonnée. Toute observation manquante, supplémentaire ou
dupliquée est fatale ; chemins, threads, largeur de lot et pas de temps doivent
également correspondre à la politique.

Un bump n'entre dans le domaine qualifié que si tous les cas et raffinements
applicables passent. Le rapport ne choisit pas le meilleur résultat après
observation : l'ordre de préférence et les critères viennent de la politique.
Un dataset appartenant au domaine doit encore rejouer des lignes core, stress
et frontières ciblées. Un domaine nouveau exige une nouvelle version de la
politique et de nouvelles preuves.

## Campagne v1 conservée mais remplacée

La campagne v1 SM89 du 16 septembre 2026 a exécuté sa charge déclarée complète.
Sur 1 876 observations agrégées, 1 320 passent et 556 échouent. Treize des 23
coordonnées possèdent au moins un facteur candidat sur toute la surface : spot,
taux, dividende, volatilité et strike Black--Scholes ; spot, taux, dividende,
variance initiale et strike Heston ; spot, sigma et strike CEV.

Restent non qualifiés : maturité Black--Scholes ; kappa, theta, gamma, rho et
maturité Heston ; taux, dividende, beta et maturité CEV. Les échecs sont dominés
selon les cas par le bruit MC, l'erreur totale ou le biais du stencil ; quatre
cas de maturité Heston échouent aussi la convergence de référence. Aucun bump
par défaut n'est modifié par ce constat. Le détail de chaque vérification et la
chaîne SHA-256 sont conservés dans
[`artifacts/price_gradients/2026-09-16-bump-qualification-merton/`](../../artifacts/price_gradients/2026-09-16-bump-qualification-merton/README.md).

Son verdict CEV maturité est invalide : l'adaptateur QuantLib arrondissait les
petites perturbations à une même date. La v1 reste une preuve historique et
n'est plus une source de candidats acceptables. La v2 corrige la référence,
rend l'inventaire strict et conserve le budget relatif global de 0,5 %. L'écart
d'environ 3,5 % observé antérieurement sur un exemple BS à un jour reste une
limite documentée, pas une tolérance de qualification implicite.

## Résultat correctif v2

La v2 conserve 1 876 observations agrégées : 1 355 passent et 521 échouent.
Treize coordonnées sur 23 gardent au moins un facteur candidat ; la liste des
coordonnées qualifiées candidates est inchangée. Les vérifications qui échouent
se recouvrent : 431 erreurs totales, 226 résidus au-delà de six SE, 68 biais de
stencil, 12 écarts FP32 fermés et quatre convergences de référence Heston.

Le diagnostic apparié des deux grilles trouve 122 déplacements coarse/fine
significatifs sous une borne prudente de six SE ; la grille fine se rapproche
de la dérivée indépendante dans 25 cas. Ce signal distingue un biais temporel
possible du seul bruit MC, sans prétendre l'identifier causalement dans tous
les cas. Les preuves, échecs et empreintes sont dans
[`artifacts/price_gradients/2026-09-16-bump-qualification-v2/`](../../artifacts/price_gradients/2026-09-16-bump-qualification-v2/README.md).

## Reproduction

```bash
cmake --build build --target study_price_gradients_bumps -j2
./build/study_price_gradients_bumps 262144 \
  > artifacts/price_gradients/<run>/native.jsonl
python3 tests/price_gradients/analyze_bumps.py \
  artifacts/price_gradients/<run>/native.jsonl \
  artifacts/price_gradients/<run>/analysis.json \
  --policy tests/price_gradients/qualification_policy_v2.json
python3 tests/price_gradients/qualify_bumps.py \
  tests/price_gradients/qualification_policy_v2.json \
  artifacts/price_gradients/<run>/analysis.json \
  artifacts/price_gradients/<run>/qualification.json
```

Le nombre de trajectoires peut être abaissé pour un smoke test, mais le rapport
le marque inéligible si la charge diffère de la politique. Une compilation ou
un smoke test ne remplace pas la campagne complète.
