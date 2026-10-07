# Domaine martingale des références de prix

Décision appliquée le **2026-10-07** aux modèles pour lesquels le domaine a
été établi. Le seuil est le seuil mathématique du modèle continu : **`ρ≤0`**
pour les deux Bergomi lognormaux en spot. Une limite `ρ≤0,05` laisserait des
martingales locales strictes dans le jeu de références ; `ρ<−0,1` retirerait
sans nécessité les contrôles `ρ=0`. Les valeurs hors domaine restent dans les
JSON historiques en tant que stress, avec un masque de référence vérifiable.

| Modèle | Lignes historiques `ρ>0` | Décision pour les références historiques de prix/SE |
|---|---:|---|
| rough Bergomi | 15 stress | 435 lignes exclues dans 29 jeux par la [qualification](../../work/catalog/qualifications/rough-bergomi-martingale-domain-20261007-v1/manifest.json) |
| log-modulated rough Bergomi | 42 stress | 1 218 lignes exclues pour le domaine, plus une ligne à queue instable (`ρ<0`) par [NUM-030](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json) |
| SABR markovien | 205, toutes `β<1` | pas d'exclusion **au titre du domaine martingale** ; 52 prix demeurent non certifiés pour une raison numérique [NUM-037](unresolved-closures.md) |
| rough SABR | 13, toutes `β<1` | pas d'exclusion au titre du domaine martingale ; la release NUM-029 a zéro signal de parité au seuil publié |

Le [théorème de Gassiat](https://arxiv.org/html/1811.10935v3) s'applique
directement au rough Bergomi standard. Son application au noyau log modulé
est détaillée dans [NUM-030](num030-martingale-domain.md). La distinction
`β<1` / `β=1` pour SABR est démontrée dans
[l'analyse SABR](sabr-martingale-domain.md). Les masques qualifient un **usage
des prix** : ils ne changent ni les paramètres, ni les prix historiques, ni
leurs `generation.yaml` authentiques, et ils ne certifient pas les prix non
masqués.

Le [gate des datasets](../../maintainer/tools/datasets/check_martingale_reference_domains.py)
contrôle que les quatre jeux de paramètres **actuels** n'ont aucune corrélation
positive. Pour les 58 anciens jeux de prix Bergomi, il recalcule séparément
les masques à partir des [entrées historiques figées](../../work/catalog/archives/parameter-catalog-20261007/manifest.json),
puis vérifie prix, reçus, hashes et index des qualifications. Le contrôle
NUM-029 relit également les anciens paramètres SABR et le produit européen
figés pour retrouver la parité publiée. Le contrôle NUM-030 et son quadrature
indépendante ont été rejoués sur leurs anciennes entrées : les huit cas
numériques sont identiques. Les anciens prix ne qualifient pas les nouveaux
paramètres. Le gate est enregistré dans CTest lorsque les JSON de production
sont présents. Exécution directe :

```bash
python3 maintainer/tools/datasets/verify_rough_bergomi_domain_qualification.py
python3 maintainer/tools/datasets/verify_num030_qualification.py
python3 maintainer/tools/datasets/check_martingale_reference_domains.py
```

Cette vérification de domaine ne résout pas la qualité des queues Monte Carlo.
`NUM-028`, `NUM-030` et `NUM-037` gardent leurs décisions et preuves propres.
Les `rho` de G2/G2++ corrèlent deux facteurs de taux ; le quadratic rough
Heston utilise le même brownien pour le spot et son facteur sans paramètre
`rho`. Les autres modèles de volatilité non analysés ici ne reçoivent aucune
certification de martingale par ce gate. Le nouveau catalogue impose néanmoins
`rho≤0` comme politique de données d'entraînement pour tous les modèles
corrélés, sans prétendre que ce signe constitue un théorème de martingale
commun à toutes les familles.
