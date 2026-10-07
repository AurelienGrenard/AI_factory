# Constats clos avec problème non résolu

Dernière revue : **2026-10-07**.

Ce registre contient les constats retirés du suivi opérationnel parce que leur
usage est bloqué ou qualifié, alors qu'un calcul ou un prix correct reste à
établir. Il contient actuellement **deux constats : NUM-037 et PERF-027**.
Les constats qui demandent encore une action sont dans [response.md](response.md) ;
les constats réglés par correction, contrat explicite ou décision de périmètre
sont dans [closed.md](closed.md). Dès que la cause et la correction d’un constat
présent ici sont vérifiées, **déplacer son entrée vers `closed.md` et la retirer
de ce fichier**. Conserver les preuves et la qualification historique dans
l’entrée transférée. Aucun constat ne doit figurer dans les deux registres.

## NUM-037 — Qualification des prix SABR markoviens signalés (2026-10-07)

**Clôture administrative par exclusion de références ; problème numérique encore actif.**

- **État :** fermé par décision de publication vérifiable. Les 26 paires
  européennes call/put signalées, huit core et 18 stress, soit 52 lignes,
  sont marquées `not_certified` pour servir de références de prix ou
  d'erreur standard dans la
  [qualification versionnée](../../work/catalog/qualifications/num037-sabr-parity-20261007-v1/manifest.json).
  Les prix et les `generation.yaml` originaux restent inchangés ; les 974
  autres paires ne sont pas certifiées par cette décision.
- **Replays :** le lanceur publié reproduit bit à bit les 26 prix et erreurs
  standards de chaque côté à 2²⁰ chemins avec les graines call/put distinctes.
  Les 26 lignes ont été rejouées à 65 536 et 262 144 chemins avec trois
  graines (156 replays), puis contrôlées trajectoire par trajectoire : premier
  moment actualisé, absorption, quantiles, maxima, queues et erreurs MC.
  Aucun chemin non fini n'a été observé. Un schéma Euler indépendant sur le
  spot a rejoué les 26 lignes avec trois graines à 262 144 chemins.
- **Diagnostic :** les six résidus positifs concernent β entre 0,020 et 0,062 ;
  cinq ont `ρ<0` et un `ρ>0` (ligne 995, `ρ=0,0189`).
  Le raffinement Lamperti réduit le résidu moyen des lignes 945 et 926 de
  +0,028284 à +0,007108 et de +0,068203 à +0,019018 entre 1 et 32 pas/jour.
  Les vingt résidus négatifs ont ρ positif ; seize conservent leur signe dans
  les trois replays Euler du spot. À la ligne 958, le résidu reste proche de
  −0,204 jusqu'à 16 pas/jour et vaut −0,203327 ± 0,000704 à 2²⁰ chemins.
  À la ligne 992, les 100 plus grands spots actualisés portent 13,86 % de
  leur somme sur 2²⁰ chemins. Les SE empiriques ne certifient donc pas la
  capture des événements plus rares. Les grilles 1/2/4/8 pas/jour de cinq
  lignes, et 16/32 de trois lignes, sont archivées.
- **Cause et limite :** les diagnostics appuient un biais de grille près de
  l'absorption pour les résidus positifs et une queue positive sous-échantillonnée
  pour les résidus négatifs. Ils établissent une limite de l'estimateur sur
  ces paramètres, pas un prix corrigé pour chaque ligne. Le SABR continu
  avec β < 1 et absorption en zéro est une vraie martingale, y compris pour
  `ρ>0` : les 1 000 paramètres publiés ont `0,01471≤β≤0,99864`.
  Le [résultat de Jourdain](https://citeseerx.ist.psu.edu/document?doi=7eb0367e88b4b83734eb45cf1311fb783523f42c&repid=rep1&type=pdf)
  distingue ce cas du SABR lognormal `β=1` à `ρ>0` ; les déficits négatifs ne
  sont donc pas acceptés comme propriété du modèle continu. La même distinction
  est vérifiée pour le rough SABR publié dans
  [l’analyse du domaine martingale](sabr-martingale-domain.md). Aucun prix extrême
  n'a été écrêté. Une nouvelle
  publication de prix certifiés exigerait une estimation du premier moment et
  des queues indépendante et suffisamment précise.
- **Provenance :** [analyse et reçus](../../artifacts/audit/num037-2026-10-06/README.md)
  conservent sources, hashes des binaires, résultats bruts et décision. Le
  vérificateur de qualification passe : 26 masques, 52 lignes, prix et reçus
  inchangés, hashes d'artefacts concordants.

## PERF-027 — Hessienne CIR–Jamshidian retirée du périmètre (2026-10-06)

**Clôture administrative par retrait de capacité ; Hessienne non corrigée.**

- **État :** fermé par décision explicite de limiter le binding analytique
  `cir/european_swaption` au prix et aux dérivées premières. La diagonale et
  les termes mixtes sont refusés par la préparation et les launchers. Le
  manifeste, le codegen, CMake et les quatre recettes payer/receiver alignées
  et cartésiennes portent uniquement l'ordre un. Le cas diagonal est retiré de
  la campagne de performance ; ses anciennes traces restent conservées comme
  preuve de la limite, sans capacité publiée.
- **Cause :** les prix de nœuds FP32 sont assez précis pour le prix, mais leur
  erreur est amplifiée par la seconde différence. Sur la ligne stress 4 à
  strike nul, la vraie Hessienne strike est zéro, contre −6,677082 en scalaire
  et −0,000919 en coopératif. Les poids FP32 créent déjà −0,002581 sur une
  fonction affine ; les poids FP64 donnent environ −1,71e−10. Sur la ligne
  core 154, la vraie Hessienne strike vaut 110,108641. Élargir le pas jusqu'à
  faire coïncider les kernels donne 78,584821, soit 28,6 % de biais. La
  référence CIR T-forward et la décomposition QuantLib concordent sur le prix
  à 1,52e−13 près. [Analyse et traces](../../artifacts/audit/perf027-2026-10-06/README.md).
- **Vérification du nouveau contrat :** compilation dans un build isolé,
  test CUDA du rejet des ordres 2 et de la parité des gradients, codegen
  complet avec comparaison nulle, tests de manifeste, benchmark ordre 1 sur
  1 000 lignes core et huit stress. Les ratios maximaux de tolérance
  scalaire/coopératif des gradients valent 0,03196 et 0,01589 ; le benchmark
  ordre 1 n'émet plus de champs Hessienne. Sur les lignes payer 0, 154 et 999,
  les prix et les gradients strike/volatilité ont été confrontés à la
  quadrature FP64 sur les nœuds représentés.
- **Génération :** quatre nouveaux générateurs ont été compilés et exécutés
  dans une campagne Ninja figée : deux fois 1 000 lignes alignées et deux fois
  1 000 000 de lignes cartésiennes. La reprise vérifie les quatre paires
  JSON/`generation.yaml` et leurs empreintes. Elles restent **en staging**
  dans `/tmp/ai_factory_perf027_first_catalog_staged_20261006` : le
  contrôleur refuse `--publish` tant que le worktree Git est modifié.
  Aucun `generation.yaml` n'a été copié au catalogue. Les deux entrées
  manquantes ont été restaurées depuis la campagne
  `work/generation/fixed-income-aligned-01` avec les mêmes SHA-256 que son
  manifeste. [Campagne ordre un](../../artifacts/audit/perf027-2026-10-06/first_order_scope/result.json).
- **Portée :** cette clôture retire la promesse de Hessienne pour CIR
  standalone/Jamshidian. Elle ne certifie pas les Hessiennes des autres
  bindings fixed income, ni la performance du preset de production sur tout
  matériel. Toute réintroduction de l'ordre 2 nécessitera une nouvelle
  qualification indépendante core/stress sur tous les axes.
