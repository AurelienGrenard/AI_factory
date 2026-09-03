# Constats ouverts de l'audit de validation

## Objet

Ce document contient uniquement les constats non resolus produits par
`query.md`. Leur revision, couverture, exclusions et preuves sont consignees
dans `status.md`. Cet audit est separe de
l'audit d'architecture principal et n'est rafraichi que lorsqu'il est demande
explicitement.

## Validation and reproducibility

### VALID-001 — Restaurer une chaine de validation publiable et fail-closed

**Etat : ouvert.**

**Coordination : a traiter apres la conception de `VALID-002`.** Toute
regeneration massive doit utiliser le manifeste de provenance final afin
d'eviter de produire puis d'invalider une seconde fois les 71 caches.

**Severite : haute · Priorite : haute · Confiance : prouvee.**

**Preuve.** Les 71 tests `cached_reference` echouent tous : empreintes de
sources ou de policy obsoletes et, pour certains caches, contradiction avec la
metadata YAML. La suite `validation/tests` execute 40 tests mais termine avec
2 echecs et 7 erreurs pour les memes categories. Sur 376 catalogues, 329 ont un
bloc `validation`; 327 sont `pending`, 326 de leurs chemins notebook n'existent
pas et seulement deux validations CIR sont `available`. Le depot ne contient
qu'un `validation.ipynb` sous `catalog`.

**Impact.** Aucun cache de reference persistant n'est actuellement acceptable
par le pipeline qui doit le publier. Le fail-closed evite une fausse validation,
mais la CI de reference reste integralement rouge et la couverture annoncee
par les catalogues n'est pas realisable depuis leurs liens.

**Changement minimal a evaluer.** Classer chaque echec entre source changee,
policy changee et metadata contradictoire; regenerer/revalider uniquement avec
le backend independant requis, puis synchroniser atomiquement cache et YAML.
Ne pas affaiblir les controles d'empreinte.

**Critere de cloture.** `ctest -L cached_reference` passe 71/71,
`python -m unittest discover -s validation/tests` passe sans erreur, tous les
artefacts declares existent et chaque statut `available` correspond a un cache
fail-closed courant.

### VALID-002 — Lier datasets et metadonnees a l'executable qui les a produits

**Etat : ouvert.**

**Coordination : a faire avec `PERF-010`, puis avant `VALID-001`.** Partager les
champs stables de revision, build et device avec la baseline de performance,
mais exclure timings et mesures volatiles de l'empreinte semantique des prix.

**Severite : moyenne · Priorite : haute · Confiance : prouvee.**

**Preuve.** Les datasets publient les seeds par ligne et une convention de
temps, mais pas la revision, l'empreinte du generateur, le compilateur, les
flags, la version CUDA ou le GPU. Les caches de reference hashent prix,
parametres et implementation Python de comparaison, pas le generateur C++/CUDA
ni sa configuration. Les YAML rough Bergomi decrivent encore un kernel unique,
512 threads et un workspace historique de 660 602 880 octets; le generateur
courant publie un pipeline FFT chunked avec `path_chunk_size`,
`workspace_bytes` et `(2 + 2*chunks)*result_count` lancements.

**Impact.** Il est impossible d'attribuer exactement un dataset a un binaire
ou de determiner automatiquement qu'une metadata d'implementation est devenue
obsolette. Une regeneration numeriquement differente peut partager la meme
enveloppe de provenance.

**Changement minimal a evaluer.** Ajouter un manifeste de provenance
deterministe couvrant revision/source du generateur, contrat temporel, build et
device; inclure son empreinte dans la publication et dans les caches de
reference, sans melanger les timings volatils aux prix semantiques.

**Critere de cloture.** Un dataset permet de reconstruire ou identifier son
executable, toute modification semantique invalide la reference attendue et les
YAML rough sont regeneres depuis les champs emis par le pipeline courant.
