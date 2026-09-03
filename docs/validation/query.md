# Referentiel de l'audit de validation et reproductibilite

## Objet

Ce document definit l'audit volontairement separe de la chaine de validation,
des references independantes et de la provenance des datasets publies. Cet
audit est souvent long et ne fait pas partie d'un passage ordinaire de
`../audit/query.md`; il doit etre lance explicitement.

`status.md` conserve la date, la revision, le perimetre, les exclusions et les
preuves de son dernier passage. Les constats non resolus sont inscrits dans
`response.md`, avec un identifiant stable, un etat, une severite, une priorite,
une confiance, une preuve, un impact concret, un changement minimal a evaluer
et un critere de cloture.

Les constats fermes sont deplaces dans `closed.md` avec leur signature, la
nature de leur cloture, leur preuve et leur condition de reouverture. Avant de
creer un identifiant, rechercher une cause et un perimetre equivalents dans
`response.md` et `closed.md`. Un constat qui regresse reutilise son identifiant.
Lorsqu'un constat partage une migration ou une preuve avec l'audit principal,
son champ **Coordination** cite l'identifiant concerne et fixe l'ordre sans
melanger les proprietaires documentaires.

Les tests cibles necessaires pour prouver un constat Dynamics, Analytics,
Numerical robustness ou Build restent dans l'audit proprietaire. Le present
audit possede l'inventaire transversal et potentiellement couteux des moteurs
externes, caches, notebooks, fingerprints, schemas et procedures de
regeneration/publication.

## Validation and reproducibility

Auditer la matrice de validation des dynamics, analytics et pricers ainsi que la
provenance des datasets publies. Ne pas confondre test de contrat, reference
mathematique deterministe, comparaison a une bibliotheque externe et test
statistique Monte-Carlo.

**Perimetre :** couverture des tests et validations, independance des
references, provenance, invalidation et reproductibilite des artefacts publies.

**Hors perimetre :** correction mathematique deja prouvee par les audits
proprietaires et performance sans consequence sur la reproductibilite.

**Preuves attendues :** matrice de couverture, tests executes, moteurs de
reference, fingerprints, seeds, schemas et chaine de regeneration/publication.

**Livrable :** cases absentes ou asymetriques, references non equivalentes,
artefacts reutilisables a tort et procedure de revalidation explicite.

Verifier notamment :

- l'existence d'une matrice explicite `model x curve x product x side x method`
  indiquant ce qui est compile, teste, valorise et valide ;
- la symetrie call/put, payer/receiver, regular/explicit et courbes supportees,
  ou la justification documentee de chaque case absente ;
- que les dynamics possedent des tests de contrat et de loi independamment des
  pricers qui les consomment ;
- que les analytics possedent des identites mathematiques et des references
  independantes avant d'etre valides seulement par un produit compose ;
- que les comparaisons Premia, QuantLib ou autres documentent conventions,
  parametrages, limites connues et adaptations necessaires, sans transformer un
  desaccord externe connu en tolerance silencieuse ;
- que les validations Monte-Carlo tiennent compte de l'erreur standard, du biais
  de discretisation, du biais Longstaff-Schwartz et de la resolution FFT ;
- que les seeds, geometries CUDA et mappings Philox permettent de reproduire
  exactement un dataset lorsque ce comportement est contractuel ;
- que SHA-256, provenance des inputs et `validation_policy_fingerprint`
  couvrent les donnees, le generateur, les criteres et la convention de temps
  necessaires pour empecher la reutilisation d'une ancienne validation ;
- qu'un changement de politique, formule, calendrier ou convention invalide les
  artefacts concernes et force une revalidation explicite ;
- que JSON et YAML publies respectent le meme schema semantique et que les
  loaders rejettent une ambiguite plutot que d'inventer une valeur par defaut ;
- que les tests de non-regression distinguent egalite bit a bit, tolerance
  numerique et intervalle statistique ;
- que les scripts de validation repetitifs sont remplaces par un CLI ou un
  registre declaratif, tout en conservant les points d'entree publies qui sont
  encore contractuels ;
- que chaque chemin de notebook, rapport, cache et dataset pointe vers un
  artefact existant et conforme au workflow de validation actuel ;
- que les procedures de regeneration, validation, publication et diagnostic
  sont documentees, automatisables et independantes d'un etat local implicite.

Une reference externe ne doit pas etre consideree comme verite absolue sans
verifier qu'elle valorise exactement le meme contrat avec les memes conventions.
