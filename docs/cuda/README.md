# Documentation CUDA

Ce dossier regroupe les guides d'architecture, contrats d'implementation et
notes materielles propres aux backends CUDA. Les workflows de catalogue, de
generation de datasets et de validation independante restent au niveau
superieur, car leur responsabilite ne se limite pas a CUDA.

## Architecture

- [`pricing-policy-composition.md`](pricing-policy-composition.md) : guide
  visuel de la composition des dynamiques, schedules, produits, handlers,
  pricing policies et kernels. La section markovienne est la premiere section
  complete ; les schemas closed form, Volterra FFT et rough N-facteurs seront
  ajoutes apres validation.
- [`model-dynamics-contract.md`](model-dynamics-contract.md) : contrat des
  etats, transitions fixes ou exactes, contexts Philox et policies de
  dynamique.
- [`model-analytics-contract.md`](model-analytics-contract.md) : contrat des
  analytics reutilisables et de leurs providers.

## Pricing et execution

- [`closed-form-and-monte-carlo-pricing-contract.md`](closed-form-and-monte-carlo-pricing-contract.md) :
  contrats des pricers closed form et Monte Carlo ordinaires.
- [`american-and-bermudan-pricing-contract.md`](american-and-bermudan-pricing-contract.md) :
  contrats Longstaff--Schwartz des produits a exercice anticipe.
- [`launch-validation-and-kernel-diagnostics.md`](launch-validation-and-kernel-diagnostics.md) :
  validation des lancements, ressources, occupation et diagnostics.

## Materiel et performance

- [`rtx4090-laptop-memory-map.ipynb`](rtx4090-laptop-memory-map.ipynb) : carte
  courte des limites CUDA du GPU SM89 de reference.
- [`../../validation/performance/README.md`](../../validation/performance/README.md) :
  protocole de benchmark, profils de tuning et baselines mesurees.
