# Conception proposée du téléchargement protégé des datasets

Ce document définit une architecture cible entre le site statique AI Factory,
Cloudflare Turnstile et le serveur Sorbonne. Elle n'est pas implémentée dans ce
dépôt. Les noms d'API et extraits ci-dessous sont des contrats de conception,
pas la description d'un service déjà déployé.

Avant l'implémentation, vérifier les détails d'intégration dans la documentation
officielle Turnstile liée en fin de page.

## Responsabilités

| Composant | Responsabilité |
|---|---|
| Site statique | Demander un token Turnstile et transmettre un identifiant de dataset |
| Turnstile | Attester la vérification anti-bot ; il ne crée aucune URL de téléchargement |
| API Sorbonne | Valider le token, appliquer les limites et signer une URL temporaire |
| Apache ou Nginx | Servir le fichier privé après autorisation |

Les trois valeurs de configuration ne sont pas interchangeables :

| Valeur | Emplacement | Secrète | Rôle |
|---|---|---:|---|
| `TURNSTILE_SITEKEY` | Site statique | non | Créer le contrôle Turnstile |
| `TURNSTILE_SECRET` | API Sorbonne | oui | Valider le token auprès de Cloudflare |
| `DOWNLOAD_SIGNING_SECRET` | API Sorbonne | oui | Signer les URL temporaires |

Les secrets sont injectés par la configuration du service. Ils ne doivent
jamais apparaître dans Git ni dans les fichiers JavaScript publiés.

## Flux cible

```text
clic Download
  -> token Turnstile à usage unique
  -> POST /api/request-download {dataset_id, captcha_token}
  -> validation Cloudflare + limites serveur + table fermée de datasets
  -> URL signée à courte durée de vie
  -> GET /download/<id>?expires=...&signature=...
  -> envoi du fichier privé par le serveur web
```

### Site statique

Le catalogue expose un `downloadId` stable, jamais un chemin de fichier :

```js
{
  title: "Heston European Calls 01",
  downloadId: "heston-european-calls-01"
}
```

Au clic, le site exécute le widget Turnstile, envoie l'ID et le token à l'API,
puis ouvre uniquement l'URL retournée par une réponse réussie. Il bloque les
doubles clics, affiche les erreurs et réinitialise le widget après chaque
tentative. Le contrôle doit rester affichable si Turnstile exige une
interaction.

Si le site et l'API partagent la même origine, aucune règle CORS additionnelle
n'est nécessaire. Sinon, l'API autorise explicitement l'origine du site et
aucune origine générique.

### API d'autorisation

L'API reçoit :

```http
POST /api/request-download
Content-Type: application/json

{"dataset_id":"heston-european-calls-01","captcha_token":"..."}
```

Elle valide le token côté serveur avec `TURNSTILE_SECRET` et refuse par défaut
une erreur, un délai dépassé ou une réponse incohérente. La réponse Turnstile
doit confirmer au minimum le succès, le domaine attendu et l'action
`request_download`.

Après cette validation, l'API :

1. applique une limite par client et une limite globale ;
2. cherche `dataset_id` dans une table fermée contrôlée côté serveur ;
3. calcule une expiration courte ;
4. signe l'identifiant et l'expiration avec HMAC-SHA-256 ;
5. renvoie un chemin relatif vers le même service.

Un identifiant reçu du navigateur ne doit jamais être converti directement en
chemin. La table interne réalise seule la correspondance, par exemple :

```text
heston-european-calls-01
  -> /srv/ai-factory/datasets/heston/european-calls-01.tar.zst
```

La réponse réussie contient uniquement l'URL temporaire :

```json
{"download_url":"/download/heston-european-calls-01?expires=...&signature=..."}
```

### Service du fichier

Pour chaque requête de téléchargement, le serveur vérifie l'expiration, compare
la signature en temps constant et retrouve le chemin privé depuis la table
fermée. L'API d'autorisation délègue ensuite le transfert à Apache ou Nginx
(`X-Sendfile`, `X-Accel-Redirect` ou mécanisme équivalent) au lieu de transporter
elle-même les fichiers volumineux.

Une URL expirée ne peut plus démarrer de transfert. Un transfert déjà autorisé
ne doit pas être interrompu par l'expiration, et les requêtes `Range` doivent
permettre la reprise.

## Exigences de sécurité

- aucun fichier ne possède d'URL publique permanente ;
- les tokens Turnstile sont validés une seule fois et uniquement côté serveur ;
- `hostname` et `action` sont vérifiés, pas seulement `success` ;
- les appels à Cloudflare ont un délai maximal court et échouent fermés ;
- les limites de débit précèdent les opérations coûteuses ;
- les messages publics ne révèlent ni chemin privé ni secret ;
- les journaux excluent tokens, signatures et secrets ;
- la rotation des deux secrets serveur est prévue avant le déploiement ;
- l'expiration, la signature et la table d'ID sont testées indépendamment de
  Turnstile.

## Vérification avant mise en service

- token valide : une URL temporaire est rendue ;
- token absent, invalide, expiré ou réutilisé : refus ;
- domaine ou action inattendu : refus ;
- ID inconnu : refus sans accès au système de fichiers ;
- limite atteinte : réponse explicite sans création d'URL ;
- signature modifiée ou URL expirée : téléchargement refusé ;
- URL valide : téléchargement complet et reprise avec `Range` ;
- indisponibilité Cloudflare : refus contrôlé dans le délai déclaré ;
- secrets et chemins privés absents des bundles et des journaux ;
- tests de charge sur l'autorisation et sur le serveur de fichiers.

## Décisions encore ouvertes

L'implémentation doit fixer et documenter le langage du service, le stockage de
la table d'ID, la durée d'expiration, les limites de débit, la stratégie de
rotation des secrets, la politique de journaux et le déploiement Apache/Nginx.
Tant que ces décisions ne sont pas prises et testées, cette page reste une
proposition et ne doit pas être citée comme fonctionnalité disponible.

## Références officielles

- [Intégration côté navigateur](https://developers.cloudflare.com/turnstile/get-started/client-side-rendering/)
- [Validation côté serveur](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)
- [Clés de test](https://developers.cloudflare.com/turnstile/troubleshooting/testing/)
