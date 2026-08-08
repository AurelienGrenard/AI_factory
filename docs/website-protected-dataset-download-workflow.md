# Workflow de téléchargement protégé des datasets du site

Cette note décrit le flux proposé entre le site statique AI Factory, Cloudflare
Turnstile et le serveur Sorbonne. Les extraits sont des exemples d'intégration :
aucun de ces changements n'est encore appliqué au site.

## Vue d'ensemble

```text
1. Clic sur Download
2. Le site demande un token à Turnstile
3. Le site envoie { dataset_id, captcha_token } à la Sorbonne
4. La Sorbonne valide le token auprès de Cloudflare
5. La Sorbonne applique ses limites et crée une URL temporaire signée
6. Le navigateur ouvre cette URL
7. Apache/Nginx envoie directement le fichier privé
```

Point important : **Cloudflare ne fournit pas l'URL de téléchargement**.
Cloudflare valide seulement le token anti-bot. Le serveur Sorbonne crée ensuite
lui-même l'URL temporaire.

## Les trois clés/valeurs à ne pas confondre

| Valeur | Emplacement | Secrète ? | Rôle |
|---|---|---:|---|
| `TURNSTILE_SITEKEY` | site statique | non | demander un token à Turnstile |
| `TURNSTILE_SECRET` | serveur Sorbonne | oui | faire valider ce token par Cloudflare |
| `DOWNLOAD_SIGNING_SECRET` | serveur Sorbonne | oui | signer les URL temporaires |

Les deux secrets doivent être fournis au service par sa configuration sécurisée
(variables d'environnement ou gestionnaire de secrets). Ils ne doivent jamais
être écrits dans Git, `catalog.js` ou `catalog-data.js`.

## 1. Modifications du site statique

### Charger Turnstile

Dans `index.html`, ajouter le conteneur, puis charger la bibliothèque officielle
**après** le script actuel `static/catalog.js` :

```html
<div id="turnstile-download"></div>

<script src="static/catalog.js"></script>
<script
  src="https://challenges.cloudflare.com/turnstile/v0/api.js?onload=initDownloadTurnstile&render=explicit"
  async defer></script>
```

Le conteneur ne doit pas être supprimé avec `display: none` : Turnstile doit
pouvoir y afficher un contrôle si une interaction est nécessaire.

### Donner un ID au bouton Download

Une ressource téléchargeable ne doit plus fournir une URL publique au bouton :

```js
{
  title: "Heston European Calls 01",
  downloadId: "heston-european-calls-01"
}
```

Le HTML du bouton contient cet ID :

```html
<button class="button" data-download-id="heston-european-calls-01">
  Download
</button>
```

L'ID n'est pas secret. Il permet au serveur de sélectionner un fichier dans une
liste fermée.

### Obtenir le token et appeler la Sorbonne

Exemple adapté au catalogue dynamique actuel :

```js
const TURNSTILE_SITEKEY = "CLE_PUBLIQUE_FOURNIE_PAR_CLOUDFLARE";
let turnstileWidgetId;
let pendingDatasetId = null;

window.initDownloadTurnstile = () => {
  turnstileWidgetId = turnstile.render("#turnstile-download", {
    sitekey: TURNSTILE_SITEKEY,
    execution: "execute",
    appearance: "interaction-only",
    action: "request_download",
    callback: sendDownloadRequest,
    "error-callback": () => showError("La vérification anti-bot a échoué.")
  });
};

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-download-id]");
  if (!button || pendingDatasetId) return;

  pendingDatasetId = button.dataset.downloadId;
  turnstile.execute(turnstileWidgetId);
});

async function sendDownloadRequest(captchaToken) {
  try {
    const response = await fetch("/api/request-download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: pendingDatasetId,
        captcha_token: captchaToken
      })
    });

    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Téléchargement refusé");

    window.location.assign(result.download_url);
  } catch (error) {
    showError(error.message);
  } finally {
    pendingDatasetId = null;
    turnstile.reset(turnstileWidgetId);
  }
}
```

Le token est temporaire (cinq minutes) et utilisable une seule fois. Le widget
est donc réinitialisé après chaque demande.

Si l'API et le site utilisent tous deux `mlp.lpma.math.upmc.fr`, l'appel reste
sur la même origine et aucune configuration CORS particulière n'est nécessaire.

## 2. Traitement côté serveur Sorbonne

Le serveur expose :

```http
POST /api/request-download
Content-Type: application/json

{
  "dataset_id": "heston-european-calls-01",
  "captcha_token": "TOKEN_RECU_DE_TURNSTILE"
}
```

### A. Valider le token auprès de Cloudflare

Le serveur appelle Cloudflare, jamais le navigateur :

```http
POST https://challenges.cloudflare.com/turnstile/v0/siteverify
Content-Type: application/json

{
  "secret": "TURNSTILE_SECRET",
  "response": "TOKEN_RECU_DE_TURNSTILE",
  "remoteip": "IP_DU_VISITEUR"
}
```

Exemple du cœur de l'appel, dans un pseudo-code JavaScript côté serveur :

```js
const cloudflareResponse = await fetch(
  "https://challenges.cloudflare.com/turnstile/v0/siteverify",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      secret: process.env.TURNSTILE_SECRET,
      response: captchaToken,
      remoteip: visitorIp
    })
  }
);

const check = await cloudflareResponse.json();

const accepted =
  check.success === true &&
  check.hostname === "mlp.lpma.math.upmc.fr" &&
  check.action === "request_download";

if (!accepted) return http403();
```

`hostname` prouve que le token a été produit sur le domaine autorisé. `action`
prouve qu'il a été demandé pour un téléchargement et non pour une autre fonction
du site.

Le serveur doit utiliser un délai maximal court pour cet appel et refuser la
demande si Cloudflare ne répond pas. Il doit également limiter le nombre de
requêtes par IP pour empêcher qu'on surcharge cette route.

### B. Contrôler puis créer l'URL temporaire

Après la validation Turnstile, le serveur :

1. applique les limites par IP et la limite globale ;
2. cherche `dataset_id` dans une table interne ;
3. refuse tout ID absent ;
4. calcule une expiration courte ;
5. signe `dataset_id + expiration` avec `DOWNLOAD_SIGNING_SECRET`.

Table interne :

```text
heston-european-calls-01
  -> /srv/ai-factory/datasets/heston/european-calls-01.tar.zst
```

Calcul conceptuel de la signature :

```text
expires   = heure_actuelle + 5 minutes
signature = HMAC-SHA-256(
  DOWNLOAD_SIGNING_SECRET,
  dataset_id + "|" + expires
)
```

Réponse envoyée au site :

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "download_url": "/download/heston-european-calls-01?expires=...&signature=..."
}
```

### C. Servir le fichier

Pour `GET /download/...`, le serveur :

1. vérifie que l'expiration n'est pas dépassée ;
2. recalcule la signature et la compare à celle de l'URL ;
3. retrouve le chemin privé à partir de l'ID ;
4. autorise Apache/Nginx à envoyer le fichier.

Le petit service d'autorisation ne doit pas transporter lui-même plusieurs Go
ou To. Nginx peut recevoir une instruction interne `X-Accel-Redirect`; Apache
peut utiliser `X-Sendfile` ou un mécanisme équivalent. Les fichiers ne doivent
avoir aucune URL publique permanente.

L'expiration limite le temps disponible pour **commencer** le téléchargement.
Elle ne doit pas interrompre un transfert déjà autorisé. Les requêtes `Range`
doivent rester possibles pour reprendre un gros téléchargement.

## 3. Tester Turnstile

Cloudflare fournit des clés de test officielles, utilisables notamment sur
`localhost` :

```text
Sitekey qui réussit toujours :
1x00000000000000000000AA

Secret qui réussit toujours :
1x0000000000000000000000000000000AA

Sitekey qui force une interaction :
3x00000000000000000000FF

Secret qui échoue toujours :
2x0000000000000000000000000000000AA
```

Pour tester directement l'API Cloudflare avec le token factice officiel :

```bash
curl -X POST \
  https://challenges.cloudflare.com/turnstile/v0/siteverify \
  --data-urlencode 'secret=1x0000000000000000000000000000000AA' \
  --data-urlencode 'response=XXXX.DUMMY.TOKEN.XXXX'
```

La paire de test doit être utilisée des deux côtés : une clé secrète de
production refuse les tokens factices. Aucune clé de test ne doit rester en
production.

Tests minimaux avant mise en service :

- token valide : une URL temporaire est rendue ;
- token absent, faux, expiré ou réutilisé : réponse `403` ;
- mauvais `hostname` ou mauvaise `action` : réponse `403` ;
- ID inconnu : réponse `404` ;
- limite atteinte : réponse `429` ou `503` ;
- signature modifiée ou URL expirée : téléchargement refusé ;
- URL valide : téléchargement et reprise avec `Range` fonctionnels.

Documentation Cloudflare :

- [Intégration côté navigateur](https://developers.cloudflare.com/turnstile/get-started/client-side-rendering/)
- [Validation côté serveur](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)
- [Clés de test officielles](https://developers.cloudflare.com/turnstile/troubleshooting/testing/)
