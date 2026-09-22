# Architecture et maintenance — fonctionnalité vocale realtime

> Mise a jour 2026-09-19 : le runtime native streame LLM/TTS, gere le barge-in,
> separe VAD et fin de tour, et expose des percentiles p50/p95.
> L'audit mesure, la carte KEEP/REPLACE et la procedure Twilio sont dans
> `voice-server/docs/VOICE_AUDIT.md`. Les sections 5.5, 9, 18 et 19 ci-dessous
> decrivent encore partiellement l'etat avant ce refresh.
>
> Correction 2026-09-22 (audit de securite) : ce document sous-estimait l'etat
> reel du code sur 3 points, corriges plus bas — le streaming LLM/TTS est deja
> actif par defaut (section "Non-objectifs actuels"), `/v1/tools/*` exige une
> cle interne en temps constant, et `/media-stream` verifie un token HMAC signe
> avant d'accepter un appel (section "Risques actuels a traiter avant
> exposition publique"). Par ailleurs, pipecat (orchestrateur candidat) a ete
> evalue puis retire du code : `livekit_agent/` est le pipeline de production,
> `app/session/*` (natif, decrit ici) reste en place en parallele.

## 1. Objet et statut du document

Ce document est la référence technique du parcours téléphonique realtime de
MySmartFood. Il décrit le comportement réellement implémenté dans :

- `voice-server/` pour le runtime vocal Python ;
- `backend/` pour Twilio, le routage, le fallback et la logique métier ;
- `frontend/src/Pages/ServiceHealth/` pour la supervision administrateur ;
- `infrastructure/scaleway-vllm/` pour l'inférence vLLM auto-hébergée.

Il doit être mis à jour lorsqu'un contrat HTTP, un événement WebSocket, une
règle métier, une stratégie de fallback ou une métrique change.

Documents complémentaires :

- `docs/VOICE_PLATFORM.md` : vue plateforme et déploiement ;
- `backend/MONITORING.md` : contrat du monitoring backend ;
- `infrastructure/scaleway-vllm/README.md` : exploitation du serveur GPU.

### Niveaux de garantie

La distinction suivante est essentielle pour maintenir la fonctionnalité :

- **Garantie applicative** : règle contrôlée par du code déterministe côté
  Fastify ou Voice Server.
- **Politique conversationnelle** : instruction donnée au LLM dans le prompt.
  Elle améliore le comportement, mais ne constitue pas une validation de
  sécurité ou une garantie métier.
- **Hypothèse d'infrastructure** : propriété attendue du réseau ou du
  déploiement, non imposée par le code.

## 2. Objectifs, non-objectifs et invariants

### Objectifs

- Répondre à un appel Twilio avec une assistante francophone.
- Détecter les tours de parole et accepter une interruption simple.
- Transcrire la parole localement avec faster-whisper.
- Utiliser vLLM comme fournisseur LLM principal.
- Synthétiser la réponse localement avec Kokoro.
- Consulter les disponibilités et créer une réservation ou une commande.
- Conserver la logique métier et l'accès MongoDB dans Fastify.
- Basculer vers OpenAI lorsqu'une brique locale est indisponible.
- Exposer un état sanitaire et des métriques sans données personnelles.

### Non-objectifs actuels

- Le Voice Server n'accède jamais directement à MongoDB.
- Le LLM n'exécute jamais de requête arbitraire ni de code applicatif.
- Le Voice Server ne calcule ni prix, ni capacité, ni droits métier.
- Le LLM et le TTS streament par defaut (`llm_stream`/`tts_stream`, voir
  `app/config.py`) ; le STT reste en mode "utterance complete" (V1 du
  `STTEngine`), un mode partiels/streaming est documente comme V2 future mais
  non implemente.
- Le runtime Python ne persiste ni conversation, ni métrique, ni session.
- La fin d'un appel local ne déclenche pas l'extraction post-appel historique.

### Invariants à préserver

1. Twilio entre par Fastify ou par le Gateway, jamais directement par Python
   dans la topologie avec fallback global.
2. Toute mutation métier passe par un endpoint Fastify.
3. Un appel possède son historique, ses buffers et son état VAD propres.
4. Un numéro Twilio français valide remplace toujours le numéro proposé par le
   LLM lors de `create_appointment`.
5. Les logs et le monitoring ne contiennent ni numéro, ni transcription, ni
   prompt, ni réponse LLM, ni secret.
6. Un modèle vLLM configuré doit correspondre exactement à l'identifiant
   renvoyé par `GET /v1/models`.

## 3. Vue d'ensemble

```text
Appelant
   |
   | réseau téléphonique
   v
Twilio
   |
   | POST /incoming-call
   | ou POST /twilio/:slug/incoming-call sur le Gateway
   v
Fastify / Gateway
   |
   | TwiML <Connect><Stream>
   | wss://hôte-public/media-stream
   v
Routeur vocal Fastify
   |
   | sonde les Voice Servers
   |
   +-- moteurs prêts --------------------------+
   |                                           |
   | WebSocket interne                         |
   | ws://voice-server:8090/media-stream       |
   v                                           |
Voice Server FastAPI                           |
   |                                           |
   | mu-law 8 kHz -> VAD -> STT -> LLM         |
   |                         |                 |
   |                         +--> outils HTTP --+--> Fastify -> MongoDB
   |                                           |
   | réponse -> TTS -> mu-law 8 kHz            |
   +---------------------------> Twilio --------+--> Appelant
   |
   +-- moteurs indisponibles --> OpenAI Realtime dans Fastify
```

Le chemin local est hybride par conception :

- audio, orchestration, VAD, STT et TTS dans Python ;
- inférence LLM dans vLLM, local ou distant ;
- métier et persistance dans Fastify ;
- transport téléphonique chez Twilio.

## 4. Topologies supportées

### 4.1 Déploiement monolithique

Le backend principal expose :

- `POST|GET /incoming-call` pour le webhook Twilio ;
- `GET /media-stream` en WebSocket pour le flux audio ;
- les endpoints métier sous `/api`.

`backend/Routes/Ws/ws.js` sonde les Voice Servers puis utilise le routeur
partagé `routeVoiceConnection`.

Cette topologie convient au développement et au déploiement mono-instance.
`INSTANCE_ID` détermine l'instance métier, avec `inst_default` comme valeur de
repli.

### 4.2 Gateway vocal multi-instance

`backend/gateway.js` expose :

- `POST /twilio/:slug/incoming-call` ;
- `GET /v1/:instanceId/media-stream` en WebSocket ;
- `GET /health`.

Le Gateway résout le `slug` vers une instance active, vérifie l'activation de
la ligne téléphonique, puis transmet `instanceId` au même routeur vocal.

### 4.3 Pool de Voice Servers

`VOICE_SERVICE_URLS` peut contenir plusieurs bases HTTP séparées par des
virgules. `backend/Config/voiceServiceTargets.js` construit pour chaque base :

- `/health` ;
- `/monitoring` ;
- `/media-stream`.

Les sondes sont exécutées en parallèle. Seuls les réplicas dont
`engines.ready` est vrai participent à une sélection round-robin.

Le compteur round-robin est local au processus Node. Il n'est pas partagé entre
plusieurs workers ou plusieurs instances Fastify.

## 5. Déroulement complet d'un appel

### 5.1 Entrée Twilio

1. Twilio appelle le webhook Fastify.
2. Fastify vérifie si la ligne vocale est activée via `PricingService`.
3. Si elle est désactivée, l'appel est transféré vers le numéro du restaurant ;
   sans numéro de transfert, Twilio reçoit un rejet `busy`.
4. Si elle est activée, Fastify génère le TwiML.
5. Le TwiML joue actuellement une introduction MP3, puis ouvre
   `<Connect><Stream>`.
6. Fastify ajoute si possible deux paramètres :
   `callerNumber` et `instanceId`.

Le numéro est normalisé avant insertion dans le XML. Les valeurs sont échappées
et `instanceId` est limité au format `[A-Za-z0-9_.:-]`, 128 caractères.

### 5.2 Sélection du moteur

À l'ouverture de `/media-stream`, Fastify :

1. met temporairement en file les premiers messages Twilio ;
2. sonde chaque cible avec un timeout de 800 ms ;
3. sélectionne un Voice Server prêt ;
4. ouvre un WebSocket interne avec un timeout de 2 secondes ;
5. rejoue les messages mis en file ;
6. proxifie ensuite les deux directions jusqu'à la fermeture.

Si aucune cible n'est prête, si la sonde échoue ou si le proxy ne s'ouvre pas,
Fastify rejoue les messages vers son implémentation OpenAI Realtime.

Une fois le proxy Python ouvert, une panne en milieu d'appel est seulement
journalisée : le code ne reconnecte pas la session et ne rebascule pas cet
appel vers OpenAI Realtime.

### 5.3 Ouverture de la session Python

`app.main.media_stream` accepte le WebSocket et crée exactement un
`CallSession`. Le `SessionManager` indexe les sessions par l'identité du
WebSocket.

À l'événement Twilio `start`, la session :

1. mémorise `streamSid` et `callSid` ;
2. normalise le numéro appelant ;
3. assainit `instanceId` ;
4. charge le contexte restaurant depuis Fastify, avec un timeout de 2 secondes ;
5. construit le prompt système spécifique à l'appel ;
6. synthétise le message d'accueil.

L'indisponibilité du contexte restaurant ne ferme pas l'appel. La session
continue avec le prompt générique et expose `voice_context_unavailable` dans
son état sanitizé.

### 5.4 Réception audio et VAD

Twilio envoie du G.711 mu-law mono à 8 kHz, généralement par trames de 20 ms.

Pour chaque événement `media` :

1. le payload base64 est décodé ;
2. le mu-law est converti en PCM16 ;
3. Silero traite des fenêtres de 256 échantillons à 8 kHz ;
4. le début et la fin de parole sont déterminés par seuil et hystérésis ;
5. le tour complet est envoyé dans une tâche asynchrone.

Le pré-roll peut croître jusqu'à environ 1 seconde. Lorsqu'il dépasse cette
limite, il est tronqué pour ne conserver que les 200 dernières millisecondes ;
sa taille effective avant une parole varie donc entre environ 200 ms et 1 s.
Une phrase de moins de 1 600 échantillons, soit 200 ms à 8 kHz, est ignorée.

Avec la configuration par défaut :

- seuil VAD : `0.5` ;
- parole minimale : `250 ms` ;
- silence terminant le tour : `400 ms`.

### 5.5 Traitement d'un tour

```text
PCM16 8 kHz
   |
   | rééchantillonnage linéaire
   v
PCM16 16 kHz
   |
   v
faster-whisper
   |
   | texte complet
   v
vLLM /v1/chat/completions
   |
   +-- réponse texte ----------------------+
   |                                       |
   +-- tool_calls -> ToolDispatcher        |
                      |                    |
                      v                    |
                    Fastify                |
                      |                    |
                      v                    |
                    MongoDB                |
                      |                    |
                      +--> résultat -> LLM-+
                                              |
                                              v
                                          Kokoro
                                              |
                                      PCM 24 kHz
                                              |
                                      PCM16 8 kHz
                                              |
                                       G.711 mu-law
                                              |
                                      trames de 20 ms
                                              v
                                            Twilio
```

Les étapes STT, LLM et TTS sont exécutées séquentiellement pour un tour.
Chaque tour possède un verrou empêchant deux traitements simultanés dans la
même session.

### 5.6 Fin d'appel

Les événements `stop` ou `closed`, une déconnexion WebSocket et l'arrêt du
processus suivent le même nettoyage :

- annulation de la tâche active ;
- vidage des buffers ;
- suppression de l'historique ;
- remise à zéro du VAD ;
- suppression de la référence WebSocket ;
- retrait de la session du manager.

`close` est idempotente. Le nettoyage est garanti par le bloc `finally` de la
route WebSocket et par `close_all` pendant l'arrêt FastAPI.

## 6. Machine d'états d'une session

États observables actuellement :

```text
connecting
   |
connected
   |
greeting
   |
listening <------------------------------------+
   |                                          |
processing                                    |
   |                                          |
stt                                           |
   |                                          |
llm <--> tool:check_availability               |
   |     tool:create_appointment               |
   |                                          |
tts ------------------------------------------+
   |
stopping
   |
closed
```

Cette machine d'états sert au diagnostic. Elle n'est pas une machine métier :
elle ne prouve pas qu'une disponibilité a été vérifiée ou qu'une confirmation
client a été reçue.

## 7. Responsabilités par composant

### `app/main.py`

- construit l'application FastAPI ;
- gère le lifespan et le préchargement ;
- expose les sondes et le WebSocket ;
- vérifie l'identifiant du modèle vLLM au démarrage ;
- construit le snapshot sanitaire ;
- ferme les ressources partagées.

Il ne doit pas contenir de logique conversationnelle ou métier.

### `app/runtime.py`

- regroupe les dépendances partagées ;
- expose l'état individuel des moteurs ;
- expose le fournisseur LLM actif.

### `app/session/`

- possède l'état privé d'un appel ;
- orchestre VAD, STT, LLM, outils et TTS ;
- gère les interruptions, erreurs et nettoyages ;
- produit uniquement un snapshot sanitizé.

### `app/twilio/`

- parse les événements JSON ;
- génère les messages `media` et `clear`.

Ce module ne connaît ni le métier, ni les modèles.

### `app/audio/`

- convertit mu-law et PCM16 ;
- encode et découpe les trames sortantes ;
- rééchantillonne les signaux ;
- fournit les buffers de session.

Le rééchantillonnage actuel utilise une interpolation linéaire.

### `app/vad/`

- charge une session ONNX Silero partagée ;
- crée un état RNN, un contexte et une hystérésis par appel ;
- détecte début et fin de parole.

L'inférence VAD utilise le CPU et limite ONNX Runtime à un thread inter-op et
un thread intra-op.

### `app/stt/`

- charge un modèle faster-whisper partagé ;
- transcrit une phrase complète ;
- exécute le calcul bloquant dans un thread ;
- sérialise les transcriptions avec un verrou global au processus.

Ce verrou protège le modèle mais limite le débit à une transcription simultanée
par réplica.

### `app/llm/`

- définit une interface indépendante du fournisseur ;
- appelle une API compatible OpenAI ;
- transforme les tool calls en objets internes ;
- tente vLLM puis éventuellement OpenAI Chat Completions.

Le client HTTP autorise plusieurs requêtes concurrentes. Il utilise un timeout
de 60 secondes.

### `app/prompts.py`

- définit le rôle de l'assistante ;
- définit la politique conversationnelle ;
- décrit les données obligatoires ;
- incorpore le numéro vérifié ;
- incorpore un contexte restaurant limité à 16 000 caractères.

Les données restaurant sont explicitement déclarées non exécutables dans le
prompt. Cette instruction réduit le risque d'injection indirecte mais ne
remplace pas une validation structurelle.

### `app/tools/`

- expose uniquement `check_availability` et `create_appointment` au LLM ;
- valide les éléments minimaux avant l'appel HTTP ;
- transforme certaines erreurs Fastify en consignes conversationnelles.

Le dispatcher ne connaît pas MongoDB.

### `app/node_client.py`

- traduit le contrat Python vers les endpoints Fastify ;
- applique le header interne ;
- choisit réservation ou commande ;
- normalise le résultat HTTP en réponse d'outil.

### Fastify

- possède les règles métier ;
- résout l'instance ;
- lit la configuration restaurant ;
- valide et normalise les entrées ;
- contrôle la capacité ;
- assure l'idempotence applicative ;
- écrit dans MongoDB ;
- décide du chemin Voice Server ou OpenAI Realtime ;
- agrège la supervision.

### Frontend administrateur

Le frontend ne participe pas au traitement d'un appel. Il :

- interroge `GET /api/monitoring` via un service ;
- normalise les anciennes et nouvelles formes du payload ;
- rafraîchit toutes les 5 secondes ;
- signale une donnée obsolète après environ 10 secondes ;
- affiche services, moteurs, appels actifs, fallback et alertes.

## 8. Isolation et concurrence

### État privé par appel

Chaque `CallSession` possède :

- son historique LLM ;
- ses buffers audio ;
- son état VAD ;
- son événement d'annulation ;
- son verrou de tour ;
- sa tâche active ;
- ses identifiants techniques.

Aucun historique ou numéro n'est partagé entre deux appels.

### Ressources partagées par processus

- modèle Silero ONNX ;
- modèle faster-whisper ;
- modèle Kokoro ;
- client HTTP LLM ;
- client HTTP Fastify ;
- télémétrie agrégée.

### Sérialisation

- STT : une transcription à la fois par processus ;
- TTS : une synthèse à la fois par processus ;
- envoi WebSocket : un message à la fois par session ;
- LLM : pas de verrou local, capacité déléguée à vLLM ;
- VAD : état privé par session sur un modèle ONNX partagé.

Ajouter des appels sans mesurer ces verrous augmente directement le temps
d'attente STT et TTS.

## 9. Barge-in et annulation

Quand Silero détecte que le client parle pendant une réponse :

1. l'événement d'annulation est posé ;
2. un message Twilio `clear` supprime l'audio déjà en attente ;
3. la tâche de réponse active est annulée ;
4. le nouveau tour remplace le précédent après la fin de parole.

Limites :

- la coupure n'est pas frame-accurate ;
- un calcul exécuté dans `asyncio.to_thread` peut continuer côté thread même si
  la coroutine est annulée ;
- une requête HTTP LLM peut être annulée côté client sans garantir l'arrêt de
  la génération côté serveur ;
- aucune marque Twilio n'est utilisée pour confirmer la lecture effective.

## 10. Contexte et politique métier

### 10.1 Contexte restaurant

Le Voice Server appelle :

```text
GET /api/voice/context/:instanceId
x-api-key: NODE_API_KEY
```

Fastify protège cette route avec `requireInternalApiKey`, puis
`VoiceContextController` récupère le contexte depuis `PricingService`.

Le prompt reçoit uniquement :

- nom du restaurant ;
- adresse ;
- horaires d'ouverture ;
- menu ;
- disponibilité générale.

### 10.2 Numéro appelant

Le runtime Python accepte actuellement uniquement un numéro français au format
`+33...` ou `0033...`, qu'il convertit en format national `0...`.

Les valeurs masquées comme `anonymous`, `private` ou `withheld` sont ignorées.

Si un numéro vérifié existe :

- le prompt interdit de le redemander ;
- `_force_caller_number` l'impose à tout appel `create_appointment`.

Si le numéro est absent :

- le prompt demande au LLM de le collecter ;
- Fastify reste responsable de sa validation.

### 10.3 Politique conversationnelle

Le prompt demande au modèle de :

- parler uniquement français ;
- produire des réponses courtes sans markdown ;
- ne pas répéter la réponse du client ;
- poser uniquement la prochaine question utile ;
- faire un seul récapitulatif final ;
- attendre une confirmation explicite ;
- vérifier la disponibilité avant de proposer un créneau ;
- ne jamais annoncer une création avant le succès de l'outil.

Ces règles sont des **politiques LLM**, pas des garanties applicatives.
Actuellement, aucun automate métier ne bloque un appel prématuré à
`create_appointment` et Fastify ne reçoit pas de preuve de confirmation.

### 10.4 Incohérence à surveiller

Le prompt décrit en dur deux services, MIDI `11h-15h` et SOIR `18h-00h`, alors
que Fastify calcule les créneaux depuis les horaires dynamiques du restaurant.
Les horaires dynamiques doivent rester la source de vérité. Toute évolution
doit supprimer ou générer ces horaires statiques depuis le contexte.

## 11. Outils disponibles pour le LLM

### `check_availability`

Entrée :

```json
{
  "date": "YYYY-MM-DD"
}
```

Le `NodeApiClient` appelle en parallèle :

- `GET /api/orders/ai/available-slots?date=...` ;
- `GET /api/reservations/ai/available-slots?date=...`.

Le résultat retourné au LLM contient actuellement :

- les créneaux `availableSlots` de la route commandes ;
- le message de cette même route ;
- les couverts restants midi et soir de la route réservations, si disponibles.

Une erreur de la route réservations n'annule pas le résultat si la route
commandes a réussi. Les créneaux sont alors retournés sans information de
capacité de réservation.

Point d'attention : les `availableSlots` de réservation ne remplacent pas ceux
des commandes. Si les deux calendriers divergent, le contrat devra être séparé
par type d'activité ou fusionné explicitement.

### `create_appointment`

Entrée logique :

```json
{
  "name": "Nom client",
  "telephone": "0612345678",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "type": "Réservation de table | Commande à emporter",
  "description": "",
  "nombrePersonnes": 2,
  "commandes": [
    {
      "nom": "Produit",
      "quantite": 1
    }
  ]
}
```

Le type sélectionne :

- `POST /api/reservations/ai/create` ;
- `POST /api/orders/ai/create`.

Les erreurs de numéro, heure, date, capacité et validation sont converties en
codes stables afin que le LLM redemande l'information utile sans exposer le
détail technique.

Codes conversationnels actuellement produits :

- `NUMERO_MANQUANT`
- `HEURE_INVALIDE`
- `DATE_INVALIDE`
- `COUVERTS_INSUFFISANTS`
- `DONNEES_INVALIDES`

### Limite des tours d'outils

Une réponse peut effectuer au maximum `LLM_MAX_TOOL_ROUNDS`, soit 4 tours par
défaut. Au-delà, la session produit une réponse générique.

## 12. Garanties métier Fastify

### Disponibilités

Fastify :

- charge les horaires dynamiques par instance ;
- génère des créneaux de 30 minutes ;
- exclut les éléments annulés ou terminés ;
- calcule les places restantes par service pour les réservations ;
- renvoie une liste vide lorsque le restaurant est fermé ;
- renvoie 503 lorsque les horaires ne sont pas configurés.

Pour les commandes, un enregistrement existant occupe actuellement le créneau
complet. Il n'existe pas de capacité de commandes par créneau dans ce flux.

### Création de réservation

Fastify garantit :

- téléphone contenant au moins 10 chiffres ;
- heure normalisée en `HH:MM` ;
- date JavaScript valide ;
- `nombrePersonnes` par défaut à 1 ;
- contrôle de capacité lorsqu'un maximum est configuré ;
- statut initial `confirme` ;
- origine `createdBy: system`.

Une recherche de réservation récente avec même instance, téléphone, date et
heure limite les doublons. Cette idempotence n'est pas protégée par une
contrainte unique atomique : deux requêtes concurrentes peuvent encore créer
deux documents.

Réponses principales :

- 201 pour une création ;
- 200 pour un doublon reconnu ;
- 400 pour une entrée invalide ou une capacité insuffisante ;
- 500 pour une erreur serveur.

### Création de commande

Fastify garantit :

- refus du type réservation sur la route commande ;
- validation minimale du téléphone, de l'heure et de la date ;
- statut initial `confirme` ;
- origine `createdBy: system` ;
- recherche d'un doublon récent.

La route accepte actuellement une liste `commandes` vide et son schéma Fastify
n'impose pas `nom` et `quantite` sur chaque élément, contrairement au schéma
fourni au LLM. La validation backend doit être considérée comme incomplète sur
ce point.

Réponses principales :

- 201 pour une création ;
- 200 pour un doublon reconnu ;
- 400 pour une entrée invalide ;
- 500 pour une erreur serveur.

## 13. Contrats réseau

### 13.1 Endpoints du Voice Server

`GET /live`

- liveness du processus ;
- retourne 200 si FastAPI répond ;
- ne vérifie aucune dépendance.

`GET /ready`

- readiness complète ;
- retourne 200 si Fastify et tous les moteurs sont prêts ;
- retourne 503 sinon.

`GET /health`

- diagnostic compatible avec le routeur Fastify ;
- retourne toujours 200 tant que FastAPI répond ;
- utilise `status: degraded` et `engines.ready: false` pour signaler un défaut ;
- ne retourne pas les sessions détaillées.

`GET /monitoring`

- même snapshot avec `active_sessions` ;
- exige `x-api-key` égal à `NODE_API_KEY` ;
- retourne 401 si la clé manque, est vide ou diffère.

`GET /media-stream` en WebSocket

- reçoit le protocole Twilio ;
- une connexion correspond à une session.

`POST /v1/tools/check-availability`

- proxy HTTP de diagnostic vers Fastify ;
- valide seulement la présence de `date`.

`POST /v1/tools/create-appointment`

- proxy HTTP de diagnostic vers Fastify ;
- valide le schéma Pydantic minimal.

### 13.2 Endpoints Fastify consommés par Python

- `GET /api/ping`
- `GET /api/voice/context/:instanceId`
- `GET /api/orders/ai/available-slots`
- `GET /api/reservations/ai/available-slots`
- `POST /api/orders/ai/create`
- `POST /api/reservations/ai/create`

Le client envoie `x-api-key` si `NODE_API_KEY` est défini.

### 13.3 Contrat vLLM

Le Voice Server consomme :

- `GET {VLLM_BASE_URL}/models`
- `POST {VLLM_BASE_URL}/chat/completions`

Le serveur doit être compatible avec le format OpenAI et les tool calls.
Le payload contient :

- `model` ;
- `messages` ;
- `temperature` ;
- `max_tokens` ;
- `tools` et `tool_choice: auto` si nécessaire.

Pour Llama 3.1 dans vLLM, le serveur doit activer le choix automatique des
outils et le parser approprié.

## 14. Stratégie de fallback

Il existe deux niveaux indépendants.

### Niveau 1 — fallback LLM dans Python

Préconditions :

- `LLM_FALLBACK_GPT=true` ;
- `OPENAI_API_KEY` non vide.

Comportement :

1. chaque completion tente vLLM ;
2. en cas d'exception, Python tente OpenAI Chat Completions ;
3. VAD, STT, outils et TTS restent dans le pipeline maison ;
4. le fournisseur actif devient `openai` ;
5. la télémétrie incrémente `llm_primary` et
   `llm_provider_fallback`.

Si vLLM est indisponible au démarrage mais que ce fallback est configuré, le
Voice Server reste prêt et Fastify continue à lui transmettre les appels.

### Niveau 2 — fallback d'appel dans Fastify

Fastify choisit OpenAI Realtime si :

- le Voice Server est injoignable ;
- aucun réplica n'est prêt ;
- `engines.ready` est faux ;
- le proxy WebSocket ne s'ouvre pas.

Dans ce cas, l'intégralité de la conversation utilise l'ancienne pile OpenAI
Realtime Fastify.

Cette pile applique actuellement un avertissement vers 4 minutes et une fin
d'appel vers 5 minutes. La session Python locale ne possède pas encore de
limite équivalente.

### Arbre de décision

```text
Voice Server joignable ?
   |
   +-- non --> OpenAI Realtime Fastify
   |
   +-- oui --> engines.ready ?
                  |
                  +-- non --> OpenAI Realtime Fastify
                  |
                  +-- oui --> proxy WebSocket possible ?
                                 |
                                 +-- non --> OpenAI Realtime Fastify
                                 |
                                 +-- oui --> pipeline Python
                                                |
                                                +-- vLLM répond --> vLLM
                                                |
                                                +-- vLLM échoue
                                                        |
                                                        +-- clé GPT --> OpenAI Chat
                                                        |
                                                        +-- sans clé --> erreur de tour
```

Les logs utilisent volontairement `VOICE LOCAL` ou `FALLBACK GPT` afin de
rendre le chemin sélectionné visible.

## 15. Configuration

### Voice Server

Réseau :

- `VOICE_HOST`
- `VOICE_PORT`
- `NODE_BASE_URL`
- `NODE_API_KEY`

Chargement :

- `VOICE_EAGER_LOAD`
- `CACHE_DIR`

LLM principal :

- `LLM_PROVIDER`
- `VLLM_BASE_URL`
- `VLLM_MODEL`
- `VLLM_API_KEY`
- `HF_TOKEN`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MAX_TOOL_ROUNDS`

Fallback LLM :

- `LLM_FALLBACK_GPT`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

STT :

- `WHISPER_MODEL_SIZE`
- `WHISPER_DEVICE`
- `WHISPER_COMPUTE_TYPE`
- `WHISPER_LANGUAGE`

TTS :

- `KOKORO_VOICE`
- `KOKORO_LANG`
- `KOKORO_SPEED`
- `KOKORO_MODEL_PATH`
- `KOKORO_VOICES_PATH`

VAD :

- `VAD_THRESHOLD`
- `VAD_SILENCE_MS`
- `VAD_MIN_SPEECH_MS`
- `VAD_MODEL_PATH`

Conversation :

- `GREETING_TEXT`

### Backend et Gateway

Routage :

- `VOICE_HEALTH_URL`
- `VOICE_WS_URL`
- `VOICE_MONITORING_URL`
- `VOICE_SERVICE_URLS`
- `VOICE_STREAM_URL`
- `GATEWAY_HEALTH_URL`
- `VOICE_GATEWAY_PUBLIC_HOST`

Identité et secrets :

- `INSTANCE_ID`
- `X_API_KEY`
- `OPENAI_API_KEY`

`VOICE_SERVICE_URLS` prend priorité sur les URL historiques.

`NODE_API_KEY` côté Python doit être identique à `X_API_KEY` côté Fastify pour
le contexte vocal et le monitoring.

`VOICE_STREAM_URL` remplace intégralement l'URL `<Stream>` générée. Dans la
topologie documentée, elle doit rester vide ou viser le point d'entrée Fastify
approprié. La faire pointer vers le port Python contourne la sonde, le proxy et
le fallback d'appel OpenAI Realtime.

### Règles de gestion des secrets

- ne jamais versionner les `.env` ;
- ne jamais placer une clé dans le frontend ;
- utiliser une clé distincte pour vLLM ;
- ne jamais journaliser les headers d'autorisation ;
- faire tourner toute clé affichée dans un terminal, un ticket ou un chat ;
- préférer un gestionnaire de secrets en production.

## 16. Démarrage et contrôle préalable

Python 3.11 ou 3.12 est recommandé pour le Voice Server.

```bash
cd voice-server
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[voice,dev]"
```

Kokoro nécessite `espeak-ng`. Silero et Kokoro téléchargent leurs fichiers dans
`~/.cache/mysmartfood-voice/` si aucun chemin local n'est fourni.

Contrôle obligatoire avant un appel :

```bash
voice-preflight
```

Le preflight vérifie séquentiellement :

- démarrage FastAPI léger ;
- inférence Silero ;
- chargement et transcription Whisper ;
- liste des modèles et completion vLLM ;
- synthèse Kokoro.

Puis lancer :

```bash
voice-api
```

Alternative équivalente depuis la racine :

```bash
python -m app.main
```

Vérifications :

```bash
curl http://127.0.0.1:8090/live
curl -i http://127.0.0.1:8090/ready
curl http://127.0.0.1:8090/health
```

## 17. Exécution vLLM

vLLM doit idéalement vivre dans son propre environnement ou conteneur. Il
communique avec Python uniquement en HTTP ; partager le même environnement
Python n'apporte aucun avantage et augmente les conflits de dépendances.

Exemple Llama 3.1 avec appels d'outils :

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

Le chemin exact du template dépend de la distribution vLLM installée. Vérifier
la documentation de la version épinglée.

Contrôle :

```bash
curl http://127.0.0.1:8000/v1/models
```

La production GPU préparée dans `infrastructure/scaleway-vllm/` :

- isole vLLM dans Docker ;
- n'expose pas directement le port 8000 ;
- termine TLS avec Caddy ;
- protège Caddy et vLLM par une clé dédiée ;
- persiste le cache Hugging Face ;
- fournit un benchmark API et tool-calling.

## 18. Observabilité

### Télémétrie Python

`VoiceTelemetry` stocke en mémoire :

- nombre de mesures ;
- moyenne ;
- dernière valeur ;
- minimum ;
- maximum ;
- compteurs d'erreurs ;
- compteurs d'événements.

Étapes mesurées selon le chemin :

- `node`
- `voice_context`
- `vad`
- `stt`
- `llm`
- `tts`

Les mesures repartent de zéro au redémarrage et ne fournissent pas de
percentiles.

### Sessions visibles

`/monitoring` expose au maximum les données suivantes :

- `streamSid`
- `callSid`
- `instanceId`
- étape
- durée
- fournisseur
- dernière erreur technique

Les numéros et contenus conversationnels n'y figurent pas.

### Agrégation Fastify

`GET /api/monitoring`, protégé par JWT administrateur, agrège :

- backend et MongoDB ;
- Gateway ;
- pool de Voice Servers ;
- VAD, STT, LLM et TTS ;
- fournisseur actif ;
- appels actifs fusionnés ;
- circuit breaker OpenAI ;
- métriques HTTP et extraction ;
- alertes récentes.

Une panne de sonde vocale dégrade le snapshot sans faire échouer l'endpoint
admin.

### Frontend

La page `/admin/services` affiche le snapshot toutes les 5 secondes. Elle ne
doit jamais interroger directement le Voice Server ni recevoir une clé interne.

## 19. Modèle de latence

La latence avant la première réponse audible est approximativement :

```text
détection fin de parole
+ file STT
+ inférence STT complète
+ file et inférence LLM complète
+ éventuels appels Fastify et tours LLM supplémentaires
+ file TTS
+ synthèse TTS complète
+ transport Twilio
```

Le silence VAD ajoute par défaut environ 400 ms après la parole.

Le système actuel attend :

- la phrase utilisateur complète avant STT ;
- la transcription complète avant LLM ;
- la completion complète avant TTS ;
- la synthèse complète avant la première trame.

La télémétrie `tts` mesure synthèse et émission des trames, pas la fin de lecture
audible chez l'appelant.

### Méthode d'optimisation

Toute optimisation doit partir d'un scénario reproductible et comparer p50,
p95 et taux d'erreur avant/après.

Ordre recommandé :

1. mesurer séparément VAD, STT, LLM, outils et TTS ;
2. identifier la file dominante ;
3. streamer les tokens LLM vers des phrases ;
4. streamer ou découper le TTS ;
5. introduire un STT partiel si nécessaire ;
6. remplacer l'interpolation audio par un resampler de meilleure qualité si la
   reconnaissance est limitée ;
7. dimensionner les réplicas à partir du test de charge.

Ne pas diminuer arbitrairement `VAD_SILENCE_MS` : une valeur trop faible coupe
les phrases naturelles.

## 20. Capacité et montée en charge

Les principaux goulots d'étranglement sont :

- verrou STT par processus ;
- verrou TTS par processus ;
- GPU et file vLLM ;
- appels d'outils pouvant ajouter plusieurs tours LLM ;
- CPU du VAD et des conversions audio ;
- mémoire des historiques et buffers.

La capacité supérieure à 10 appels simultanés n'est pas certifiée.

Pour augmenter la capacité :

- multiplier les Voice Servers ;
- garder un modèle par réplica ou service d'inférence dédié ;
- utiliser `VOICE_SERVICE_URLS` pour distribuer les appels ;
- dimensionner `VLLM_MAX_NUM_SEQS` avec un benchmark ;
- surveiller le p95 plutôt que le débit seul ;
- limiter les workers FastAPI si chaque worker duplique les modèles.

Un benchmark vLLM seul ne valide ni Whisper, ni Kokoro, ni le transport Twilio.

## 21. Gestion des erreurs

### Message Twilio invalide

Le message est journalisé sous `twilio.bad_message` puis ignoré.

### Contexte restaurant indisponible

L'appel continue avec le contexte générique et
`voice_context_unavailable`.

### Phrase vide ou trop courte

Le tour est ignoré sans réponse.

### Échec STT, LLM ou TTS

- le compteur correspondant est incrémenté ;
- `last_error` est positionné ;
- la session tente de prononcer une réponse générique si cela reste possible.

### Appel d'outil invalide

Le dispatcher retourne un objet `success: false` au LLM. Une fonction inconnue
n'est jamais exécutée.

### Déconnexion

Le nettoyage de session est effectué même sans événement Twilio `stop`.

## 22. Sécurité et frontières de confiance

### Frontières prévues

- Twilio vers Fastify/Gateway : Internet, TLS obligatoire.
- Fastify vers Voice Server : réseau privé.
- Voice Server vers Fastify : réseau privé avec clé interne.
- Voice Server vers vLLM : localhost, réseau privé ou HTTPS avec clé.
- Administrateur vers monitoring : JWT via Fastify uniquement.

### Contrôles présents

- échappement des attributs TwiML ;
- assainissement des identifiants techniques ;
- comparaison constante de la clé `/monitoring` ;
- clé obligatoire pour le contexte vocal ;
- numéro Twilio imposé au tool call ;
- absence de données personnelles dans le monitoring ;
- requêtes MongoDB construites côté backend, sans accès LLM direct ;
- liste fermée d'outils ;
- Caddy et clé vLLM dans la topologie Scaleway.

### Risques actuels à traiter avant exposition publique

1. Les routes `/incoming-call` ne vérifient pas actuellement
   `X-Twilio-Signature`.
2. RESOLU (2026-09-22) : `/media-stream` exige desormais un token HMAC-SHA256
   signe (tenant + expiration) porte par les `customParameters` Twilio,
   verifie en temps constant (`hmac.compare_digest`) dans
   `app/session/bootstrap.py` via `app/stream_token.py`. Une signature absente,
   invalide ou expiree fait echouer la connexion (fail-closed, secret vide
   inclus). La configuration Fastify `verifyClient: callback(true)` n'effectue
   donc plus un accept-all cote securite applicative : le filtrage reel est
   dans la verification du token, pas dans le WebSocket handshake.
3. Les routes Fastify `/api/*/ai/*` ne possèdent pas de middleware interne
   dédié dans leur déclaration actuelle. Le header envoyé par Python ne suffit
   pas s'il n'est pas contrôlé.
4. RESOLU (2026-09-22) : les endpoints FastAPI `/v1/tools/*` exigent desormais
   l'en-tete `x-api-key`, compare a `NODE_API_KEY` en temps constant
   (`secrets.compare_digest`) via `require_internal_api_key` dans
   `app/main.py`. Une cle absente, vide ou invalide renvoie HTTP 401.
5. Le port 8090 ne doit donc jamais être publié directement sur Internet.
6. Le buffer d'une phrase n'a pas de durée maximale ; une parole sans fin peut
   consommer de la mémoire.
7. La taille WebSocket Fastify autorisée, 100 Mo, est élevée pour des trames
   Twilio.
8. La confirmation client et l'appel préalable à la disponibilité reposent sur
   le prompt, pas sur un automate serveur.
9. L'idempotence des créations n'est pas protégée contre une course par une
   contrainte MongoDB atomique.
10. Le contexte menu est une donnée non fiable incorporée au prompt ; la
    consigne anti-injection n'est pas une isolation de sécurité.
11. En mode Gateway multi-instance, `instanceId` sert au contexte vocal mais
    les routes AI de création utilisent `request.instanceId` ou
    `inst_default`. Comme leur enregistrement actuel n'applique pas le hook
    `multiTenantAuth`, une écriture peut viser la mauvaise instance.

Ces points sont documentés ici, mais ne sont pas corrigés par ce document.

## 23. Tests

### Suite Python

```bash
cd voice-server
python -m pytest -q
```

Couverture actuelle :

- codec mu-law et rééchantillonnage ;
- protocole Twilio ;
- normalisation du numéro ;
- message d'accueil ;
- pipeline STT → LLM → TTS ;
- appels d'outils ;
- fallback LLM ;
- correspondance du modèle vLLM ;
- sondes et monitoring privé ;
- nettoyage sur stop, déconnexion et shutdown ;
- scénario WebSocket de bout en bout avec moteurs simulés.

Les tests utilisent des fakes pour l'E2E Python. Ils ne mesurent pas la latence
réelle des modèles ou du réseau Twilio. Aucun test automatisé ne couvre encore
un barge-in réel.

### Tests backend ciblés

Les suites pertinentes couvrent :

- génération TwiML ;
- présence de `/incoming-call` et `/media-stream` ;
- décision Voice Server ou OpenAI ;
- construction d'un pool de cibles ;
- registre des appels sanitizé ;
- agrégation sanitaire ;
- monitoring et alertes.

État actuel du harnais de test : Jest ne démarre pas les suites ciblées car
`babel.config.js` utilise `module.exports` alors que le package est déclaré
ESM. De plus, `Routes/Calls/call.test.js` attend encore une balise `<Say>` alors
que le TwiML produit désormais `<Play>` puis `<Stream>`. Ces problèmes sont
antérieurs à la documentation et doivent être corrigés avant de considérer la
suite Node comme opérationnelle.

### Validation réelle obligatoire

Avant production :

1. exécuter les tests Python ;
2. exécuter les tests backend ciblés ;
3. exécuter le preflight avec les vrais modèles ;
4. vérifier `/ready` à 200 ;
5. effectuer une réservation réelle de test ;
6. effectuer une commande réelle de test ;
7. tester un numéro masqué ;
8. parler pendant la réponse pour tester le barge-in ;
9. arrêter vLLM pour valider le fallback LLM ;
10. arrêter Python pour valider le fallback d'appel ;
11. vérifier qu'aucune donnée personnelle n'apparaît dans logs et monitoring ;
12. réaliser un test de charge avec le même scénario métier.

## 24. Diagnostic

### `/health` retourne `degraded`

1. consulter `components` et `engines.error` ;
2. vérifier `GET /api/ping` ;
3. vérifier le chargement VAD, STT et TTS ;
4. vérifier `GET /v1/models` côté vLLM ;
5. vérifier l'identifiant exact de `VLLM_MODEL` ;
6. vérifier la présence du fallback OpenAI si attendu.

### Fastify affiche `FALLBACK GPT`

1. confirmer que `/ready` retourne 200 ;
2. confirmer que `engines.ready` est vrai ;
3. vérifier les URL construites dans `voiceServiceTargets.js` ;
4. vérifier le timeout de 800 ms ;
5. vérifier l'ouverture du WebSocket interne en moins de 2 secondes.

### Aucun contexte restaurant

1. vérifier `instanceId` dans les paramètres Twilio ;
2. vérifier que `NODE_API_KEY` et `X_API_KEY` correspondent ;
3. appeler manuellement `/api/voice/context/:instanceId` avec la clé ;
4. vérifier la configuration Pricing de l'instance.

### Création impossible

1. identifier `tool:create_appointment` dans les logs ;
2. vérifier le type exact ;
3. vérifier téléphone, date et heure ;
4. vérifier horaires et capacité ;
5. vérifier le résultat Fastify ;
6. ne jamais contourner Fastify depuis Python.

### Latence élevée

1. consulter `telemetry.latency_ms` ;
2. comparer `stt`, `llm` et `tts` ;
3. vérifier le nombre d'appels actifs ;
4. vérifier la file vLLM ;
5. vérifier si un outil ajoute un second tour LLM ;
6. répéter le même scénario avant de conclure.

## 25. Guide de modification

Pour changer le protocole Twilio :

- `backend/Services/twilioServices/twilioServices.js`
- `backend/Routes/Ws/ws.js`
- `voice-server/app/twilio/protocol.py`
- tests TwiML, protocole et E2E.

Pour changer la détection de parole :

- `voice-server/app/vad/silero.py`
- variables `VAD_*`
- tests de session et tests audio réels.

Pour changer le STT :

- interface `voice-server/app/stt/__init__.py`
- implémentation `voice-server/app/stt/faster_whisper.py`
- construction dans `app/main.py`.

Pour changer le fournisseur LLM :

- interface `voice-server/app/llm/__init__.py`
- client `voice-server/app/llm/vllm.py`
- stratégie `voice-server/app/llm/fallback.py`
- preflight, health et tests fallback.

Pour changer les règles conversationnelles :

- `voice-server/app/prompts.py`
- tests métier conversationnels.

Une règle critique ne doit pas rester uniquement dans ce fichier : elle doit
être contrôlée dans Fastify ou dans un automate déterministe.

Pour ajouter un outil :

1. définir son schéma dans `app/llm/tools.py` ;
2. l'ajouter à la liste fermée ;
3. implémenter son dispatch ;
4. ajouter une méthode `NodeApiClient` ;
5. créer un endpoint Fastify authentifié ;
6. placer la logique métier dans le service/contrôleur backend ;
7. ajouter les tests Python et Node ;
8. documenter son idempotence et ses erreurs.

Pour changer le TTS :

- interface `voice-server/app/tts/__init__.py`
- implémentation `voice-server/app/tts/kokoro.py`
- codec et fréquence de sortie ;
- tests de qualité sur un appel réel.

Pour changer le monitoring :

- `voice-server/app/telemetry.py`
- snapshot dans `voice-server/app/main.py`
- `backend/Services/monitoring/`
- `frontend/src/Hooks/Admin/useServiceHealth.js`.

## 26. Limites et dette technique

- Pipeline STT, LLM et TTS non streamé.
- Rééchantillonnage linéaire basique.
- STT et TTS sérialisés par processus.
- Métriques volatiles sans p95/p99.
- Pas de limite de durée d'une phrase ou d'un appel dans le pipeline Python.
- Pas de contrôle de signature Twilio.
- Endpoints internes insuffisamment authentifiés.
- Pas d'automate métier garantissant confirmation et ordre des outils.
- Divergence possible entre horaires statiques du prompt et horaires dynamiques.
- Contrat disponibilité ambigu entre commandes et réservations.
- Idempotence non atomique.
- Isolation multi-instance non garantie sur les endpoints métier AI.
- Pas de repli OpenAI en milieu d'appel après ouverture du proxy Python.
- Fin d'appel local non reliée à l'extraction/persistance de transcription.
- Introduction MP3 dépendante d'une URL GitHub externe codée en dur dans le
  générateur TwiML.
- Compteurs et round-robin locaux au processus.
- Capacité réelle non certifiée au-delà d'un petit nombre d'appels.
- OpenAI encore nécessaire pour les fallbacks.

## 27. Critères de livraison d'une évolution realtime

Une évolution est considérée livrable si :

- ses responsabilités restent dans le bon service ;
- les données externes sont validées ;
- les mutations sont authentifiées et autorisées ;
- les erreurs ont un comportement vocal compréhensible ;
- les sessions sont nettoyées sur tous les chemins ;
- aucune donnée personnelle n'est ajoutée aux logs ou métriques ;
- les tests unitaires et d'intégration passent ;
- le preflight passe avec les vrais moteurs ;
- un appel Twilio réel passe ;
- le fallback correspondant est testé ;
- la latence avant/après est mesurée ;
- ce document et les contrats concernés sont mis à jour.

