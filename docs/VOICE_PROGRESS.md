# Rapport d'avancement — assistant vocal restaurant

Date: 2026-09-19
Perimetre: voice-server + Fastify Twilio. Chiffres GPU non inventes.

## Verdict produit

L'assistant telephonique **n'est pas pret production temps reel**.

- Native (`VOICE_ORCHESTRATOR=native`) est le **seul chemin assez valide** pour rester le defaut.
- Ce n'est **pas** un jugement "Native est architecturalement meilleur que Pipecat".
- Le goulet est le STT : Whisper `small` CPU, WER 1.02, p50 **27096.6 ms**.
- Pipecat 1.11.0 est installe, candidat, **pas** le defaut.

## Ce qui est pret

| Bloc | Etat | Comment tester |
|---|---|---|
| Twilio + proxy Fastify `/twilio/incoming-call` | Code en place | Appel reel vers le webhook Fastify, pas le port 8090 public |
| streamToken + contexte restaurant + numero force | Partage Native/Pipecat | Tests `test_session.py`, `test_pipecat_runner.py` |
| Tools `check_availability` / `create_appointment` | Identiques des deux cotes | Reservation ; le LLM ne peut pas changer le telephone |
| Turn-taking heuristique + hold 800 ms | Mesure (fakes) | "Je voudrais reserver pour..." puis "six personnes" |
| Barge-in confirme 160 ms + epoch | Mesure (fakes) | Parler >160 ms pendant le TTS ; claquement court ignore |
| Hangup Fastify | Native 1x ; Pipecat idempotent | Raccrocher une fois, pas de double persist |
| TTS Kokoro | Executable | Accueil audible si voice-api tourne |

## Ce qui n'est pas pret

| Bloc | Pourquoi | Impact test live |
|---|---|---|
| STT telephone | small CPU ~27 s, WER ~1.0 | L'appel repond trop tard ; ne juge pas le produit |
| A/B Native vs Pipecat live | Pas d'appel Twilio instrumente | Ne change pas le defaut |
| Smart Turn "humain" | Corpus Kokoro coupe non concluant | Ignore pour le choix d'orchestrateur |
| L4 (small CUDA / turbo / Nemotron / Parakeet) | `nvidia-smi` absent | Aucune metrique GPU |
| Modif / annulation | Outils absents | Le bot ne peut pas modifier ou annuler via tool |
| Bruit restaurant soutenu | Gaussien 250 ms interrupt encore 5/5 | Un fond parlant peut encore couper le bot |

## Comment tester maintenant

### 1. Control-plane (recommande sur cette machine)

```bash
cd voice-server
.venv/bin/python -m pytest tests/test_barge_in.py tests/test_interrupt.py tests/test_pipecat_runner.py tests/test_session.py tests/test_session_cleanup.py -q
.venv/bin/python -m benchmarks.benchmark_barge_in
```

Attendu: tests verts. Burst 50 ms ne declenche pas de `clear`. Parole 200 ms oui.

### 2. Sante runtime

```bash
cd voice-server
source .venv/bin/activate
# .env : NODE_API_KEY, TWILIO_AUTH_TOKEN, VLLM_*
voice-preflight
voice-api
```

```bash
curl -s http://127.0.0.1:8090/live
curl -si http://127.0.0.1:8090/ready
curl -s http://127.0.0.1:8090/health
```

`/ready` 200 seulement si Fastify + moteurs + vLLM sont joignables. Sur CPU, `engines.ready` n'implique pas un appel fluide.

### 3. Appel Twilio (tuyau, pas qualite STT)

1. Fastify (`npm run dev` dans `backend/`) + voice-server + vLLM si possible.
2. Webhook Twilio = Fastify `POST /twilio/incoming-call`.
3. Media Stream = proxy Fastify `/v1/:instanceId/media-stream`, **pas** `8090` public.
4. Scenarios utiles meme avec STT lent:
   - Accueil Kokoro entendu.
   - Claquement pendant l'accueil : le bot continue.
   - Phrase >160 ms pendant l'accueil : `interruption_detected` + `clear`, pas d'ancienne voix.
   - "Je voudrais reserver pour..." + pause courte + "six personnes."
   - Confirmation reservation : `create_appointment` avec le numero session.
   - Numero masque : le bot demande le telephone.
5. Ne pas conclure "c'est trop lent donc Native est mauvais" : c'est Whisper small CPU.

### 4. Ce qu'il ne faut pas tester comme critere produit

- Latence fin de phrase → premiere audio (STT domine).
- WER d'un appel CPU.
- Pipecat en defaut.
- Smart Turn sur phrases Kokoro coupees.

## Prochaine etape

La cible n'est plus un A/B STT GPU. C'est Qwen2.5-Omni-7B GPTQ-Int4 en speech-to-speech.

1. Fenetre de mesure sur la L4: arreter Whisper / vLLM / Kokoro (confirmation operateur, la ligne est live).
2. Executer `python -m prototypes.omni_s2s.proto_s2s` depuis `voice-server`.
3. Lire `benchmarks/results/omni_s2s_probe.json`: VRAM, latence audio->audio, francais, tool calls thinker.
4. Seulement si GO: Twilio streaming + interruptions, puis outils Fastify. Supprimer l'ancien pipeline. Pas deux orchestrateurs.

Delete-zone (ne pas recreer comme cible): A/B Whisper turbo / Parakeet / Nemotron / Smart Turn comme moteur conversationnel.

Sources: `docs/VOICE_AUDIT.md` section 19, `benchmarks/results/omni_s2s_probe.json`.
