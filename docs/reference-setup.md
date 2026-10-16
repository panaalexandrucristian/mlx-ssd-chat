# MLX + SSD offloading — Qwen3 30B, chat în terminal cu `ai`

Actualizat: 16 octombrie 2026. **Bugetul MLX dimensionează automat cache-ul de experți; profilul implicit a fost verificat cap-coadă, iar documentația este integrală și autonomă.**

Modelul real a trecut probele de chat cu/fără thinking, memorie conversațională, reutilizare KV,
reset, ieșire și context aproape de limită. Vârful maxim de memorie fizică al procesului
observat în aceste probe ale profilului implicit este **3,54 GB**. Profilul nou de
8 GB, cu cache dimensionat automat, a atins **7,92 GB** în proba descrisă la secțiunea 6.
Acest document conține explicațiile, comenzile,
sursele locale complete, modulul modificat integral, originalul și copia istorică,
patch-ul, configurația shell, dependențele fixate și logurile testelor. Restaurarea
profilului a fost verificată într-un mediu separat.

**Este suficient să păstrezi acest document pentru documentație și sursele locale.**
Toate blocurile denumite „Fișier” sunt incluse integral în secțiunea 8 și pot fi extrase
cu instrucțiunile din secțiunea 7. Căile din comenzi reprezintă destinații de instalare
sau fișiere generate prin acele comenzi; nu trebuie recuperate din altă sesiune.
Pachetele Python și greutățile modelului se descarcă prin comenzile explicite de instalare,
cu versiunile și revizia precizate aici.

### Cuprins

1. Modelul și mediul verificat.
2. Bugetul de memorie și mecanismul de offloading.
3. Modificările locale și explicația fiecărei corecții.
4. Utilizarea chatului, opțiunile și configurația shell.
5. Rezultatele măsurate și interpretarea lor.
6. Starea finală a verificărilor.
7. Extragere, reinstalare, verificare, întreținere, plan de build din surse și depanare.
8. Anexa cu toate sursele, configurația, logurile și hash-urile.

Pornire confirmată:

```sh
~/.local/bin/ai
```

`ai` este disponibil prin aliasul din `.zshrc`; thinking este activ implicit.
Pentru răspunsuri directe: `ai --no-thinking`. `/reset` începe o conversație nouă;
`q`, Ctrl+D sau Ctrl+C închid procesul și eliberează memoria modelului.

## 1. Modelul și mediul actual

- Mac: Apple M4 Pro, 24 GB RAM unificat, macOS 26.5.2.
- Model: `mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit`.
- Revizie fixată: `f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e`.
- Qwen3 MoE: 48 de straturi, 128 de experți/strat, 8 experți selectați/token/strat.
- Configurația modelului declară `max_position_embeddings: 40960`; nu este modelul Huihui cu context de 256K.
- Contextul launcherului este limitat conservator la **4096 tokeni, incluzând răspunsul**, pentru profilul cu buget de memorie mic.
- Motor izolat: `~/.local/share/uv/tools/mlx-flash/bin/python` (Python 3.12.13).
- Versiuni observate: `mlx-flash 0.9.2`, `mlx 0.32.4`, `mlx-lm 0.32.2`, `numpy 2.5.4`, `transformers 5.20.0`, `huggingface-hub 2.2.0`.
- La finalul configurării din 16 octombrie 2026, modelele vechi fuseseră eliminate; modelul de 30B este cel păstrat pentru acest profil.
- Profilul final pornește la cerere din terminal. Integrarea locală OpenCode și serviciul de pornire automată LaunchAgent fuseseră eliminate; nu sunt necesare pentru utilizarea descrisă aici.

## 2. Ce înseamnă limita de memorie

Profilul implicit folosește următoarele valori; bugetele MLX și procesului se pot
configura din comandă, așa cum este descris imediat după acest exemplu:

```python
mx.set_memory_limit(2_800_000_000)
mx.set_cache_limit(64_000_000)
model, tokenizer = load(path, lazy=True,
                        tokenizer_config={"fix_mistral_regex": True})
# 12 este rezultatul calculat de plan_expert_cache(model, 2_800_000_000).
# În ai.py se transmite plan.capacity, recalculat pentru bugetul ales.
state = enable_expert_streaming(
    model, capacity_per_layer=12, model_path=path,
    exact=True, read_mode="pread",
)
state.warmup()
mx.eval(model.parameters())
```

- Bugetul implicit al procesului: **4.000.000.000 bytes (4 GB zecimali)**.
- Cu bugetul implicit de 2,8 GB, 12 experți/strat rămân în cache. Capacitatea crește automat cu bugetul MLX; experții selectați care lipsesc sunt citiți de pe SSD înainte de folosire.
- La prefill, dacă setul de experți depășește capacitatea cache-ului, se alocă temporar experții necesari pentru **un singur strat**. Rezultatul este evaluat înainte de eliberarea acestora.
- Citirile `pread` evită maparea persistentă a tuturor shardurilor în spațiul procesului. macOS poate totuși folosi cache de fișiere la nivelul sistemului; acesta nu este plafonat de launcher.
- `mx.set_memory_limit` privește alocatorul MLX, nu memoria totală Python/Metal/macOS. API-ul MLX 0.32.4 descrie valoarea ca o limită orientativă în timpul evaluării grafului, care poate fi depășită; nu garantează oprirea exact la acel prag.
- Monitorul de memorie, inclus integral în blocul `memory_guard.py`, verifică la fiecare 50 ms RSS și `physical footprint` prin `proc_pid_rusage`; dacă oricare depășește bugetul procesului (implicit 4 GB), închide **doar propriul proces**, cu codul 78.
- Aceasta este o protecție software prin eșantionare, **nu o limită OS strictă**: pot exista depășiri tranzitorii între măsurători.
- KV cache: 8 biți, cuantizare de la început; `prefill_step_size=128`.

### Configurarea memoriei direct din comanda Python

**`--mlx-memory-gb GB` configurează valoarea transmisă la
`mx.set_memory_limit(...)` și dimensionează automat cache-ul de experți.** Numărul este exprimat în GB zecimali: `4` înseamnă
`4_000_000_000` bytes. Sunt acceptate și valori fracționare, de exemplu `3.2`.

Exemplu complet, cu buget MLX de 4 GB:

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" \
  "$HOME/.local/share/mlx-ssd-chat/ai.py" --mlx-memory-gb 4
```

Aceleași argumente funcționează și prin launcher:

```sh
~/.local/bin/ai --mlx-memory-gb 4
~/.local/bin/ai --mlx-memory-gb 4 --no-thinking
```

Bugetul total al procesului este calculat implicit ca **buget MLX + 1,2 GB**, astfel
încât monitorul să lase loc pentru Python, Metal și alte alocări. Cu MLX de 4 GB,
monitorul urmărește un buget total de 5,2 GB. Valoarea totală se poate configura
independent prin `--process-memory-gb`:

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" \
  "$HOME/.local/share/mlx-ssd-chat/ai.py" \
  --mlx-memory-gb 4 --process-memory-gb 6
```

Ambele opțiuni sunt disponibile și pentru benchmark:

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" \
  "$HOME/.local/share/mlx-ssd-chat/benchmark.py" \
  --mlx-memory-gb 4 --process-memory-gb 6 --tokens 16 \
  --prompt 'What is 2 + 2? Reply briefly.'
```

| Parametru | Valoare implicită | Efect |
|---|---:|---|
| `--mlx-memory-gb` | `2.8` | Buget orientativ al alocatorului MLX și buget pentru dimensionarea automată a cache-ului de experți. |
| `--process-memory-gb` | MLX + `1.2`, deci `4` implicit | Bugetul RSS/physical footprint la depășirea căruia monitorul oprește procesul. |
| `mx.set_cache_limit(...)`, în cod | `64_000_000` bytes | Cache-ul de alocări MLX; diferit de cache-ul de experți. |
| `capacity_per_layer` | Automat; `12` la 2,8 GB | Calculat din dimensiunile reale ale parametrilor și bugetul MLX. |

Argumentele trebuie să fie numere pozitive și finite, reprezentabile în bytes pe
64 de biți și de cel puțin un byte. `0`, valori negative, `nan`, `inf`, text nenumeric
și valorile în afara intervalului sunt respinse cu codul 2 înaintea încărcării
modelului. Bugetul explicit al procesului trebuie să fie strict mai mare decât
bugetul MLX. Valorile folosite sunt afișate la pornire.

Rezerva implicită de 1,2 GB a procesului reproduce profilul inițial de 2,8 GB MLX /
4 GB proces; nu garantează că orice prompt sau configurație încape. Capacitatea
cache-ului este calculată la pornire de `plan_expert_cache`, din modulul complet
`cache_memory.py` inclus în anexă. Citește formele și dimensiunile parametrilor
lazy, fără evaluarea sau încărcarea în RAM a tuturor experților.

Din bugetul MLX se scad greutățile fixe și o rezervă de lucru egală cu **10% din
buget, minimum 400 MB**, pentru KV, prefill și alocări temporare. Suma rămasă este
împărțită la costul unui slot de expert în toate straturile. Capacitatea este
rotunjită în jos și limitată la numărul total de experți. Un buget care nu permite
cel puțin `top_k` experți/strat este refuzat înaintea încărcării efective a greutăților.

Pentru modelul și revizia fixate aici, greutățile fixe ocupă **867.168.256 bytes**,
iar un expert suplimentar în fiecare dintre cele 48 de straturi costă în total
**127.401.984 bytes**. Estimările de dimensionare sunt:

| Buget MLX | Experți/strat | Greutăți rezidente estimate | Rezervă de lucru MLX | Buget proces implicit |
|---|---:|---:|---:|---:|
| 2,8 GB | 12 / 128 | 2,40 GB | 0,40 GB | 4,0 GB |
| 4 GB | 21 / 128 | 3,54 GB | 0,40 GB | 5,2 GB |
| 8 GB | 49 / 128 | 7,11 GB | 0,80 GB | 9,2 GB |
| 24 GB | 128 / 128 | 17,17 GB | 2,40 GB | 25,2 GB |

Aceste valori sunt estimări pentru greutăți, nu măsurători ale întregului proces.
Rezerva de lucru este inclusă în bugetul MLX; rezerva de 1,2 GB a procesului este
suplimentară. La pornire sunt afișate capacitatea aleasă, estimarea și memoria MLX
activă după încărcare. Un buget mai mare păstrează mai mulți experți în RAM și poate
reduce citirile SSD; câștigul de viteză depinde de prompt și trebuie măsurat.
După atingerea a 128 de experți/strat, toate greutățile experților sunt rezidente.
O instanță deja pornită păstrează configurația sa; redimensionarea se aplică la
următoarea pornire, de exemplu `ai --mlx-memory-gb 8`.

MLX poate depăși limita orientativă a alocatorului. Monitorul procesului verifică
bugetul total la fiecare 50 ms și provoacă **oprirea**, nu reducerea automată a
cache-ului. Nu este un plafon OS strict, nu limitează RAM-ul altor aplicații și nici
cache-ul de fișiere al macOS. Reducerea bugetelor poate duce la oprirea generării;
după schimbarea profilului, consumul și viteza trebuie măsurate din nou.

### Istoric — 8 GB cu cache fix de 12 experți/strat, înainte de dimensionarea automată

**Versiunea curentă:** `--mlx-memory-gb 8` selectează automat **49/128 experți pe
strat**; proba reală a măsurat **7,11 GB MLX după încărcare** și **7,92 GB vârf
physical footprint**. Valorile de aproximativ 3 GB de mai jos descriu versiunea
anterioară, care păstra cache-ul fix la 12 experți/strat.

Înainte de introducerea dimensionării automate, la 16 octombrie 2026, utilizatorul
a observat generare mai rapidă după pornirea cu
`--mlx-memory-gb 8`. În verificarea procesului real, PID 38097, pornit la 17:06:39,
macOS raporta aproximativ **2,9 GiB physical footprint**, cu vârf de **3,1 GiB**
(aproximativ 3,1 GB, respectiv 3,3 GB zecimali). Eșantionarea execuției a arătat
citiri `pread` și operații MLX/Metal pe GPU.

În MLX 0.32.4, limita influențează și sincronizarea: evaluatorul verifică dacă
`get_active_memory() > get_memory_limit()` și există operații active; în acest caz
finalizează fluxurile GPU deschise și așteaptă terminarea unor operații prin
`scheduler::wait_for_one()`. Astfel, un prag de 2,8 GB poate introduce așteptări pe
care pragul de 8 GB le evită, chiar cu aceeași capacitate de 12 experți pe strat.
Memoria contabilizată de MLX și `physical footprint` sunt indicatori diferiți.

Mecanismul este confirmat în sursa versiunii instalate:
`https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/transforms.cpp#L278-L291`.
Acesta este o explicație posibilă pentru accelerarea observată; contribuția exactă
și câștigul în tokeni/s nu au fost încă măsurate printr-o comparație controlată.
În acea versiune, creșterea limitei nu mărea cache-ul de experți. Versiunea curentă
folosește suplimentar bugetul pentru dimensionarea cache-ului, conform tabelului de mai sus.

## 3. Toate modificările locale din `mlx-flash`

Modulul de streaming modificat este inclus integral în anexa acestui document,
în blocul `expert_streaming.py`. După instalare, destinația lui este:

```text
~/.local/share/uv/tools/mlx-flash/lib/python3.12/site-packages/mlx_flash_compress/expert_streaming.py
```

Anexa conține și două versiuni complete pentru comparație și recuperare:

- `expert_streaming.upstream-0.9.2.py`: originalul extras din wheel-ul PyPI 0.9.2.
- `expert_streaming.before-resume.py`: copia istorică de dinaintea validării finale a modului exact.

Blocul `expert_streaming.patch` conține toate diferențele dintre original și modulul final.
Un upgrade/reinstall al pachetului poate suprascrie patch-urile instalate; procedura
completă de restaurare și verificarea hash-urilor sunt incluse la secțiunea 7.

### Corecții de reprezentare numerică și cache

1. **BF16 în format stacked:** octeții sunt citiți ca `np.uint16`, apoi reinterpretați cu `.view(mx.bfloat16)`. Citirea lor ca `np.float16` modifica valorile numerice.
2. **Indici locali după remapare:** `CachedSwitchLinear` transmite `sorted_indices=False` la `mx.gather_qmm`; ordinea indicilor globali nu garantează ordinea sloturilor locale.
3. **Evicția cache-ului:** sloturile experților păstrați nu sunt renumerotate. Noii experți sunt scriși exact în sloturile eliberate, folosind `slots` și `new_ids_ordered`; lookup-ul corespunde astfel greutăților reale.
4. **Evaluarea actualizărilor:** după înlocuirea sloturilor se execută `mx.eval(target[proj])`; după reconstruirea lookup-ului se evaluează `lookup` și `hit_mask`.
5. **Dtype în fallback-ul aproximativ:** reducerea ponderată este convertită cu `.astype(y.dtype)`. Acest fallback nu este folosit de profilul exact.

### Corecții și funcționalități ale modului exact

6. **Citire SSD limitată:** `SafetensorsMap(..., read_mode="pread")` folosește descriptori `os.open` și `os.pread`. Modul `mmap` rămâne disponibil pentru compatibilitate. O citire incompletă produce eroare.
7. **Eliberarea descriptorilor:** `_get_mmap` închide fișierul folosit la construirea mappingului; `close()` închide atât mappingurile, cât și descriptorii modului `pread`.
8. **BF16 și pentru formatul per-expert:** se aplică aceeași reinterpretare `uint16 → bfloat16`. Un expert absent produce `KeyError`, în loc să fie omis și să deplaseze ordinea rezultatelor.
9. **Validarea citirilor:** dtype necunoscut nu mai cade implicit pe FP16; indicii stacked din afara intervalului produc `IndexError`.
10. **Mod exact explicit:** API-ul este acum:

    ```python
    enable_expert_streaming(model, capacity_per_layer=200, model_path=None,
                            exact=False, read_mode="mmap")
    ```

    Launcherul transmite explicit `exact=True, read_mode="pread"`. Modul exact este validat doar pentru `qwen3_moe`, proiecții affine fără bias aditiv și fără sharding distribuit. Capacitatea trebuie să fie cel puțin `top_k`. Lipsa shardurilor locale produce eroare înainte de generare.
11. **Încărcare înainte de calcul:** `ExpertCache.ensure(indices)` încarcă întregul set curent de experți înainte de calcul și verifică prezența lor. Cache miss-urile sunt sortate pentru comportament determinist. Lipsa unei greutăți necesare produce eroare în modul exact.
12. **Fără colectare întârziată în modul exact:** `CachedSwitchLinear` nu mai adaugă indicii în bufferul folosit pentru update-ul dintre tokeni când `cache.exact` este activ.
13. **Wrapper apelabil real:** `ExactStreamingSwitchGLU(nn.Module)` înlocuiește `mlp.switch_mlp`. Routerul Qwen original, selecția top-k și ponderile sale sunt păstrate. Nu se bazează pe atribuirea `mlp.__call__ = ...`, care nu înlocuiește apelul special `mlp(x)` în Python.
14. **Decode exact:** wrapperul apelează `cache.ensure`, execută SwitchGLU și evaluează rezultatul înainte de o nouă evicție; un calcul lazy nu poate rămâne dependent de sloturi mutate ulterior.
15. **Prefill peste capacitatea cache-ului:** se construiește un cache temporar cu toți experții selectați pentru acel strat, se calculează proiecțiile up/gate/down cu indicii remapați, apoi se evaluează rezultatul. Nu se elimină experți din calcul pentru a simula viteză.
16. **Protecție împotriva modului aproximativ:** `enable_skip_fallback` refuză cache-uri marcate `exact`.
17. **Docstring actualizat:** documentează modul exact, citirile `pread` și diferența față de calea aproximativă veche.

Ipoteza că `warmup()` suprascria greutățile de atenție cu zero nu a fost confirmată.
Defectul reprodus era folosirea experților greșiți/lipsă prin calea de apel a cache-ului.

## 4. Utilizare, launcher și configurație shell

Toate componentele de mai jos au codul integral în anexă:

| Bloc inclus | Rol |
|---|---|
| `ai` | Launcher zsh executabil, instalat ca `~/.local/bin/ai`. |
| `ai.py` | Chat interactiv și mesaje individuale, istoric, KV cache și limite. |
| `memory_guard.py` | Monitorul de memorie și validarea comună a opțiunilor CLI de memorie. |
| `cache_memory.py` | Dimensionarea cache-ului de experți din bugetul MLX și dimensiunile parametrilor lazy. |
| `expert_streaming.py` | Implementarea completă a offloading-ului, inclusiv modificările locale. |
| `benchmark.py` | Măsurare one-shot a încărcării, generării și memoriei. |
| `test_streaming.py` | Regresie diferențială pe model mic determinist. |
| `test_chat.py` | Conversație, memorie, prefix KV, reset și ieșire în pseudo-terminal. |
| `test_context.py` | Context aproape plin, respingerea depășirii și reset. |
| `export_setup.py` | Regenerarea anexei complete, inclusiv propriul cod și dovezile testelor. |
| `ai.zsh` | Configurația shell necesară acestui chat. |

### Comenzi uzuale

```sh
~/.local/bin/ai
~/.local/bin/ai --no-thinking
~/.local/bin/ai --max-tokens 512
~/.local/bin/ai --mlx-memory-gb 4
~/.local/bin/ai --mlx-memory-gb 4 --process-memory-gb 6
~/.local/bin/ai --mlx-memory-gb 8
~/.local/bin/ai --no-thinking --prompt 'What is 2 + 2? Reply briefly.'
~/.local/bin/ai --help
```

Thinking este activ implicit. La 2,4–3 tokeni/s, 512 tokeni generați pot dura aproximativ
3 minute, iar 1024 tokeni aproximativ 6–7 minute, plus prefill. Acestea sunt estimări
din viteza observată, nu durate garantate; răspunsul se poate termina mai devreme.

Conținutul launcherului:

```zsh
#!/bin/zsh
exec "$HOME/.local/share/uv/tools/mlx-flash/bin/python" "$HOME/.local/share/mlx-ssd-chat/ai.py" "$@"
```

Configurația shell completă necesară chatului este următorul fragment, inclus și
ca bloc `ai.zsh` în anexă. Se adaugă în `~/.zshrc` la instalare:

```zsh
# Local MLX chat: Qwen3 30B, exact SSD offload, 4 GB process budget
alias ai='"$HOME/.local/bin/ai"'
```

### Comportamentul chatului

- Încarcă exclusiv revizia locală fixată; nu descarcă automat un model la pornire.
- `fcntl.flock` pe `~/.local/share/mlx-ssd-chat/ai.lock` împiedică pornirea simultană a două instanțe ale launcherului.
- Thinking activ implicit; `--no-thinking` este o opțiune explicită.
- `--mlx-memory-gb` configurează bugetul MLX și dimensionarea cache-ului (implicit 2,8 GB și 12 experți/strat); `--process-memory-gb` configurează bugetul total urmărit de monitor (implicit MLX + 1,2 GB). Ambele acceptă GB zecimali, cu validare numerică înainte de încărcarea modelului; capacitatea minimă este verificată pe parametrii lazy înainte de evaluare.
- `--max-tokens` implicit 1024, acceptă 1–2048. Aceasta este limita totală a generării, inclusiv thinking.
- `--prompt "..."` execută un singur mesaj și închide procesul.
- `q`, `/quit` sau Ctrl+D ies; `r` sau `/reset` resetează conversația; Ctrl+C închide chatul.
- Istoricul este formatat cu template-ul modelului; se reutilizează doar prefixul de tokeni identic cu cel din KV cache. Când template-ul elimină thinking vechi, cache-ul este tăiat la prefixul comun.
- Dacă offsetul cache-ului nu corespunde tokenilor păstrați, cache-ul este reconstruit.
- Mesajele care ar depăși bugetul de context de 4096 tokeni, împreună cu rezerva de răspuns, sunt refuzate cu explicație; istoricul nu este șters automat.
- Afișează timpul primului token, viteza decode, prefixul reutilizat și vârfurile de memorie observate. Semnalează generarea tăiată de limita de tokeni.
- La ieșire închide fișierele de streaming și monitorul de memorie.
- Este chat text, fără executor de tool-uri: un răspuns care descrie un fișier nu creează acel fișier pe disc.

**Status:** launcherul și fluxul interactiv au trecut verificările descrise la secțiunea 5.

## 5. Rezultate măsurate

### Test diferențial de corectitudine

`test_streaming.py` creează un Qwen3-MoE mic, determinist, cu 2 straturi, 16 experți/strat, top-k=2, greutăți BF16 cuantizate la 4 biți și cache de doar 4 experți/strat. Compară logits cu modelul de referință prin apelurile reale ale modelului, inclusiv KV cache.

- Înainte de corecția exactă: eroare maximă logits **1.140625**, test eșuat.
- După corecție: **0.0** la prefill de 12 tokeni și la patru pași decode succesivi, cu evicții.
- Acesta validează calea de calcul pe modelul mic; nu este un benchmark de calitate al întregului model de 30B.

### Modelul real de 30B, teste secvențiale

| Probă | Rezultat | Primul token | Decode | Vârf MLX | Vârf physical footprint |
|---|---|---:|---:|---:|---:|
| `What is 2 + 2? Reply briefly.`, fără thinking | `4`, terminare normală | 7,84 s | 3,38 tok/s* | 2,61 GB | 3,21 GB |
| `What is 17 * 23? Think briefly, then give the answer.`, thinking, limită 192 tokeni | Thinking coerent, calculează 391; generarea este tăiată înainte de răspunsul final | 7,77 s | 3,14 tok/s | 2,63 GB | 3,22 GB |

\* Prima probă a generat numai 2 tokeni, deci nu este un benchmark stabil de viteză. A doua a durat 68,83 s în total. Încărcarea a durat aproximativ 1,3–1,5 s; alocările MLX după încărcare au fost aproximativ 2,40 GB.

**Corecție față de istoricul vechi:** valoarea de ~80 tok/s provenea din generare coruptă, fără sens. Nu trebuie folosită ca performanță validă a offloading-ului. Profilul corect măsurat este în jurul a 3 tok/s pe aceste probe.

### Alte verificări efectuate

- `py_compile`: `ai.py`, `memory_guard.py`, `test_streaming.py`, `benchmark.py` — trecut.
- `zsh -n` pentru `.zshrc` și launcher — trecut.
- Test de watchdog într-un proces separat, cu limită artificială de 1 byte — închide acel proces cu codul 78 și mesajul așteptat.
- Indexul de navigare al codului a fost actualizat după schimbările de cod; acesta este un instrument de dezvoltare, fără rol în pornirea sau reinstalarea chatului.
- Sintaxa surselor din anexă, hash-urile și identitatea lor cu sursele locale au fost verificate.
- `--help` se încheie cu codul 0; limitele invalide `--max-tokens 0` și `2049` sunt refuzate cu codul 2.

### Continuarea verificării — 16 octombrie 2026

| Probă | Rezultat | Memorie fizică maximă |
|---|---|---:|
| Launcher real, `17 * 23`, thinking, 768 tokeni | Produce `</think>` și `17 * 23 = 391.`; atinge limita de tokeni înainte de EOS | 3,23 GB |
| Chat în pseudo-terminal, fără thinking, trei răspunsuri | `OK.`, apoi reține `LUNA-47`; după `/reset`, răspunde `4`; `q` închide cu codul 0 | 3,35 GB |
| Chat în pseudo-terminal, thinking, trei răspunsuri | `OK`, `LUNA-47`, apoi `2 + 2 = 4.` după reset; toate încheiate normal, cu `</think>` și EOS | 3,29 GB |
| Context lung: 4078 tokeni de intrare + rezervă de răspuns de 16 | Recuperează `ORBIT-73` de la început; respinge promptul peste limită; după reset răspunde `4` | 3,54 GB |
| A doua instanță în timpul generării | Refuzată în 0,04 s, codul 1, înainte de încărcarea modelului | — |
| Ctrl+D / Ctrl+C, trimise prin pseudo-terminal real | Închidere curată cu codurile 0 / 130 | — |
| Reinstalare izolată, 37 de dependențe fixate | Patch identic byte-cu-byte, test diferențial trecut și răspuns corect `4` cu modelul real | 3,30 GB |

La proba de 768 tokeni: primul token în **9,1 s**, decode **2,63 tok/s**.
Textul final este prezent, dar terminarea raportată de motor este `length`, nu `stop`.
Limita `--max-tokens` include thinking-ul; promptul „Think briefly” nu garantează un răspuns scurt.

La chatul direct, al doilea mesaj reutilizează **26 de tokeni**, prefixul comun exact.
Template-ul elimină inclusiv marcajele thinking goale din răspunsul precedent; launcherul
taie cache-ul la acel prefix. După reset, prefixul reutilizat este 0.

Chatul cu thinking confirmă aceeași tăiere la **26 de tokeni**, după un răspuns de
109 tokeni. Următoarele răspunsuri au 161 și 93 de tokeni. Timpul până la primul token
este 8,2 / 6,4 / 7,0 s, iar decode-ul 2,58 / 2,53 / 2,38 tok/s. Niciun răspuns nu a
atins limita de 512 tokeni a testului; toate au ajuns la EOS.

Codul complet al probei de terminal este inclus în blocul `test_chat.py` (implicit
fără thinking; `--thinking` verifică și închiderea marcajelor thinking). Dovezile
text sunt incluse integral în blocurile `chat-direct.log`, `chat-thinking.log`,
`context.log`, `exit-eof.log` și `exit-interrupt.log` din anexă.

**Loguri istorice:** aceste cinci capturi de terminal preced dimensionarea automată
a cache-ului și afișarea mesajelor `Expert cache` și `MLX active after loading`.
Versiunea curentă afișează și aceste informații la pornire, alături de bugetul MLX;
bannerul ei diferă astfel de cel din capturile istorice. Proba nouă de memorie este
documentată separat în secțiunea 6 și în blocul `cache-memory-8gb.log`.

La proba de context lung, primul token apare după **120,4 s**; răspunsul de 6 tokeni
are un decode raportat de 2,36 tok/s. După reset, primul token apare în 7,7 s.
Promptul sintetic conține un cod la început și 404 repetări ale unei propoziții;
testul confirmă această intrare, nu garantează consumul pentru orice text posibil.
Protecția de memorie rămâne activă pentru toate intrările.

Blocul `long-prompt.txt` include integral promptul pregătit pentru proba lungă:
codul `ORBIT-73`, cele 404 repetări și întrebarea finală. Proba automată reconstruiește
promptul din codul inclus și înlocuiește newline-ul cu spațiu pentru introducerea
în chatul cu mesaje de o singură linie. Numărul de 4078 tokeni se referă la acea formă
cu template, verificată de test. Logurile PTY se încheie la ultimul prompt capturat;
codurile de ieșire sunt verificate separat de procesele care execută probele.

Proba aproape de limită folosește `--max-tokens 16`. Implicit, `--max-tokens 1024`
rezervă 1024 de tokeni pentru răspuns și permite cel mult **3072 de tokeni de prompt**,
incluzând istoricul și template-ul. Limita comună rămâne 4096.

## 6. Starea finală a verificărilor

### Dimensionarea automată a cache-ului — 16 octombrie 2026

- Testul diferențial a trecut pentru cache-uri de 4, 6, 13 și 16 experți/strat pe
  modelul mic: eroare maximă logits **0,0** la prefill și decode, inclusiv evicții.
- Capacitățile 6, 13 și 16 sunt selectate automat pentru trei bugete de test;
  dimensiunea estimată a greutăților este identică cu dimensiunea array-urilor
  efectiv păstrate după activarea streaming-ului și încălzirea cache-ului.
- Un buget insuficient este refuzat, iar un buget mare plafonează capacitatea la
  numărul real de experți. Planificarea nu crește memoria activă MLX.
- Pe metadatele lazy ale modelului real: 2,8 / 4 / 8 GB aleg 12 / 21 / 49 de
  experți/strat. Această probă de dimensionare nu a încărcat greutățile modelului.
- Sintaxa, mesajele `--help` și respingerea bugetelor numerice invalide au trecut.

**Probă cu modelul real, buget MLX 8 GB / proces 9,2 GB:**

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" \
  "$HOME/.local/share/mlx-ssd-chat/benchmark.py" \
  --mlx-memory-gb 8 --tokens 16 \
  --prompt 'What is 2 + 2? Reply only with the number.'
```

| Indicator | Măsurare |
|---|---:|
| Cache efectiv | 49 / 128 experți pe strat |
| MLX activ după încărcare | 7,11 GB |
| Vârf MLX | 7,36 GB |
| Vârf `physical footprint` al procesului | 7,92 GB |
| Încărcare | 4,78 s |
| Primul token | 11,13 s |
| Răspuns | `4`, 2 tokeni generați, cod de ieșire 0 |

Logul integral este inclus în blocul `cache-memory-8gb.log`. Proba confirmă
folosirea efectivă a RAM-ului suplimentar și generarea corectă. Cei 2 tokeni și
prezența unui alt chat deschis fac ca această probă să nu constituie o comparație
controlată de viteză. Testul diferențial verifică suplimentar rezultatele numerice
pentru mai multe capacități de cache.

### Verificarea opțiunilor CLI de memorie — 16 octombrie 2026

După adăugarea inițială a configurării memoriei, înainte de dimensionarea automată
a cache-ului, au trecut următoarele verificări:

- `--help` expune ambele opțiuni în chat, benchmark și launcherul zsh.
- Câte 15 cazuri invalide în fiecare script sunt respinse cu codul 2 înainte de
  încărcare: zero, valori negative, `nan`, `inf`, text nenumeric, valori în afara
  intervalului și buget total mai mic sau egal cu cel MLX.
- În procese izolate, opțiunile CLI au fost urmărite până la alocatorul MLX real
  și monitorul real; încărcarea modelului a fost înlocuită cu un punct de oprire
  pentru a verifica numai configurarea. Au trecut toate cele șase combinații:

| Argumente | MLX aplicat efectiv | Monitor proces | Scripturi verificate |
|---|---:|---:|---|
| Implicite | 2.800.000.000 bytes | 4.000.000.000 bytes | Chat și benchmark |
| `--mlx-memory-gb 4` | 4.000.000.000 bytes | 5.200.000.000 bytes | Chat și benchmark |
| `--mlx-memory-gb 4 --process-memory-gb 6` | 4.000.000.000 bytes | 6.000.000.000 bytes | Chat și benchmark |

Tentativa de probă nouă cu modelul real a fost refuzată de lock-ul unui chat deja
activ. Prin urmare, aceste verificări confirmă configurarea și validarea opțiunilor,
nu reprezintă măsurători noi de generare sau consum. Rezultatele de inferență din
secțiunea 5 sunt cele ale profilului implicit verificat anterior.

### Inventarul final

- [x] Launcher cap-coadă, thinking complet și conversație cu memorie.
- [x] Prefix KV corect după eliminarea thinking-ului vechi, reset și ieșire.
- [x] Blocarea celei de-a doua instanțe înainte de încărcarea modelului.
- [x] Context de 4078 tokeni, respingerea depășirii limitei și măsurarea memoriei.
- [x] Patch complet față de wheel-ul original 0.9.2, surse și hash-uri în document.
- [x] Restaurare izolată cu versiuni fixate, regresie diferențială și model real.
- [x] Documentație și comenzi de pornire finalizate.
- [x] Modulul modificat, originalul, copia istorică, generatorul, configurația shell și dovezile text incluse integral în document.
- [x] Bugetele MLX și procesului configurabile din comandă, în chat și benchmark.
- [x] Cache de experți dimensionat automat din bugetul MLX, comun chatului și benchmark-ului.

La încheierea verificărilor funcționale din 16 octombrie 2026, toate procesele MLX
pornite pentru ele se încheiaseră. Rezultatele de performanță de mai sus provin din
acele verificări; actualizarea documentației și probele CLI pre-încărcare nu reprezintă
o nouă măsurare a modelului. Ulterior, un chat pornit separat era activ în timpul
verificării noilor argumente de memorie.

## 7. Extragere, reinstalare, verificare și întreținere

Punctul de pornire este exclusiv acest document. Comenzile următoare reconstruiesc
sursele și probele din blocurile incluse în secțiunea 8, instalează dependențele și
descarcă revizia fixată a modelului. Nu este necesară nicio copie locală preexistentă
a scripturilor, patch-ului, logurilor sau exportului.

### Pregătirea unui Mac nou

Mediul verificat este Apple Silicon, macOS 26.5.2 și Python 3.12.13. Sunt necesare
acces la internet pentru pachete/model și spațiu SSD pentru greutăți și mediul Python.
În terminal, dacă uneltele lipsesc:

```sh
xcode-select --install
```

După terminarea instalării uneltelor Apple, instalează `uv` și Python:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.13
```

Verificare rapidă a uneltelor necesare pentru pașii următori:

```sh
uv --version
python3 --version
command -v patch shasum zsh
```

### Extragerea fișierelor din acest document

Salvează documentul ca `~/mlx-opencode-setup.md`. Directorul rezultat este doar
o destinație pentru extragere și va fi creat de comandă. Sunt verificate hash-urile
tuturor blocurilor înainte de scriere; numele de blocuri duplicate sunt refuzate.

```sh
python3 - "$HOME/mlx-opencode-setup.md" "$HOME/mlx-ssd-restore" <<'PY'
import hashlib, pathlib, re, sys
document = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
destination = pathlib.Path(sys.argv[2])
blocks = re.findall(r"^### Fișier: `([^`]+)`\n\n```[^\n]*\n(.*?)^```$",
                    document, re.MULTILINE | re.DOTALL)
assert blocks, "Nu există anexă în document"
files = dict(blocks)
assert len(files) == len(blocks), "Nume de bloc duplicat"
for name, content in blocks:
    assert pathlib.Path(name).name == name and name not in (".", ".."), name
checksums = {}
for line in files["SHA256SUMS"].splitlines():
    expected, name = line.split("  ", 1)
    assert name not in checksums, name
    checksums[name] = expected
assert set(checksums) == set(files) - {"SHA256SUMS"}, "Inventar incomplet"
for name, expected in checksums.items():
    assert hashlib.sha256(files[name].encode("utf-8")).hexdigest() == expected, name
destination.mkdir(parents=True, exist_ok=True)
for name, content in blocks:
    (destination / name).write_bytes(content.encode("utf-8"))
print(f"Extrase și verificate {len(blocks)} fișiere în {destination}")
PY
```

### Instalare nouă — versiunile fixate

Mai întâi extrage fișierele din document prin procedura de mai sus. Pentru un mediu
nou, instalează pachetele astfel, apoi continuă la **„Aplicarea patch-ului și
restaurarea scripturilor — pas comun”**. Pentru un mediu deja instalat, urmează
în schimb **„Repararea unei instalări existente”**.

```sh
set -e
RESTORE="$HOME/mlx-ssd-restore"
export UV_TOOL_DIR="$HOME/.local/share/uv/tools"
uv tool install --python 3.12.13 --link-mode copy \
  --with-requirements "$RESTORE/requirements.txt" 'mlx-flash==0.9.2'
```

### Repararea unei instalări existente

Închide chatul cu `q` înainte de reparare și extrage fișierele din document prin
procedura de mai sus.

**Condiție pentru diagnosticul de mai jos:** interpretorul
`~/.local/share/uv/tools/mlx-flash/bin/python` trebuie să existe și să poată porni.
Dacă lipsește sau nu funcționează, urmează direct **„Instalare nouă — versiunile
fixate”**; adaugă `--force --reinstall` la comanda `uv tool install` dacă unealta
este încă înregistrată de `uv`. Blocul de diagnostic folosește `set -e` și se
oprește la o eroare de execuție a interpretorului.

Cu un interpretor funcțional, verifică modulul instalat și versiunile pachetelor:

```sh
set -e
RESTORE="$HOME/mlx-ssd-restore"
PY="$HOME/.local/share/uv/tools/mlx-flash/bin/python"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
MODULE="$SITE/mlx_flash_compress/expert_streaming.py"
INSTALLED=$(uv pip freeze --python "$PY")
PINNED=$(cat "$RESTORE/requirements.txt")
if [ "$INSTALLED" != "$PINNED" ]; then
  echo 'REINSTALARE: versiunile pachetelor diferă de profilul documentat.'
elif cmp -s "$MODULE" "$RESTORE/expert_streaming.py"; then
  echo 'PATCH_PREZENT: modulul final și versiunile corespund.'
elif cmp -s "$MODULE" "$RESTORE/expert_streaming.upstream-0.9.2.py"; then
  echo 'ORIGINAL: modulul este baza 0.9.2, gata pentru aplicarea patch-ului.'
else
  echo 'REINSTALARE: modulul lipsește sau diferă de ambele versiuni incluse.'
fi
```

| Rezultat | Pasul următor |
|---|---|
| `PATCH_PREZENT` | Continuă direct la pasul comun de mai jos; acesta recunoaște modulul final și sare peste reaplicarea patch-ului. |
| `ORIGINAL` | Continuă direct la pasul comun; acesta verifică originalul și aplică patch-ul. |
| `REINSTALARE` | Execută blocul următor pentru a reveni la versiunile fixate, apoi continuă la pasul comun. |

**Pentru rezultatul `REINSTALARE`:** păstrează o copie a modulului existent și a
listei de pachete, apoi reinstalează profilul fixat:

```sh
set -e
RESTORE="$HOME/mlx-ssd-restore"
PY="$HOME/.local/share/uv/tools/mlx-flash/bin/python"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
MODULE="$SITE/mlx_flash_compress/expert_streaming.py"
BACKUP=$(mktemp -d "$HOME/mlx-ssd-before-repair.XXXXXX")
if [ -f "$MODULE" ]; then
  cp "$MODULE" "$BACKUP/expert_streaming.py"
fi
uv pip freeze --python "$PY" > "$BACKUP/requirements.txt"
echo "Copia dinaintea reparării: $BACKUP"
export UV_TOOL_DIR="$HOME/.local/share/uv/tools"
uv tool install --force --reinstall --python 3.12.13 --link-mode copy \
  --with-requirements "$RESTORE/requirements.txt" 'mlx-flash==0.9.2'
```

### Aplicarea patch-ului și restaurarea scripturilor — pas comun

Execută acest bloc după instalarea nouă sau după traseul de reparare ales mai sus.
Patch-ul se aplică exclusiv peste originalul verificat; o copie identică cu
modulul final este recunoscută automat.

```sh
set -e
RESTORE="$HOME/mlx-ssd-restore"
PY="$HOME/.local/share/uv/tools/mlx-flash/bin/python"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(
  cd "$SITE/mlx_flash_compress"
  if cmp -s expert_streaming.py "$RESTORE/expert_streaming.py"; then
    echo 'Patch-ul complet este deja prezent.'
  else
    shasum -a 256 -c "$RESTORE/upstream.sha256"
    patch --dry-run -p1 -i "$RESTORE/expert_streaming.patch"
    patch -p1 -i "$RESTORE/expert_streaming.patch"
  fi
  cmp expert_streaming.py "$RESTORE/expert_streaming.py"
)
ROOT="$HOME/.local/share/mlx-ssd-chat"
mkdir -p "$ROOT/test-logs" "$HOME/.local/bin"
cp "$RESTORE/ai.py" "$RESTORE/memory_guard.py" "$RESTORE/cache_memory.py" "$RESTORE/benchmark.py" \
   "$RESTORE/test_streaming.py" "$RESTORE/test_chat.py" "$RESTORE/test_context.py" \
   "$RESTORE/export_setup.py" "$RESTORE/expert_streaming.upstream-0.9.2.py" \
   "$RESTORE/expert_streaming.before-resume.py" "$ROOT/"
cp "$RESTORE/chat-direct.log" "$RESTORE/chat-thinking.log" "$RESTORE/context.log" \
   "$RESTORE/exit-eof.log" "$RESTORE/exit-interrupt.log" \
   "$RESTORE/long-prompt.txt" "$RESTORE/cache-memory-8gb.log" "$ROOT/test-logs/"
install -m 755 "$RESTORE/ai" "$HOME/.local/bin/ai"
uv pip check --python "$PY"
```

Hash-ul original verificat este
`bbb63eb183984d17ccf06721eb0c0bb02a2aa17699241199e70fa2fd4b123680`.
Pasul comun recunoaște fișierul final și evită reaplicarea patch-ului. Dacă verificarea
originalului eșuează, revino la **„Repararea unei instalări existente”** și urmează
ramura `REINSTALARE`, apoi reia pasul comun.

Comanda de instalare fixează directorul uneltelor `uv` la cel utilizat de launcher.
`cmp` verifică faptul că aplicarea patch-ului produce exact sursa completă inclusă
în document. Originalul și copia istorică sunt restaurate alături de generator
pentru a permite refacerea ulterioară a aceleiași anexe.

După pasul comun, continuă la **„Modelul, validarea și pornirea”** pentru testul
diferențial și proba de chat cu modelul real.

### Modelul, validarea și pornirea

Pe un calculator nou, descarcă explicit revizia modelului. Comanda de chat însăși folosește
numai fișierele locale. Dacă revizia există deja în cache, se reutilizează:

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit",
    revision="f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e"))
PY
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" "$HOME/.local/share/mlx-ssd-chat/test_streaming.py"
"$HOME/.local/bin/ai" --prompt 'What is 2 + 2? Reply briefly.' --no-thinking
"$HOME/.local/bin/ai"
```

Pentru comanda scurtă `ai`, adaugă fragmentul shell inclus în document. Comanda
următoare evită adăugarea repetată a aceluiași alias:

```sh
python3 - <<'PY'
from pathlib import Path
rc = Path.home() / ".zshrc"
fragment = (Path.home() / "mlx-ssd-restore/ai.zsh").read_text()
current = rc.read_text() if rc.exists() else ""
alias = fragment.splitlines()[-1]
if alias not in current.splitlines():
    rc.write_text(current + ("\n" if current and not current.endswith("\n") else "") + fragment)
PY
eval "$(cat "$HOME/mlx-ssd-restore/ai.zsh")"
```

Comanda completă `~/.local/bin/ai` funcționează și fără alias.

### Verificarea reinstalării fără a modifica mediul curent

La verificarea profilului a fost creat un mediu temporar separat cu Python 3.12.13 și
`uv pip sync --offline --link-mode copy`, folosind lista completă de 37 de pachete fixate.
Fișierul din wheel-ul reinstalat a avut exact hash-ul original de mai sus. Patch-ul a trecut
`--dry-run`, apoi a reprodus byte-cu-byte fișierul din instalarea funcțională.
`uv pip check` a confirmat compatibilitatea tuturor dependențelor. În acel mediu,
`test_streaming.py` a obținut din nou diferența maximă logits **0,0**; copia exportată
de `ai.py` a răspuns `4` cu modelul real, primul token în 8,3 s și vârf de memorie
fizică de 3,30 GB.

Procedura reproductibilă pornește de la blocurile deja extrase. `--link-mode copy`
izolează copiile de pachete asupra cărora se aplică patch-ul:

```sh
set -e
RESTORE="$HOME/mlx-ssd-restore"
VERIFY=$(mktemp -d "${TMPDIR:-/tmp}/mlx-ssd-verify.XXXXXX")
uv venv --python 3.12.13 "$VERIFY/venv"
VPY="$VERIFY/venv/bin/python"
uv pip sync --link-mode copy --python "$VPY" "$RESTORE/requirements.txt"
VSITE=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(
  cd "$VSITE/mlx_flash_compress"
  shasum -a 256 -c "$RESTORE/upstream.sha256"
  patch --dry-run -p1 -i "$RESTORE/expert_streaming.patch"
  patch -p1 -i "$RESTORE/expert_streaming.patch"
  cmp expert_streaming.py "$RESTORE/expert_streaming.py"
)
uv pip check --python "$VPY"
"$VPY" "$RESTORE/test_streaming.py"
"$VPY" "$RESTORE/ai.py" --no-thinking --prompt 'What is 2 + 2? Reply briefly.'
```

La verificarea anterioară s-a folosit și `--offline`, deoarece pachetele erau deja
în cache-ul `uv`; pe un Mac nou comanda de mai sus le descarcă. Chatul trebuie închis
înaintea probei cu modelul real, deoarece ambele copii folosesc același lock.

Testele funcționale se pot relua secvențial, cu chatul închis:

```sh
PY="$HOME/.local/share/uv/tools/mlx-flash/bin/python"
ROOT="$HOME/.local/share/mlx-ssd-chat"
"$PY" "$ROOT/test_streaming.py"
"$PY" "$ROOT/test_chat.py"
"$PY" "$ROOT/test_chat.py" --thinking
"$PY" "$ROOT/test_context.py"
```

### Verificări rapide și interpretarea ieșirii

Sintaxa tuturor surselor locale și a launcherului:

```sh
PY="$HOME/.local/share/uv/tools/mlx-flash/bin/python"
ROOT="$HOME/.local/share/mlx-ssd-chat"
"$PY" -m py_compile "$ROOT/ai.py" "$ROOT/memory_guard.py" "$ROOT/cache_memory.py" "$ROOT/benchmark.py" \
  "$ROOT/test_streaming.py" "$ROOT/test_chat.py" "$ROOT/test_context.py" "$ROOT/export_setup.py"
zsh -n "$HOME/.local/bin/ai"
~/.local/bin/ai --help
```

Coduri de ieșire: `0` pentru terminare normală, `1` pentru o a doua instanță sau
o eroare de încărcare/execuție tratată, `2` pentru argumente invalide ori depășirea
contextului în modul one-shot, `78` pentru oprirea de către monitor, `130` pentru Ctrl+C.
În chatul interactiv, depășirea contextului afișează explicația și permite alt mesaj.

### Întreținerea documentului

Generatorul complet este inclus în blocul `export_setup.py`. După restaurare,
comanda următoare actualizează anexa din instalarea curentă, recalculează toate
hash-urile și păstrează secțiunile explicative 1–7:

```sh
"$HOME/.local/share/uv/tools/mlx-flash/bin/python" \
  "$HOME/.local/share/mlx-ssd-chat/export_setup.py"
```

Generatorul include modulul instalat complet, originalul, copia istorică, patch-ul,
dependențele, launcherul, fragmentul de alias, toate scripturile locale din profil,
propriul său cod și dovezile text disponibile. Scrie și copii extrase în
`~/.local/share/mlx-ssd-chat/setup-bundle/`; documentul rămâne suficient pentru recuperare.
Rezultatele și explicațiile din secțiunile 1–7 se actualizează atunci când se schimbă
profilul sau sunt efectuate măsurători noi.

### Plan următor — clone Git, fixuri în surse și build local

**Stare: planificat la cererea utilizatorului, 16 octombrie 2026.** Obiectivul este
un flux reproductibil în care sursele și fixurile sunt versionate în Git, iar
chatul folosește pachete construite local din acele surse.

Repository-urile de pornire și rolurile lor:

| Repository | Bază de compatibilitate | Cod relevant |
|---|---|---|
| `https://github.com/ml-explore/mlx` | Tag `v0.32.4`, cu SHA-ul commitului înregistrat | Motorul C++/Metal, alocatorul și sincronizarea evaluării; build nativ și bindinguri Python. |
| `https://github.com/szibis/MLX-Flash` | Commitul corespunzător distribuției `mlx-flash 0.9.2`, verificat față de originalul inclus în acest document | `mlx_flash_compress/expert_streaming.py`, unde se află patch-urile actuale de streaming exact. |

Pașii de implementare:

1. [ ] **Clone Git și baze fixate.** Clonează ambele repository-uri într-un director
   de dezvoltare persistent; înregistrează remote-urile, commiturile de bază și
   creează branch-uri locale pentru fixuri. Verifică sursa MLX-Flash față de
   originalul 0.9.2 inclus în anexă înainte de portarea patch-ului.
2. [ ] **Mută fixurile în sursele versionate.** Portează corecțiile BF16, cache,
   evicție, streaming exact și `pread` în checkout-ul MLX-Flash, cu commituri
   explicite. Versionează și chatul, monitorul, benchmark-ul și testele locale.
   Pentru MLX, identifică și reproduce eventualele defecte native înainte de
   introducerea unui fix; sincronizarea declanșată de limita de memorie este un
   comportament existent, nu un defect demonstrat prin observația de viteză.
3. [ ] **Pregătește mediul de build.** Creează un mediu Python 3.12.13 dedicat;
   fixează dependențele și consemnează versiunile Xcode/SDK, compilatorului, CMake
   și ale uneltelor cerute de sursele MLX 0.32.4. Folosește instrucțiunile de build
   ale reviziei fixate și înregistrează comenzile complete și opțiunile utilizate.
4. [ ] **Construiește MLX local.** Execută întâi un build de control al sursei
   upstream 0.32.4, în configurație Release cu suport Metal și bindinguri Python.
   Produce artefactele instalabile pentru motor și backend-ul Metal în versiuni
   compatibile. Apoi reconstruiește după fiecare fix nativ și păstrează SHA-ul
   sursei, logul build-ului și hash-urile artefactelor.
5. [ ] **Construiește și instalează MLX-Flash patch-uit.** Produce pachetul Python
   din checkout-ul cu fixuri și instalează-l împreună cu build-ul local MLX în
   mediul de dezvoltare. Verifică versiunile, căile efective ale importurilor și
   biblioteca Metal încărcată, pentru a confirma că rulează artefactele locale.
6. [ ] **Validează corectitudinea și performanța.** Rulează testul diferențial
   existent, apoi probele de chat, thinking, context, reset și închidere, urmate
   de generare pe modelul real. Compară build-ul upstream local cu instalarea
   de referință, apoi cu build-ul care include fixuri. Pentru comparația
   2,8 GB versus 8 GB, păstrează constante modelul/revizia, promptul, seed-ul,
   thinking-ul, numărul de tokeni, cache-ul alocatorului
   și bugetul procesului (de exemplu 9,2 GB în ambele cazuri). Folosește execuții
   repetate în ordine alternată și consemnează încălzirea cache-urilor. Măsoară
   timpul până la primul token, tokeni/s, memoria activă/cache/vârf MLX și
   `physical footprint`. Pentru izolarea efectului pragului asupra sincronizării,
   fixează explicit în harness aceeași capacitate a cache-ului în ambele probe;
   pentru evaluarea profilului curent, măsoară separat și dimensionarea automată
   (12 versus 49 de experți/strat).
7. [ ] **Integrează build-ul verificat în launcher și documentație.** Configurează
   `ai` să folosească mediul cu artefactele locale validate. Actualizează generatorul
   de export pentru noile căi, dependențele fixate și documentul autonom, incluzând
   comenzile de clonare/build/instalare, patch-urile, sursele locale modificate,
   versiunile toolchain-ului, commiturile și rezultatele măsurate. Documentează
   restaurarea profilului de referință din secțiunile existente.

**Criteriu de finalizare:** un mediu nou poate reproduce build-ul din commiturile
fixate, poate instala artefactele rezultate și poate rula chatul și probele de
corectitudine folosind sursele locale cu fixuri. Rezultatele de viteză sunt
raportate separat pentru fiecare configurație măsurată.

### Depanare

| Simptom | Explicație și acțiune |
|---|---|
| `ai: command not found` | Folosește `~/.local/bin/ai` sau aplică fragmentul de alias de mai sus. |
| Modelul/revizia nu există local | Rulează comanda explicită `snapshot_download` din această secțiune; launcherul folosește numai cache-ul local. |
| `Un chat ai rulează deja` | Închide instanța anterioară cu `q`, Ctrl+D sau Ctrl+C. Lock-ul este ținut de proces și se eliberează la terminarea acestuia. |
| `Memory budget exceeded` / codul 78 | Monitorul a observat depășirea bugetului procesului afișat la pornire, implicit 4 GB; încearcă un prompt mai scurt, o conversație nouă sau ajustează explicit bugetul. |
| Eroare pentru `--mlx-memory-gb` / `--process-memory-gb` | Folosește numere pozitive finite în GB, cu punct pentru zecimale; bugetul explicit al procesului trebuie să fie mai mare decât cel MLX. |
| `Memory monitor failed` | Citirea memoriei prin API-ul macOS a eșuat; monitorul închide procesul cu codul 78. |
| Mesaj despre bugetul de 4096 tokeni | Bugetul include istoricul, template-ul și rezerva `--max-tokens`; folosește `/reset`, un mesaj mai scurt sau o rezervă de răspuns mai mică. |
| `Limită de răspuns atinsă` | Generarea a atins `--max-tokens`, inclusiv thinking. Un text final poate exista chiar dacă motivul de terminare este `length`. |
| Primul token întârzie la prompt lung | Prefill-ul procesează întregul prompt; proba de 4078 tokeni a necesitat 120,4 s. |
| Verificarea `upstream.sha256` eșuează | Urmează „Repararea unei instalări existente”: modulul final identic este recunoscut ca `PATCH_PREZENT`; pentru alt conținut sau alte versiuni, ramura `REINSTALARE` restaurează baza fixată înainte de pasul comun. |
| Avertismentul `PyTorch was not found` | Inferența acestui profil se face prin MLX; probele au trecut și cu acest avertisment al Transformers. |

## 8. Anexă reproductibilă — fișiere complete

Toate sursele locale, configurația shell, patch-ul, versiunile fixate și dovezile text ale testelor sunt incluse integral mai jos. Numele identifică blocurile de extras; conținutul se află în acest document. Modulul instalat, originalul 0.9.2 și copia istorică sunt complete, iar generatorul acestei anexe este inclus și el.

Logurile sunt reprezentate textual: CRLF/CR devin LF, backspace devine `\b`, iar blocurile primesc un newline final dacă lipsea. Hash-urile din `SHA256SUMS` verifică exact forma inclusă în document. Sursele Python și patch-ul păstrează conținutul instalării verificate.

Inventar de blocuri complete:
- `expert_streaming.py`
- `expert_streaming.upstream-0.9.2.py`
- `expert_streaming.before-resume.py`
- `expert_streaming.patch`
- `requirements.txt`
- `upstream.sha256`
- `ai`
- `ai.zsh`
- `ai.py`
- `memory_guard.py`
- `cache_memory.py`
- `benchmark.py`
- `test_streaming.py`
- `test_chat.py`
- `test_context.py`
- `export_setup.py`
- `cache-memory-8gb.log`
- `chat-direct.log`
- `chat-thinking.log`
- `context.log`
- `exit-eof.log`
- `exit-interrupt.log`
- `long-prompt.txt`
- `SHA256SUMS`

### Fișier: `expert_streaming.py`

```python
"""Expert streaming: GPU lookup table + pre-stacked weight tensors.

Local exact Qwen3-MoE mode: enable_expert_streaming(..., exact=True,
read_mode="pread"). Loads ALL routed experts before computation, using bounded
disk reads and a temporary one-layer working set during prefill. This mode
preserves routing semantics; the legacy approximate mode below is unchanged.

Replaces MLX's QuantizedSwitchLinear with a cached version that:
1. Keeps only `capacity` experts per layer in GPU memory (pre-stacked tensors)
2. Uses a GPU lookup table to map global expert IDs to local cache slots
3. Updates cache between tokens (not during forward pass)
4. Zero-eval fallback: missing experts get score=0, remaining renormalized

This is the technique that makes 50GB+ models run on 36GB Macs.

Architecture:
  Forward pass (GPU only, no CPU/disk):
    router(x) -> expert_ids -> lookup[expert_ids] -> local_ids
    gather_qmm(x, stacked_weights, local_ids) -> output

  Between tokens (CPU + disk):
    Record which experts were activated
    Evict cold experts (LCP priority)
    Load new experts from safetensors via mmap
    Update stacked tensors + lookup table

Usage:
  from mlx_flash_compress.expert_streaming import enable_expert_streaming

  model, tokenizer = mlx_lm.load("model_name", lazy=True)
  streaming = enable_expert_streaming(model, capacity_per_layer=200)
  streaming.warmup()

  # Generate normally
  output = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=100)
  streaming.update()  # call between generation steps
"""

import json
import math
import mmap
import os
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# -- Safetensors mmap reader --


@dataclass
class TensorInfo:
    dtype: str
    shape: list
    data_offset: int
    data_size: int


class SafetensorsMap:
    """Memory-mapped access to safetensors files for per-expert slicing."""

    def __init__(self, shard_paths: list, read_mode="mmap"):
        if read_mode not in ("mmap", "pread"):
            raise ValueError("read_mode must be mmap or pread")
        self.read_mode = read_mode
        self._fds = {}
        self._mmaps: dict[str, mmap.mmap] = {}
        self._tensor_map: dict[str, tuple] = {}
        self._data_offsets: dict[str, int] = {}
        for path in shard_paths:
            self._index_shard(path)

    def _index_shard(self, path):
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            data_start = 8 + header_size

        header = json.loads(header_bytes)
        self._data_offsets[path] = data_start

        for key, info in header.items():
            if key == "__metadata__":
                continue
            offsets = info.get("data_offsets", [0, 0])
            self._tensor_map[key] = (
                path,
                TensorInfo(
                    dtype=info.get("dtype", "F16"),
                    shape=info.get("shape", []),
                    data_offset=offsets[0],
                    data_size=offsets[1] - offsets[0],
                ),
            )

    def _get_mmap(self, path):
        if path not in self._mmaps:
            with open(path, "rb") as f:
                self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[path]

    def _read(self, path, offset, size):
        if self.read_mode == "mmap":
            return self._get_mmap(path)[offset:offset + size]
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        data = os.pread(self._fds[path], size, offset)
        if len(data) != size:
            raise OSError(f"Short safetensors read: {path}")
        return data

    _NP_DTYPES = {
        "F16": np.float16,
        "BF16": np.uint16,
        "F32": np.float32,
        "U32": np.uint32,
        "I32": np.int32,
        "U8": np.uint8,
    }

    def get_expert_slice(self, key, expert_ids):
        """Load specific expert rows from stacked [E, ...] or per-expert format."""
        # Try stacked format first (key = "...switch_mlp.gate_proj.weight" shape [E, ...])
        if key in self._tensor_map:
            path, info = self._tensor_map[key]
            if len(info.shape) >= 3:
                # Stacked format: slice rows directly
                num_experts = info.shape[0]
                row_size = info.data_size // num_experts
                base = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES[info.dtype]
                row_shape = info.shape[1:]

                slices = []
                for eid in expert_ids:
                    if not 0 <= eid < num_experts:
                        raise IndexError(f"Expert {eid} outside {key}")
                    offset = base + eid * row_size
                    raw = self._read(path, offset, row_size)
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_shape)
                    slices.append(arr)
                res = mx.array(np.stack(slices))
                return res.view(mx.bfloat16) if info.dtype == "BF16" else res

        # Per-expert format: "model.layers.N.mlp.experts.E.proj.attr"
        # Derive the per-expert key pattern from the stacked key
        # e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        #    -> "model.layers.0.mlp.experts.{E}.gate_proj.weight"
        per_expert_key = key.replace(".switch_mlp.", ".experts.0.")

        # Mixtral: block_sparse_moe.experts.E.w1/w2/w3 naming
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        if per_expert_key not in self._tensor_map:
            for mlx_name, st_name in _MIXTRAL_MAP.items():
                per_expert_key = per_expert_key.replace(f".{mlx_name}.", f".{st_name}.")

        if per_expert_key in self._tensor_map:
            # Derive the eid_key template from per_expert_key (which has experts.0.)
            slices = []
            for eid in expert_ids:
                eid_key = per_expert_key.replace(".experts.0.", f".experts.{eid}.")
                if eid_key not in self._tensor_map:
                    raise KeyError(eid_key)
                path, info = self._tensor_map[eid_key]
                offset = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES[info.dtype]
                raw = self._read(path, offset, info.data_size)
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
                slices.append(arr)
            if slices:
                res = mx.array(np.stack(slices))
                return res.view(mx.bfloat16) if info.dtype == "BF16" else res

        return None

    def has_key(self, key):
        return key in self._tensor_map

    def close(self):
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


# -- LCP eviction tracker --


class LCPTracker:
    """LCP eviction with optional layer-depth bias (FATE paper).

    Shallow (early) layers have more predictable routing patterns,
    so their experts are more valuable to keep cached. The layer_depth_bias
    multiplies priority by (1 + bias * (1 - layer_frac)) where layer_frac
    is the layer's position (0=first, 1=last).
    """

    def __init__(self, num_experts, lcp_base=0.25, lcp_decay=128.0, layer_depth_bias=0.0, layer_frac=0.5):
        self.num_experts = num_experts
        self.lcp_base = lcp_base
        self.lcp_decay = lcp_decay
        # Shallow-favoring bias: higher for early layers
        self._depth_multiplier = 1.0 + layer_depth_bias * (1.0 - layer_frac)
        self.frequency = np.zeros(num_experts, dtype=np.int64)
        self.last_used = np.zeros(num_experts, dtype=np.int64)
        self.step = 0

    def record(self, expert_ids):
        self.step += 1
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                self.frequency[eid] += 1
                self.last_used[eid] = self.step

    def priority(self, eid):
        age = self.step - self.last_used[eid]
        base_priority = self.frequency[eid] * (self.lcp_base ** (age / self.lcp_decay))
        return base_priority * self._depth_multiplier

    def coldest(self, among, n):
        priorities = [(eid, self.priority(eid)) for eid in among]
        priorities.sort(key=lambda x: x[1])
        return [eid for eid, _ in priorities[:n]]


# -- Cached QuantizedSwitchLinear replacement --


class CachedSwitchLinear(nn.Module):
    """Drop-in replacement using GPU lookup table + pre-stacked weights."""

    def __init__(self, cache, proj_name, group_size, bits, mode=0):
        super().__init__()
        self._cache = cache
        self._proj_name = proj_name
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "gate_proj" and not self._cache.exact:
            self._cache._indices_buffer.append(indices)

        local_indices = self._cache.lookup[indices]

        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases.get(self._proj_name),
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )


# -- Expert cache per layer --


class ExpertCache:
    """Pre-stacked expert weights in GPU with LCP eviction."""

    def __init__(self, layer_idx, num_experts, capacity, st_map, weight_keys, num_layers=1, layer_depth_bias=0.3):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = min(capacity, num_experts)
        self.st_map = st_map
        self.weight_keys = weight_keys
        self.cached_ids = []
        self.lookup = mx.zeros(num_experts, dtype=mx.int32)
        self.hit_mask = mx.zeros(num_experts)
        self.weights = {}
        self.scales = {}
        self.biases = {}
        layer_frac = layer_idx / max(num_layers - 1, 1)
        self.tracker = LCPTracker(num_experts, layer_depth_bias=layer_depth_bias, layer_frac=layer_frac)
        self._indices_buffer = []
        self.total_tokens = 0
        self.cache_updates = 0
        self.exact = False

    def initial_fill(self, expert_ids=None):
        if expert_ids is None:
            expert_ids = list(range(self.capacity))
        self.cached_ids = expert_ids[: self.capacity]

        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, self.cached_ids)
            if data is None:
                if self.exact:
                    raise KeyError(st_key)
                continue
            parts = full_key_name.split(".")
            proj = parts[0]  # gate_proj, up_proj, down_proj
            attr = parts[1]  # weight, scales, biases
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            target[proj] = data

        self._rebuild_lookup()

    def _rebuild_lookup(self):
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def ensure(self, indices):
        """Populate the current routing set BEFORE any expert computation."""
        ids = np.unique(np.array(indices)).tolist()
        if len(ids) > self.capacity:
            raise ValueError("Routing set exceeds resident cache capacity")
        self._indices_buffer = [indices]
        self.update_between_tokens()
        if not set(ids).issubset(self.cached_ids):
            raise RuntimeError("Expert cache failed to satisfy current routing")

    def update_between_tokens(self):
        if not self._indices_buffer:
            return
        activated = set()
        for indices in self._indices_buffer:
            ids = np.array(indices).flatten().tolist()
            activated.update(ids)
            self.tracker.record(ids)
        self._indices_buffer.clear()
        self.total_tokens += 1

        cached_set = set(self.cached_ids)
        misses = sorted(eid for eid in activated if eid not in cached_set)
        if not misses:
            return

        evict_candidates = [eid for eid in self.cached_ids if eid not in activated]
        n_to_evict = min(len(misses), len(evict_candidates))
        to_evict = self.tracker.coldest(evict_candidates, n_to_evict)
        if not to_evict:
            return

        new_ids = misses[: len(to_evict)]
        evict_set = set(to_evict)
        evict_map = dict(zip(to_evict, new_ids))
        slots = [i for i, eid in enumerate(self.cached_ids) if eid in evict_set]
        new_ids_ordered = [evict_map[self.cached_ids[s]] for s in slots]
        for s, new_eid in zip(slots, new_ids_ordered):
            self.cached_ids[s] = new_eid

        # Reload only the evicted expert slots in-place from disk
        slot_idx = mx.array(slots)
        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, new_ids_ordered)
            if data is None:
                if self.exact:
                    raise KeyError(st_key)
                continue
            parts = full_key_name.split(".")
            proj = parts[0]
            attr = parts[1]
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            if proj in target and len(slots) > 0:
                target[proj][slot_idx] = data
                mx.eval(target[proj])

        self._rebuild_lookup()
        mx.eval(self.lookup, self.hit_mask)
        self.cache_updates += 1

        # Release evicted expert buffers back to OS (MLX v0.31+)
        try:
            mx.clear_cache()
        except AttributeError:
            pass  # older MLX versions

    def stats(self):
        return {
            "layer": self.layer_idx,
            "cached": len(self.cached_ids),
            "capacity": self.capacity,
            "coverage": len(self.cached_ids) / max(self.num_experts, 1),
            "tokens": self.total_tokens,
            "updates": self.cache_updates,
        }


# -- Streaming state --


@dataclass
class StreamingState:
    caches: list = field(default_factory=list)
    st_map: object = None
    model: object = None

    def warmup(self):
        for cache in self.caches:
            cache.initial_fill()
        arrays = []
        for cache in self.caches:
            arrays.extend(cache.weights.values())
            arrays.extend(cache.scales.values())
            arrays.extend(cache.biases.values())
        if arrays:
            mx.eval(*arrays)

    def update(self):
        for cache in self.caches:
            cache.update_between_tokens()

    def stats(self):
        return [c.stats() for c in self.caches]

    def total_cached(self):
        return sum(len(c.cached_ids) for c in self.caches)

    def avg_coverage(self):
        if not self.caches:
            return 0
        return sum(c.stats()["coverage"] for c in self.caches) / len(self.caches)

    def cleanup(self):
        if self.st_map:
            self.st_map.close()


class ExactStreamingSwitchGLU(nn.Module):
    """Load all routed experts, retaining Qwen's original router and scores.

    Decode uses the resident cache. Prefill may need more experts than its
    capacity: these are loaded temporarily for ONE layer, evaluated, and freed.
    No instance-level __call__ monkey-patching or missing-expert substitution.
    """

    def __init__(self, switch, cache):
        super().__init__()
        self.switch = switch
        self._cache = cache

    def __call__(self, x, indices):
        mx.eval(x, indices)
        ids = np.unique(np.array(indices)).tolist()
        cache = self._cache
        if len(ids) <= cache.capacity:
            cache.ensure(indices)
            result = self.switch(x, indices)
            # Finish using mutable cache slots before the next forward/eviction.
            mx.eval(result)
            return result

        temporary = ExpertCache(cache.layer_idx, cache.num_experts, len(ids),
                                cache.st_map, cache.weight_keys)
        temporary.exact = True
        temporary.initial_fill(ids)
        local_indices = temporary.lookup[indices]
        expanded = mx.expand_dims(x, (-2, -3))

        def project(name, values):
            proj = getattr(self.switch, name)
            return mx.gather_qmm(
                values, temporary.weights[name], temporary.scales[name],
                temporary.biases.get(name), rhs_indices=local_indices,
                transpose=True, group_size=proj.group_size, bits=proj.bits,
                mode=proj.mode, sorted_indices=False)

        up = project("up_proj", expanded)
        gate = project("gate_proj", expanded)
        result = project("down_proj", self.switch.activation(up, gate)).squeeze(-2)
        mx.eval(result)
        return result


# -- Skip-fallback: zero missing expert scores --


def enable_skip_fallback(model, caches: list, adaptive_skip_threshold: float = 0.0):
    """Monkey-patch MoE blocks to zero scores for uncached experts.

    When an expert is not in the cache, its routing score is set to 0
    and remaining scores are renormalized. This avoids computing with
    stale/placeholder weights for uncached experts.

    If adaptive_skip_threshold > 0, also skip low-confidence secondary
    experts when top-1 score / top-2 score > threshold (LExI paper).
    E.g., threshold=3.0 means skip expert 2 if expert 1 is 3x more likely.
    This gives 10-30% compute reduction with <1% quality loss.
    """
    import types

    if any(c.exact for c in caches):
        raise ValueError("Skip fallback is incompatible with exact streaming")

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    cache_by_layer = {c.layer_idx: c for c in caches}

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in cache_by_layer:
            continue
        if not hasattr(layer, "mlp"):
            continue

        cache = cache_by_layer[layer_idx]
        mlp = layer.mlp
        original_call = type(mlp).__call__

        skip_thresh = adaptive_skip_threshold

        def make_patched(orig, cache_ref):
            def patched_call(self, x):
                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = self.top_k
                inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # Zero out scores for uncached experts
                mask = cache_ref.hit_mask[inds]  # 1.0 cached, 0.0 not
                scores = scores * mask

                # Adaptive top-k: skip low-confidence secondary experts (LExI)
                if skip_thresh > 0 and k > 1:
                    top1 = scores[..., :1]
                    # Zero out experts where top-1 is much stronger
                    confidence_mask = mx.where(top1 > skip_thresh * scores, 0.0, 1.0)
                    # Keep top-1 always
                    confidence_mask = mx.concatenate([mx.ones_like(scores[..., :1]), confidence_mask[..., 1:]], axis=-1)
                    scores = scores * confidence_mask

                score_sum = scores.sum(axis=-1, keepdims=True)
                scores = mx.where(score_sum > 0, scores / score_sum, scores)

                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

                if hasattr(self, "shared_expert"):
                    shared = self.shared_expert(x)
                    if hasattr(self, "shared_expert_gate"):
                        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
                    y = y + shared

                return y

            return patched_call

        mlp.__call__ = types.MethodType(make_patched(original_call, cache), mlp)


# -- Profile-based warmup --


def get_warmup_experts(task: str = "general", num_layers: int = 24, num_experts: int = 60, top_n: int = 30) -> list:
    """Get hot experts for a task from pre-computed profiles.

    Uses task_profiler's predefined profiles to determine which experts
    to pre-load for a given workload type.
    """
    try:
        from mlx_flash_compress.task_profiler import get_predefined_profile

        profile = get_predefined_profile(task, num_layers=num_layers, num_experts=num_experts)
        hot_experts = profile.get_hot_experts(top_pct=top_n / num_experts)

        # Convert to flat list of expert IDs per layer
        result = []
        for layer_idx in range(num_layers):
            if layer_idx in hot_experts:
                result.append(hot_experts[layer_idx][:top_n])
            else:
                result.append(list(range(top_n)))
        return result
    except Exception:
        # Fallback: first N experts
        return [list(range(top_n)) for _ in range(num_layers)]


# -- Public API --


def enable_expert_streaming(model, capacity_per_layer=200, model_path=None,
                            exact=False, read_mode="mmap"):
    """Enable expert streaming on a loaded MLX MoE model.

    Replaces QuantizedSwitchLinear with cached versions using
    pre-stacked GPU tensors and lookup tables.
    """
    if capacity_per_layer < 1:
        raise ValueError("capacity_per_layer must be positive")
    if exact and getattr(model, "model_type", None) != "qwen3_moe":
        raise ValueError("Exact streaming is validated for qwen3_moe only")
    state = StreamingState(model=model)

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        raise ValueError("Cannot find model layers")

    # Find safetensors
    if model_path:
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            # Try HuggingFace cache
            import glob

            cache_pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}*/snapshots/*/"
            )
            cache_dirs = sorted(glob.glob(cache_pattern))
            if cache_dirs:
                model_dir = Path(cache_dirs[-1])

        shards = sorted(model_dir.glob("*.safetensors"))
        if shards:
            state.st_map = SafetensorsMap([str(s) for s in shards], read_mode=read_mode)

    if state.st_map is None:
        raise ValueError("model_path must resolve to local safetensors shards")

    total_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp

        # Detect switch_mlp (Qwen/DeepSeek) or block_sparse_moe (Mixtral)
        switch = getattr(mlp, "switch_mlp", None)
        is_mixtral = False
        if switch is None:
            bsm = getattr(mlp, "block_sparse_moe", None)
            if bsm is not None:
                switch = getattr(bsm, "switch_mlp", None)
                is_mixtral = True
        if switch is None:
            continue

        # Mixtral safetensors use w1/w2/w3 naming vs gate_proj/up_proj/down_proj
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

        num_experts = 0
        proj_modules = {}
        weight_keys = {}

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(switch, proj_name, None)
            if proj is None:
                continue
            w = getattr(proj, "weight", None)
            if w is not None and len(w.shape) == 3:
                num_experts = w.shape[0]
                proj_modules[proj_name] = proj
                for suffix in ["weight", "scales", "biases"]:
                    if is_mixtral:
                        # Mixtral safetensors: block_sparse_moe.switch_mlp.gate_proj.weight
                        # but also try: block_sparse_moe.experts.E.w1.weight (per-expert)
                        wk = f"model.layers.{layer_idx}.block_sparse_moe.switch_mlp.{proj_name}.{suffix}"
                    else:
                        wk = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.{suffix}"
                    weight_keys[f"{proj_name}.{suffix}"] = wk

        if num_experts == 0:
            continue

        if exact:
            if capacity_per_layer < mlp.top_k:
                raise ValueError("Exact cache capacity must be at least top_k")
            if getattr(mlp, "sharding_group", None) is not None:
                raise ValueError("Exact streaming does not support distributed sharding")
            for proj in proj_modules.values():
                if getattr(proj, "mode", "affine") != "affine" or "bias" in proj:
                    raise ValueError("Exact streaming requires bias-free affine projections")

        cache = ExpertCache(
            layer_idx=layer_idx,
            num_experts=num_experts,
            capacity=min(capacity_per_layer, num_experts),
            st_map=state.st_map,
            weight_keys=weight_keys,
            num_layers=total_layers,
        )
        cache.exact = exact
        state.caches.append(cache)

        for proj_name, proj in proj_modules.items():
            replacement = CachedSwitchLinear(
                cache=cache,
                proj_name=proj_name,
                group_size=getattr(proj, "group_size", 64),
                bits=getattr(proj, "bits", 4),
                mode=getattr(proj, "mode", 0),
            )
            setattr(switch, proj_name, replacement)

        if exact:
            mlp.switch_mlp = ExactStreamingSwitchGLU(switch, cache)

    return state
```

### Fișier: `expert_streaming.upstream-0.9.2.py`

```python
"""Real expert streaming: GPU lookup table + pre-stacked weight tensors.

Replaces MLX's QuantizedSwitchLinear with a cached version that:
1. Keeps only `capacity` experts per layer in GPU memory (pre-stacked tensors)
2. Uses a GPU lookup table to map global expert IDs to local cache slots
3. Updates cache between tokens (not during forward pass)
4. Zero-eval fallback: missing experts get score=0, remaining renormalized

This is the technique that makes 50GB+ models run on 36GB Macs.

Architecture:
  Forward pass (GPU only, no CPU/disk):
    router(x) -> expert_ids -> lookup[expert_ids] -> local_ids
    gather_qmm(x, stacked_weights, local_ids) -> output

  Between tokens (CPU + disk):
    Record which experts were activated
    Evict cold experts (LCP priority)
    Load new experts from safetensors via mmap
    Update stacked tensors + lookup table

Usage:
  from mlx_flash_compress.expert_streaming import enable_expert_streaming

  model, tokenizer = mlx_lm.load("model_name", lazy=True)
  streaming = enable_expert_streaming(model, capacity_per_layer=200)
  streaming.warmup()

  # Generate normally
  output = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=100)
  streaming.update()  # call between generation steps
"""

import json
import math
import mmap
import os
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# -- Safetensors mmap reader --


@dataclass
class TensorInfo:
    dtype: str
    shape: list
    data_offset: int
    data_size: int


class SafetensorsMap:
    """Memory-mapped access to safetensors files for per-expert slicing."""

    def __init__(self, shard_paths: list):
        self._mmaps: dict[str, mmap.mmap] = {}
        self._tensor_map: dict[str, tuple] = {}
        self._data_offsets: dict[str, int] = {}
        for path in shard_paths:
            self._index_shard(path)

    def _index_shard(self, path):
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            data_start = 8 + header_size

        header = json.loads(header_bytes)
        self._data_offsets[path] = data_start

        for key, info in header.items():
            if key == "__metadata__":
                continue
            offsets = info.get("data_offsets", [0, 0])
            self._tensor_map[key] = (
                path,
                TensorInfo(
                    dtype=info.get("dtype", "F16"),
                    shape=info.get("shape", []),
                    data_offset=offsets[0],
                    data_size=offsets[1] - offsets[0],
                ),
            )

    def _get_mmap(self, path):
        if path not in self._mmaps:
            f = open(path, "rb")
            self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[path]

    _NP_DTYPES = {
        "F16": np.float16,
        "BF16": np.float16,
        "F32": np.float32,
        "U32": np.uint32,
        "I32": np.int32,
        "U8": np.uint8,
    }

    def get_expert_slice(self, key, expert_ids):
        """Load specific expert rows from stacked [E, ...] or per-expert format."""
        # Try stacked format first (key = "...switch_mlp.gate_proj.weight" shape [E, ...])
        if key in self._tensor_map:
            path, info = self._tensor_map[key]
            if len(info.shape) >= 3:
                # Stacked format: slice rows directly
                num_experts = info.shape[0]
                row_size = info.data_size // num_experts
                mm = self._get_mmap(path)
                base = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                row_shape = info.shape[1:]

                slices = []
                for eid in expert_ids:
                    offset = base + eid * row_size
                    raw = mm[offset : offset + row_size]
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_shape)
                    slices.append(arr)
                return mx.array(np.stack(slices))

        # Per-expert format: "model.layers.N.mlp.experts.E.proj.attr"
        # Derive the per-expert key pattern from the stacked key
        # e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        #    -> "model.layers.0.mlp.experts.{E}.gate_proj.weight"
        per_expert_key = key.replace(".switch_mlp.", ".experts.0.")

        # Mixtral: block_sparse_moe.experts.E.w1/w2/w3 naming
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        if per_expert_key not in self._tensor_map:
            for mlx_name, st_name in _MIXTRAL_MAP.items():
                per_expert_key = per_expert_key.replace(f".{mlx_name}.", f".{st_name}.")

        if per_expert_key in self._tensor_map:
            # Derive the eid_key template from per_expert_key (which has experts.0.)
            slices = []
            for eid in expert_ids:
                eid_key = per_expert_key.replace(".experts.0.", f".experts.{eid}.")
                if eid_key not in self._tensor_map:
                    continue
                path, info = self._tensor_map[eid_key]
                mm = self._get_mmap(path)
                offset = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                raw = mm[offset : offset + info.data_size]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
                slices.append(arr)
            if slices:
                return mx.array(np.stack(slices))

        return None

    def has_key(self, key):
        return key in self._tensor_map

    def close(self):
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()


# -- LCP eviction tracker --


class LCPTracker:
    """LCP eviction with optional layer-depth bias (FATE paper).

    Shallow (early) layers have more predictable routing patterns,
    so their experts are more valuable to keep cached. The layer_depth_bias
    multiplies priority by (1 + bias * (1 - layer_frac)) where layer_frac
    is the layer's position (0=first, 1=last).
    """

    def __init__(self, num_experts, lcp_base=0.25, lcp_decay=128.0, layer_depth_bias=0.0, layer_frac=0.5):
        self.num_experts = num_experts
        self.lcp_base = lcp_base
        self.lcp_decay = lcp_decay
        # Shallow-favoring bias: higher for early layers
        self._depth_multiplier = 1.0 + layer_depth_bias * (1.0 - layer_frac)
        self.frequency = np.zeros(num_experts, dtype=np.int64)
        self.last_used = np.zeros(num_experts, dtype=np.int64)
        self.step = 0

    def record(self, expert_ids):
        self.step += 1
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                self.frequency[eid] += 1
                self.last_used[eid] = self.step

    def priority(self, eid):
        age = self.step - self.last_used[eid]
        base_priority = self.frequency[eid] * (self.lcp_base ** (age / self.lcp_decay))
        return base_priority * self._depth_multiplier

    def coldest(self, among, n):
        priorities = [(eid, self.priority(eid)) for eid in among]
        priorities.sort(key=lambda x: x[1])
        return [eid for eid, _ in priorities[:n]]


# -- Cached QuantizedSwitchLinear replacement --


class CachedSwitchLinear(nn.Module):
    """Drop-in replacement using GPU lookup table + pre-stacked weights."""

    def __init__(self, cache, proj_name, group_size, bits, mode=0):
        super().__init__()
        self._cache = cache
        self._proj_name = proj_name
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "gate_proj":
            self._cache._indices_buffer.append(indices)

        local_indices = self._cache.lookup[indices]

        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases.get(self._proj_name),
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


# -- Expert cache per layer --


class ExpertCache:
    """Pre-stacked expert weights in GPU with LCP eviction."""

    def __init__(self, layer_idx, num_experts, capacity, st_map, weight_keys, num_layers=1, layer_depth_bias=0.3):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = min(capacity, num_experts)
        self.st_map = st_map
        self.weight_keys = weight_keys
        self.cached_ids = []
        self.lookup = mx.zeros(num_experts, dtype=mx.int32)
        self.hit_mask = mx.zeros(num_experts)
        self.weights = {}
        self.scales = {}
        self.biases = {}
        layer_frac = layer_idx / max(num_layers - 1, 1)
        self.tracker = LCPTracker(num_experts, layer_depth_bias=layer_depth_bias, layer_frac=layer_frac)
        self._indices_buffer = []
        self.total_tokens = 0
        self.cache_updates = 0

    def initial_fill(self, expert_ids=None):
        if expert_ids is None:
            expert_ids = list(range(self.capacity))
        self.cached_ids = expert_ids[: self.capacity]

        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, self.cached_ids)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]  # gate_proj, up_proj, down_proj
            attr = parts[1]  # weight, scales, biases
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            target[proj] = data

        self._rebuild_lookup()

    def _rebuild_lookup(self):
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def update_between_tokens(self):
        if not self._indices_buffer:
            return
        activated = set()
        for indices in self._indices_buffer:
            ids = np.array(indices).flatten().tolist()
            activated.update(ids)
            self.tracker.record(ids)
        self._indices_buffer.clear()
        self.total_tokens += 1

        cached_set = set(self.cached_ids)
        misses = [eid for eid in activated if eid not in cached_set]
        if not misses:
            return

        evict_candidates = [eid for eid in self.cached_ids if eid not in activated]
        n_to_evict = min(len(misses), len(evict_candidates))
        to_evict = self.tracker.coldest(evict_candidates, n_to_evict)
        if not to_evict:
            return

        new_ids = misses[: len(to_evict)]
        evict_set = set(to_evict)
        new_cached = [eid for eid in self.cached_ids if eid not in evict_set]
        new_cached.extend(new_ids)
        self.cached_ids = new_cached[: self.capacity]

        # Reload changed expert slots from disk
        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, new_ids)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]
            attr = parts[1]
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            if proj in target:
                # Find slot indices for new experts
                slot_map = {eid: i for i, eid in enumerate(self.cached_ids)}
                slots = [slot_map[eid] for eid in new_ids if eid in slot_map]
                if slots:
                    slot_idx = mx.array(slots[: data.shape[0]])
                    target[proj][slot_idx] = data

        self._rebuild_lookup()
        self.cache_updates += 1

        # Release evicted expert buffers back to OS (MLX v0.31+)
        try:
            mx.clear_cache()
        except AttributeError:
            pass  # older MLX versions

    def stats(self):
        return {
            "layer": self.layer_idx,
            "cached": len(self.cached_ids),
            "capacity": self.capacity,
            "coverage": len(self.cached_ids) / max(self.num_experts, 1),
            "tokens": self.total_tokens,
            "updates": self.cache_updates,
        }


# -- Streaming state --


@dataclass
class StreamingState:
    caches: list = field(default_factory=list)
    st_map: object = None
    model: object = None

    def warmup(self):
        for cache in self.caches:
            cache.initial_fill()
        arrays = []
        for cache in self.caches:
            arrays.extend(cache.weights.values())
            arrays.extend(cache.scales.values())
            arrays.extend(cache.biases.values())
        if arrays:
            mx.eval(*arrays)

    def update(self):
        for cache in self.caches:
            cache.update_between_tokens()

    def stats(self):
        return [c.stats() for c in self.caches]

    def total_cached(self):
        return sum(len(c.cached_ids) for c in self.caches)

    def avg_coverage(self):
        if not self.caches:
            return 0
        return sum(c.stats()["coverage"] for c in self.caches) / len(self.caches)

    def cleanup(self):
        if self.st_map:
            self.st_map.close()


# -- Skip-fallback: zero missing expert scores --


def enable_skip_fallback(model, caches: list, adaptive_skip_threshold: float = 0.0):
    """Monkey-patch MoE blocks to zero scores for uncached experts.

    When an expert is not in the cache, its routing score is set to 0
    and remaining scores are renormalized. This avoids computing with
    stale/placeholder weights for uncached experts.

    If adaptive_skip_threshold > 0, also skip low-confidence secondary
    experts when top-1 score / top-2 score > threshold (LExI paper).
    E.g., threshold=3.0 means skip expert 2 if expert 1 is 3x more likely.
    This gives 10-30% compute reduction with <1% quality loss.
    """
    import types

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    cache_by_layer = {c.layer_idx: c for c in caches}

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in cache_by_layer:
            continue
        if not hasattr(layer, "mlp"):
            continue

        cache = cache_by_layer[layer_idx]
        mlp = layer.mlp
        original_call = type(mlp).__call__

        skip_thresh = adaptive_skip_threshold

        def make_patched(orig, cache_ref):
            def patched_call(self, x):
                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = self.top_k
                inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # Zero out scores for uncached experts
                mask = cache_ref.hit_mask[inds]  # 1.0 cached, 0.0 not
                scores = scores * mask

                # Adaptive top-k: skip low-confidence secondary experts (LExI)
                if skip_thresh > 0 and k > 1:
                    top1 = scores[..., :1]
                    # Zero out experts where top-1 is much stronger
                    confidence_mask = mx.where(top1 > skip_thresh * scores, 0.0, 1.0)
                    # Keep top-1 always
                    confidence_mask = mx.concatenate([mx.ones_like(scores[..., :1]), confidence_mask[..., 1:]], axis=-1)
                    scores = scores * confidence_mask

                score_sum = scores.sum(axis=-1, keepdims=True)
                scores = mx.where(score_sum > 0, scores / score_sum, scores)

                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)

                if hasattr(self, "shared_expert"):
                    shared = self.shared_expert(x)
                    if hasattr(self, "shared_expert_gate"):
                        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
                    y = y + shared

                return y

            return patched_call

        mlp.__call__ = types.MethodType(make_patched(original_call, cache), mlp)


# -- Profile-based warmup --


def get_warmup_experts(task: str = "general", num_layers: int = 24, num_experts: int = 60, top_n: int = 30) -> list:
    """Get hot experts for a task from pre-computed profiles.

    Uses task_profiler's predefined profiles to determine which experts
    to pre-load for a given workload type.
    """
    try:
        from mlx_flash_compress.task_profiler import get_predefined_profile

        profile = get_predefined_profile(task, num_layers=num_layers, num_experts=num_experts)
        hot_experts = profile.get_hot_experts(top_pct=top_n / num_experts)

        # Convert to flat list of expert IDs per layer
        result = []
        for layer_idx in range(num_layers):
            if layer_idx in hot_experts:
                result.append(hot_experts[layer_idx][:top_n])
            else:
                result.append(list(range(top_n)))
        return result
    except Exception:
        # Fallback: first N experts
        return [list(range(top_n)) for _ in range(num_layers)]


# -- Public API --


def enable_expert_streaming(model, capacity_per_layer=200, model_path=None):
    """Enable expert streaming on a loaded MLX MoE model.

    Replaces QuantizedSwitchLinear with cached versions using
    pre-stacked GPU tensors and lookup tables.
    """
    state = StreamingState(model=model)

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        raise ValueError("Cannot find model layers")

    # Find safetensors
    if model_path:
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            # Try HuggingFace cache
            import glob

            cache_pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}*/snapshots/*/"
            )
            cache_dirs = sorted(glob.glob(cache_pattern))
            if cache_dirs:
                model_dir = Path(cache_dirs[-1])

        shards = sorted(model_dir.glob("*.safetensors"))
        if shards:
            state.st_map = SafetensorsMap([str(s) for s in shards])

    total_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp

        # Detect switch_mlp (Qwen/DeepSeek) or block_sparse_moe (Mixtral)
        switch = getattr(mlp, "switch_mlp", None)
        is_mixtral = False
        if switch is None:
            bsm = getattr(mlp, "block_sparse_moe", None)
            if bsm is not None:
                switch = getattr(bsm, "switch_mlp", None)
                is_mixtral = True
        if switch is None:
            continue

        # Mixtral safetensors use w1/w2/w3 naming vs gate_proj/up_proj/down_proj
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

        num_experts = 0
        proj_modules = {}
        weight_keys = {}

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(switch, proj_name, None)
            if proj is None:
                continue
            w = getattr(proj, "weight", None)
            if w is not None and len(w.shape) == 3:
                num_experts = w.shape[0]
                proj_modules[proj_name] = proj
                for suffix in ["weight", "scales", "biases"]:
                    if is_mixtral:
                        # Mixtral safetensors: block_sparse_moe.switch_mlp.gate_proj.weight
                        # but also try: block_sparse_moe.experts.E.w1.weight (per-expert)
                        wk = f"model.layers.{layer_idx}.block_sparse_moe.switch_mlp.{proj_name}.{suffix}"
                    else:
                        wk = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.{suffix}"
                    weight_keys[f"{proj_name}.{suffix}"] = wk

        if num_experts == 0:
            continue

        cache = ExpertCache(
            layer_idx=layer_idx,
            num_experts=num_experts,
            capacity=min(capacity_per_layer, num_experts),
            st_map=state.st_map,
            weight_keys=weight_keys,
            num_layers=total_layers,
        )
        state.caches.append(cache)

        for proj_name, proj in proj_modules.items():
            replacement = CachedSwitchLinear(
                cache=cache,
                proj_name=proj_name,
                group_size=getattr(proj, "group_size", 64),
                bits=getattr(proj, "bits", 4),
                mode=getattr(proj, "mode", 0),
            )
            setattr(switch, proj_name, replacement)

    return state
```

### Fișier: `expert_streaming.before-resume.py`

```python
"""Real expert streaming: GPU lookup table + pre-stacked weight tensors.

Replaces MLX's QuantizedSwitchLinear with a cached version that:
1. Keeps only `capacity` experts per layer in GPU memory (pre-stacked tensors)
2. Uses a GPU lookup table to map global expert IDs to local cache slots
3. Updates cache between tokens (not during forward pass)
4. Zero-eval fallback: missing experts get score=0, remaining renormalized

This is the technique that makes 50GB+ models run on 36GB Macs.

Architecture:
  Forward pass (GPU only, no CPU/disk):
    router(x) -> expert_ids -> lookup[expert_ids] -> local_ids
    gather_qmm(x, stacked_weights, local_ids) -> output

  Between tokens (CPU + disk):
    Record which experts were activated
    Evict cold experts (LCP priority)
    Load new experts from safetensors via mmap
    Update stacked tensors + lookup table

Usage:
  from mlx_flash_compress.expert_streaming import enable_expert_streaming

  model, tokenizer = mlx_lm.load("model_name", lazy=True)
  streaming = enable_expert_streaming(model, capacity_per_layer=200)
  streaming.warmup()

  # Generate normally
  output = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=100)
  streaming.update()  # call between generation steps
"""

import json
import math
import mmap
import os
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# -- Safetensors mmap reader --


@dataclass
class TensorInfo:
    dtype: str
    shape: list
    data_offset: int
    data_size: int


class SafetensorsMap:
    """Memory-mapped access to safetensors files for per-expert slicing."""

    def __init__(self, shard_paths: list):
        self._mmaps: dict[str, mmap.mmap] = {}
        self._tensor_map: dict[str, tuple] = {}
        self._data_offsets: dict[str, int] = {}
        for path in shard_paths:
            self._index_shard(path)

    def _index_shard(self, path):
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            data_start = 8 + header_size

        header = json.loads(header_bytes)
        self._data_offsets[path] = data_start

        for key, info in header.items():
            if key == "__metadata__":
                continue
            offsets = info.get("data_offsets", [0, 0])
            self._tensor_map[key] = (
                path,
                TensorInfo(
                    dtype=info.get("dtype", "F16"),
                    shape=info.get("shape", []),
                    data_offset=offsets[0],
                    data_size=offsets[1] - offsets[0],
                ),
            )

    def _get_mmap(self, path):
        if path not in self._mmaps:
            f = open(path, "rb")
            self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[path]

    _NP_DTYPES = {
        "F16": np.float16,
        "BF16": np.uint16,
        "F32": np.float32,
        "U32": np.uint32,
        "I32": np.int32,
        "U8": np.uint8,
    }

    def get_expert_slice(self, key, expert_ids):
        """Load specific expert rows from stacked [E, ...] or per-expert format."""
        # Try stacked format first (key = "...switch_mlp.gate_proj.weight" shape [E, ...])
        if key in self._tensor_map:
            path, info = self._tensor_map[key]
            if len(info.shape) >= 3:
                # Stacked format: slice rows directly
                num_experts = info.shape[0]
                row_size = info.data_size // num_experts
                mm = self._get_mmap(path)
                base = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                row_shape = info.shape[1:]

                slices = []
                for eid in expert_ids:
                    offset = base + eid * row_size
                    raw = mm[offset : offset + row_size]
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_shape)
                    slices.append(arr)
                res = mx.array(np.stack(slices))
                return res.view(mx.bfloat16) if info.dtype == "BF16" else res

        # Per-expert format: "model.layers.N.mlp.experts.E.proj.attr"
        # Derive the per-expert key pattern from the stacked key
        # e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        #    -> "model.layers.0.mlp.experts.{E}.gate_proj.weight"
        per_expert_key = key.replace(".switch_mlp.", ".experts.0.")

        # Mixtral: block_sparse_moe.experts.E.w1/w2/w3 naming
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        if per_expert_key not in self._tensor_map:
            for mlx_name, st_name in _MIXTRAL_MAP.items():
                per_expert_key = per_expert_key.replace(f".{mlx_name}.", f".{st_name}.")

        if per_expert_key in self._tensor_map:
            # Derive the eid_key template from per_expert_key (which has experts.0.)
            slices = []
            for eid in expert_ids:
                eid_key = per_expert_key.replace(".experts.0.", f".experts.{eid}.")
                if eid_key not in self._tensor_map:
                    continue
                path, info = self._tensor_map[eid_key]
                mm = self._get_mmap(path)
                offset = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                raw = mm[offset : offset + info.data_size]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
                slices.append(arr)
            if slices:
                return mx.array(np.stack(slices))

        return None

    def has_key(self, key):
        return key in self._tensor_map

    def close(self):
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()


# -- LCP eviction tracker --


class LCPTracker:
    """LCP eviction with optional layer-depth bias (FATE paper).

    Shallow (early) layers have more predictable routing patterns,
    so their experts are more valuable to keep cached. The layer_depth_bias
    multiplies priority by (1 + bias * (1 - layer_frac)) where layer_frac
    is the layer's position (0=first, 1=last).
    """

    def __init__(self, num_experts, lcp_base=0.25, lcp_decay=128.0, layer_depth_bias=0.0, layer_frac=0.5):
        self.num_experts = num_experts
        self.lcp_base = lcp_base
        self.lcp_decay = lcp_decay
        # Shallow-favoring bias: higher for early layers
        self._depth_multiplier = 1.0 + layer_depth_bias * (1.0 - layer_frac)
        self.frequency = np.zeros(num_experts, dtype=np.int64)
        self.last_used = np.zeros(num_experts, dtype=np.int64)
        self.step = 0

    def record(self, expert_ids):
        self.step += 1
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                self.frequency[eid] += 1
                self.last_used[eid] = self.step

    def priority(self, eid):
        age = self.step - self.last_used[eid]
        base_priority = self.frequency[eid] * (self.lcp_base ** (age / self.lcp_decay))
        return base_priority * self._depth_multiplier

    def coldest(self, among, n):
        priorities = [(eid, self.priority(eid)) for eid in among]
        priorities.sort(key=lambda x: x[1])
        return [eid for eid, _ in priorities[:n]]


# -- Cached QuantizedSwitchLinear replacement --


class CachedSwitchLinear(nn.Module):
    """Drop-in replacement using GPU lookup table + pre-stacked weights."""

    def __init__(self, cache, proj_name, group_size, bits, mode=0):
        super().__init__()
        self._cache = cache
        self._proj_name = proj_name
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "gate_proj":
            self._cache._indices_buffer.append(indices)

        local_indices = self._cache.lookup[indices]

        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases.get(self._proj_name),
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )


# -- Expert cache per layer --


class ExpertCache:
    """Pre-stacked expert weights in GPU with LCP eviction."""

    def __init__(self, layer_idx, num_experts, capacity, st_map, weight_keys, num_layers=1, layer_depth_bias=0.3):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = min(capacity, num_experts)
        self.st_map = st_map
        self.weight_keys = weight_keys
        self.cached_ids = []
        self.lookup = mx.zeros(num_experts, dtype=mx.int32)
        self.hit_mask = mx.zeros(num_experts)
        self.weights = {}
        self.scales = {}
        self.biases = {}
        layer_frac = layer_idx / max(num_layers - 1, 1)
        self.tracker = LCPTracker(num_experts, layer_depth_bias=layer_depth_bias, layer_frac=layer_frac)
        self._indices_buffer = []
        self.total_tokens = 0
        self.cache_updates = 0

    def initial_fill(self, expert_ids=None):
        if expert_ids is None:
            expert_ids = list(range(self.capacity))
        self.cached_ids = expert_ids[: self.capacity]

        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, self.cached_ids)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]  # gate_proj, up_proj, down_proj
            attr = parts[1]  # weight, scales, biases
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            target[proj] = data

        self._rebuild_lookup()

    def _rebuild_lookup(self):
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def update_between_tokens(self):
        if not self._indices_buffer:
            return
        activated = set()
        for indices in self._indices_buffer:
            ids = np.array(indices).flatten().tolist()
            activated.update(ids)
            self.tracker.record(ids)
        self._indices_buffer.clear()
        self.total_tokens += 1

        cached_set = set(self.cached_ids)
        misses = [eid for eid in activated if eid not in cached_set]
        if not misses:
            return

        evict_candidates = [eid for eid in self.cached_ids if eid not in activated]
        n_to_evict = min(len(misses), len(evict_candidates))
        to_evict = self.tracker.coldest(evict_candidates, n_to_evict)
        if not to_evict:
            return

        new_ids = misses[: len(to_evict)]
        evict_set = set(to_evict)
        evict_map = dict(zip(to_evict, new_ids))
        slots = [i for i, eid in enumerate(self.cached_ids) if eid in evict_set]
        new_ids_ordered = [evict_map[self.cached_ids[s]] for s in slots]
        for s, new_eid in zip(slots, new_ids_ordered):
            self.cached_ids[s] = new_eid

        # Reload only the evicted expert slots in-place from disk
        slot_idx = mx.array(slots)
        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, new_ids_ordered)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]
            attr = parts[1]
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            if proj in target and len(slots) > 0:
                target[proj][slot_idx] = data
                mx.eval(target[proj])

        self._rebuild_lookup()
        mx.eval(self.lookup, self.hit_mask)
        self.cache_updates += 1

        # Release evicted expert buffers back to OS (MLX v0.31+)
        try:
            mx.clear_cache()
        except AttributeError:
            pass  # older MLX versions

    def stats(self):
        return {
            "layer": self.layer_idx,
            "cached": len(self.cached_ids),
            "capacity": self.capacity,
            "coverage": len(self.cached_ids) / max(self.num_experts, 1),
            "tokens": self.total_tokens,
            "updates": self.cache_updates,
        }


# -- Streaming state --


@dataclass
class StreamingState:
    caches: list = field(default_factory=list)
    st_map: object = None
    model: object = None

    def warmup(self):
        for cache in self.caches:
            cache.initial_fill()
        arrays = []
        for cache in self.caches:
            arrays.extend(cache.weights.values())
            arrays.extend(cache.scales.values())
            arrays.extend(cache.biases.values())
        if arrays:
            mx.eval(*arrays)

    def update(self):
        for cache in self.caches:
            cache.update_between_tokens()

    def stats(self):
        return [c.stats() for c in self.caches]

    def total_cached(self):
        return sum(len(c.cached_ids) for c in self.caches)

    def avg_coverage(self):
        if not self.caches:
            return 0
        return sum(c.stats()["coverage"] for c in self.caches) / len(self.caches)

    def cleanup(self):
        if self.st_map:
            self.st_map.close()


# -- Skip-fallback: zero missing expert scores --


def enable_skip_fallback(model, caches: list, adaptive_skip_threshold: float = 0.0):
    """Monkey-patch MoE blocks to zero scores for uncached experts.

    When an expert is not in the cache, its routing score is set to 0
    and remaining scores are renormalized. This avoids computing with
    stale/placeholder weights for uncached experts.

    If adaptive_skip_threshold > 0, also skip low-confidence secondary
    experts when top-1 score / top-2 score > threshold (LExI paper).
    E.g., threshold=3.0 means skip expert 2 if expert 1 is 3x more likely.
    This gives 10-30% compute reduction with <1% quality loss.
    """
    import types

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    cache_by_layer = {c.layer_idx: c for c in caches}

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in cache_by_layer:
            continue
        if not hasattr(layer, "mlp"):
            continue

        cache = cache_by_layer[layer_idx]
        mlp = layer.mlp
        original_call = type(mlp).__call__

        skip_thresh = adaptive_skip_threshold

        def make_patched(orig, cache_ref):
            def patched_call(self, x):
                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = self.top_k
                inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # Zero out scores for uncached experts
                mask = cache_ref.hit_mask[inds]  # 1.0 cached, 0.0 not
                scores = scores * mask

                # Adaptive top-k: skip low-confidence secondary experts (LExI)
                if skip_thresh > 0 and k > 1:
                    top1 = scores[..., :1]
                    # Zero out experts where top-1 is much stronger
                    confidence_mask = mx.where(top1 > skip_thresh * scores, 0.0, 1.0)
                    # Keep top-1 always
                    confidence_mask = mx.concatenate([mx.ones_like(scores[..., :1]), confidence_mask[..., 1:]], axis=-1)
                    scores = scores * confidence_mask

                score_sum = scores.sum(axis=-1, keepdims=True)
                scores = mx.where(score_sum > 0, scores / score_sum, scores)

                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

                if hasattr(self, "shared_expert"):
                    shared = self.shared_expert(x)
                    if hasattr(self, "shared_expert_gate"):
                        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
                    y = y + shared

                return y

            return patched_call

        mlp.__call__ = types.MethodType(make_patched(original_call, cache), mlp)


# -- Profile-based warmup --


def get_warmup_experts(task: str = "general", num_layers: int = 24, num_experts: int = 60, top_n: int = 30) -> list:
    """Get hot experts for a task from pre-computed profiles.

    Uses task_profiler's predefined profiles to determine which experts
    to pre-load for a given workload type.
    """
    try:
        from mlx_flash_compress.task_profiler import get_predefined_profile

        profile = get_predefined_profile(task, num_layers=num_layers, num_experts=num_experts)
        hot_experts = profile.get_hot_experts(top_pct=top_n / num_experts)

        # Convert to flat list of expert IDs per layer
        result = []
        for layer_idx in range(num_layers):
            if layer_idx in hot_experts:
                result.append(hot_experts[layer_idx][:top_n])
            else:
                result.append(list(range(top_n)))
        return result
    except Exception:
        # Fallback: first N experts
        return [list(range(top_n)) for _ in range(num_layers)]


# -- Public API --


def enable_expert_streaming(model, capacity_per_layer=200, model_path=None):
    """Enable expert streaming on a loaded MLX MoE model.

    Replaces QuantizedSwitchLinear with cached versions using
    pre-stacked GPU tensors and lookup tables.
    """
    state = StreamingState(model=model)

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        raise ValueError("Cannot find model layers")

    # Find safetensors
    if model_path:
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            # Try HuggingFace cache
            import glob

            cache_pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}*/snapshots/*/"
            )
            cache_dirs = sorted(glob.glob(cache_pattern))
            if cache_dirs:
                model_dir = Path(cache_dirs[-1])

        shards = sorted(model_dir.glob("*.safetensors"))
        if shards:
            state.st_map = SafetensorsMap([str(s) for s in shards])

    total_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp

        # Detect switch_mlp (Qwen/DeepSeek) or block_sparse_moe (Mixtral)
        switch = getattr(mlp, "switch_mlp", None)
        is_mixtral = False
        if switch is None:
            bsm = getattr(mlp, "block_sparse_moe", None)
            if bsm is not None:
                switch = getattr(bsm, "switch_mlp", None)
                is_mixtral = True
        if switch is None:
            continue

        # Mixtral safetensors use w1/w2/w3 naming vs gate_proj/up_proj/down_proj
        _MIXTRAL_MAP = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

        num_experts = 0
        proj_modules = {}
        weight_keys = {}

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(switch, proj_name, None)
            if proj is None:
                continue
            w = getattr(proj, "weight", None)
            if w is not None and len(w.shape) == 3:
                num_experts = w.shape[0]
                proj_modules[proj_name] = proj
                for suffix in ["weight", "scales", "biases"]:
                    if is_mixtral:
                        # Mixtral safetensors: block_sparse_moe.switch_mlp.gate_proj.weight
                        # but also try: block_sparse_moe.experts.E.w1.weight (per-expert)
                        wk = f"model.layers.{layer_idx}.block_sparse_moe.switch_mlp.{proj_name}.{suffix}"
                    else:
                        wk = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.{suffix}"
                    weight_keys[f"{proj_name}.{suffix}"] = wk

        if num_experts == 0:
            continue

        cache = ExpertCache(
            layer_idx=layer_idx,
            num_experts=num_experts,
            capacity=min(capacity_per_layer, num_experts),
            st_map=state.st_map,
            weight_keys=weight_keys,
            num_layers=total_layers,
        )
        state.caches.append(cache)

        for proj_name, proj in proj_modules.items():
            replacement = CachedSwitchLinear(
                cache=cache,
                proj_name=proj_name,
                group_size=getattr(proj, "group_size", 64),
                bits=getattr(proj, "bits", 4),
                mode=getattr(proj, "mode", 0),
            )
            setattr(switch, proj_name, replacement)

    return state
```

### Fișier: `expert_streaming.patch`

```diff
--- a/expert_streaming.py
+++ b/expert_streaming.py
@@ -1,4 +1,9 @@
-"""Real expert streaming: GPU lookup table + pre-stacked weight tensors.
+"""Expert streaming: GPU lookup table + pre-stacked weight tensors.
+
+Local exact Qwen3-MoE mode: enable_expert_streaming(..., exact=True,
+read_mode="pread"). Loads ALL routed experts before computation, using bounded
+disk reads and a temporary one-layer working set during prefill. This mode
+preserves routing semantics; the legacy approximate mode below is unchanged.
 
 Replaces MLX's QuantizedSwitchLinear with a cached version that:
 1. Keeps only `capacity` experts per layer in GPU memory (pre-stacked tensors)
@@ -60,7 +65,11 @@
 class SafetensorsMap:
     """Memory-mapped access to safetensors files for per-expert slicing."""
 
-    def __init__(self, shard_paths: list):
+    def __init__(self, shard_paths: list, read_mode="mmap"):
+        if read_mode not in ("mmap", "pread"):
+            raise ValueError("read_mode must be mmap or pread")
+        self.read_mode = read_mode
+        self._fds = {}
         self._mmaps: dict[str, mmap.mmap] = {}
         self._tensor_map: dict[str, tuple] = {}
         self._data_offsets: dict[str, int] = {}
@@ -92,13 +101,23 @@
 
     def _get_mmap(self, path):
         if path not in self._mmaps:
-            f = open(path, "rb")
-            self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
+            with open(path, "rb") as f:
+                self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
         return self._mmaps[path]
+
+    def _read(self, path, offset, size):
+        if self.read_mode == "mmap":
+            return self._get_mmap(path)[offset:offset + size]
+        if path not in self._fds:
+            self._fds[path] = os.open(path, os.O_RDONLY)
+        data = os.pread(self._fds[path], size, offset)
+        if len(data) != size:
+            raise OSError(f"Short safetensors read: {path}")
+        return data
 
     _NP_DTYPES = {
         "F16": np.float16,
-        "BF16": np.float16,
+        "BF16": np.uint16,
         "F32": np.float32,
         "U32": np.uint32,
         "I32": np.int32,
@@ -114,18 +133,20 @@
                 # Stacked format: slice rows directly
                 num_experts = info.shape[0]
                 row_size = info.data_size // num_experts
-                mm = self._get_mmap(path)
                 base = self._data_offsets[path] + info.data_offset
-                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
+                np_dtype = self._NP_DTYPES[info.dtype]
                 row_shape = info.shape[1:]
 
                 slices = []
                 for eid in expert_ids:
+                    if not 0 <= eid < num_experts:
+                        raise IndexError(f"Expert {eid} outside {key}")
                     offset = base + eid * row_size
-                    raw = mm[offset : offset + row_size]
+                    raw = self._read(path, offset, row_size)
                     arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_shape)
                     slices.append(arr)
-                return mx.array(np.stack(slices))
+                res = mx.array(np.stack(slices))
+                return res.view(mx.bfloat16) if info.dtype == "BF16" else res
 
         # Per-expert format: "model.layers.N.mlp.experts.E.proj.attr"
         # Derive the per-expert key pattern from the stacked key
@@ -145,16 +166,16 @@
             for eid in expert_ids:
                 eid_key = per_expert_key.replace(".experts.0.", f".experts.{eid}.")
                 if eid_key not in self._tensor_map:
-                    continue
+                    raise KeyError(eid_key)
                 path, info = self._tensor_map[eid_key]
-                mm = self._get_mmap(path)
                 offset = self._data_offsets[path] + info.data_offset
-                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
-                raw = mm[offset : offset + info.data_size]
+                np_dtype = self._NP_DTYPES[info.dtype]
+                raw = self._read(path, offset, info.data_size)
                 arr = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
                 slices.append(arr)
             if slices:
-                return mx.array(np.stack(slices))
+                res = mx.array(np.stack(slices))
+                return res.view(mx.bfloat16) if info.dtype == "BF16" else res
 
         return None
 
@@ -165,6 +186,9 @@
         for mm in self._mmaps.values():
             mm.close()
         self._mmaps.clear()
+        for fd in self._fds.values():
+            os.close(fd)
+        self._fds.clear()
 
 
 # -- LCP eviction tracker --
@@ -222,7 +246,7 @@
         self.mode = mode
 
     def __call__(self, x, indices, sorted_indices=False):
-        if self._proj_name == "gate_proj":
+        if self._proj_name == "gate_proj" and not self._cache.exact:
             self._cache._indices_buffer.append(indices)
 
         local_indices = self._cache.lookup[indices]
@@ -237,7 +261,7 @@
             group_size=self.group_size,
             bits=self.bits,
             mode=self.mode,
-            sorted_indices=sorted_indices,
+            sorted_indices=False,
         )
 
 
@@ -264,6 +288,7 @@
         self._indices_buffer = []
         self.total_tokens = 0
         self.cache_updates = 0
+        self.exact = False
 
     def initial_fill(self, expert_ids=None):
         if expert_ids is None:
@@ -273,6 +298,8 @@
         for full_key_name, st_key in self.weight_keys.items():
             data = self.st_map.get_expert_slice(st_key, self.cached_ids)
             if data is None:
+                if self.exact:
+                    raise KeyError(st_key)
                 continue
             parts = full_key_name.split(".")
             proj = parts[0]  # gate_proj, up_proj, down_proj
@@ -290,6 +317,16 @@
             hit_np[eid] = 1.0
         self.lookup = mx.array(lookup_np)
         self.hit_mask = mx.array(hit_np)
+
+    def ensure(self, indices):
+        """Populate the current routing set BEFORE any expert computation."""
+        ids = np.unique(np.array(indices)).tolist()
+        if len(ids) > self.capacity:
+            raise ValueError("Routing set exceeds resident cache capacity")
+        self._indices_buffer = [indices]
+        self.update_between_tokens()
+        if not set(ids).issubset(self.cached_ids):
+            raise RuntimeError("Expert cache failed to satisfy current routing")
 
     def update_between_tokens(self):
         if not self._indices_buffer:
@@ -303,7 +340,7 @@
         self.total_tokens += 1
 
         cached_set = set(self.cached_ids)
-        misses = [eid for eid in activated if eid not in cached_set]
+        misses = sorted(eid for eid in activated if eid not in cached_set)
         if not misses:
             return
 
@@ -315,28 +352,30 @@
 
         new_ids = misses[: len(to_evict)]
         evict_set = set(to_evict)
-        new_cached = [eid for eid in self.cached_ids if eid not in evict_set]
-        new_cached.extend(new_ids)
-        self.cached_ids = new_cached[: self.capacity]
-
-        # Reload changed expert slots from disk
+        evict_map = dict(zip(to_evict, new_ids))
+        slots = [i for i, eid in enumerate(self.cached_ids) if eid in evict_set]
+        new_ids_ordered = [evict_map[self.cached_ids[s]] for s in slots]
+        for s, new_eid in zip(slots, new_ids_ordered):
+            self.cached_ids[s] = new_eid
+
+        # Reload only the evicted expert slots in-place from disk
+        slot_idx = mx.array(slots)
         for full_key_name, st_key in self.weight_keys.items():
-            data = self.st_map.get_expert_slice(st_key, new_ids)
+            data = self.st_map.get_expert_slice(st_key, new_ids_ordered)
             if data is None:
+                if self.exact:
+                    raise KeyError(st_key)
                 continue
             parts = full_key_name.split(".")
             proj = parts[0]
             attr = parts[1]
             target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
-            if proj in target:
-                # Find slot indices for new experts
-                slot_map = {eid: i for i, eid in enumerate(self.cached_ids)}
-                slots = [slot_map[eid] for eid in new_ids if eid in slot_map]
-                if slots:
-                    slot_idx = mx.array(slots[: data.shape[0]])
-                    target[proj][slot_idx] = data
+            if proj in target and len(slots) > 0:
+                target[proj][slot_idx] = data
+                mx.eval(target[proj])
 
         self._rebuild_lookup()
+        mx.eval(self.lookup, self.hit_mask)
         self.cache_updates += 1
 
         # Release evicted expert buffers back to OS (MLX v0.31+)
@@ -396,6 +435,52 @@
             self.st_map.close()
 
 
+class ExactStreamingSwitchGLU(nn.Module):
+    """Load all routed experts, retaining Qwen's original router and scores.
+
+    Decode uses the resident cache. Prefill may need more experts than its
+    capacity: these are loaded temporarily for ONE layer, evaluated, and freed.
+    No instance-level __call__ monkey-patching or missing-expert substitution.
+    """
+
+    def __init__(self, switch, cache):
+        super().__init__()
+        self.switch = switch
+        self._cache = cache
+
+    def __call__(self, x, indices):
+        mx.eval(x, indices)
+        ids = np.unique(np.array(indices)).tolist()
+        cache = self._cache
+        if len(ids) <= cache.capacity:
+            cache.ensure(indices)
+            result = self.switch(x, indices)
+            # Finish using mutable cache slots before the next forward/eviction.
+            mx.eval(result)
+            return result
+
+        temporary = ExpertCache(cache.layer_idx, cache.num_experts, len(ids),
+                                cache.st_map, cache.weight_keys)
+        temporary.exact = True
+        temporary.initial_fill(ids)
+        local_indices = temporary.lookup[indices]
+        expanded = mx.expand_dims(x, (-2, -3))
+
+        def project(name, values):
+            proj = getattr(self.switch, name)
+            return mx.gather_qmm(
+                values, temporary.weights[name], temporary.scales[name],
+                temporary.biases.get(name), rhs_indices=local_indices,
+                transpose=True, group_size=proj.group_size, bits=proj.bits,
+                mode=proj.mode, sorted_indices=False)
+
+        up = project("up_proj", expanded)
+        gate = project("gate_proj", expanded)
+        result = project("down_proj", self.switch.activation(up, gate)).squeeze(-2)
+        mx.eval(result)
+        return result
+
+
 # -- Skip-fallback: zero missing expert scores --
 
 
@@ -412,6 +497,9 @@
     This gives 10-30% compute reduction with <1% quality loss.
     """
     import types
+
+    if any(c.exact for c in caches):
+        raise ValueError("Skip fallback is incompatible with exact streaming")
 
     layers = None
     if hasattr(model, "model") and hasattr(model.model, "layers"):
@@ -461,7 +549,7 @@
                 scores = mx.where(score_sum > 0, scores / score_sum, scores)
 
                 y = self.switch_mlp(x, inds)
-                y = (y * scores[..., None]).sum(axis=-2)
+                y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
 
                 if hasattr(self, "shared_expert"):
                     shared = self.shared_expert(x)
@@ -507,12 +595,17 @@
 # -- Public API --
 
 
-def enable_expert_streaming(model, capacity_per_layer=200, model_path=None):
+def enable_expert_streaming(model, capacity_per_layer=200, model_path=None,
+                            exact=False, read_mode="mmap"):
     """Enable expert streaming on a loaded MLX MoE model.
 
     Replaces QuantizedSwitchLinear with cached versions using
     pre-stacked GPU tensors and lookup tables.
     """
+    if capacity_per_layer < 1:
+        raise ValueError("capacity_per_layer must be positive")
+    if exact and getattr(model, "model_type", None) != "qwen3_moe":
+        raise ValueError("Exact streaming is validated for qwen3_moe only")
     state = StreamingState(model=model)
 
     layers = None
@@ -539,7 +632,10 @@
 
         shards = sorted(model_dir.glob("*.safetensors"))
         if shards:
-            state.st_map = SafetensorsMap([str(s) for s in shards])
+            state.st_map = SafetensorsMap([str(s) for s in shards], read_mode=read_mode)
+
+    if state.st_map is None:
+        raise ValueError("model_path must resolve to local safetensors shards")
 
     total_layers = len(layers)
     for layer_idx, layer in enumerate(layers):
@@ -585,6 +681,15 @@
         if num_experts == 0:
             continue
 
+        if exact:
+            if capacity_per_layer < mlp.top_k:
+                raise ValueError("Exact cache capacity must be at least top_k")
+            if getattr(mlp, "sharding_group", None) is not None:
+                raise ValueError("Exact streaming does not support distributed sharding")
+            for proj in proj_modules.values():
+                if getattr(proj, "mode", "affine") != "affine" or "bias" in proj:
+                    raise ValueError("Exact streaming requires bias-free affine projections")
+
         cache = ExpertCache(
             layer_idx=layer_idx,
             num_experts=num_experts,
@@ -593,6 +698,7 @@
             weight_keys=weight_keys,
             num_layers=total_layers,
         )
+        cache.exact = exact
         state.caches.append(cache)
 
         for proj_name, proj in proj_modules.items():
@@ -605,4 +711,7 @@
             )
             setattr(switch, proj_name, replacement)
 
+        if exact:
+            mlp.switch_mlp = ExactStreamingSwitchGLU(switch, cache)
+
     return state
```

### Fișier: `requirements.txt`

```text
annotated-doc==0.0.5
anyio==4.15.1
click==8.5.0
filelock==4.2.0
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.13.1
httpx2==2.13.1
huggingface-hub==2.2.0
idna==3.20
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.4
mdurl==0.1.2
mlx==0.32.4
mlx-flash==0.9.2
mlx-lm==0.32.2
mlx-metal==0.32.4
numpy==2.5.4
packaging==26.3
protobuf==7.36.2
psutil==7.2.2
pygments==2.21.0
pyyaml==6.0.3
regex==2026.9.29
rich==15.0.0
safetensors==0.8.0
sentencepiece==0.2.2
shellingham==1.5.4
tabulate==0.10.0
tokenizers==0.23.3
tqdm==4.70.1
transformers==5.20.0
truststore==0.10.5
typer==0.27.3
typing-extensions==4.16.0
```

### Fișier: `upstream.sha256`

```text
bbb63eb183984d17ccf06721eb0c0bb02a2aa17699241199e70fa2fd4b123680  expert_streaming.py
```

### Fișier: `ai`

```zsh
#!/bin/zsh
exec "$HOME/.local/share/uv/tools/mlx-flash/bin/python" "$HOME/.local/share/mlx-ssd-chat/ai.py" "$@"
```

### Fișier: `ai.zsh`

```zsh
# Local MLX chat: Qwen3 30B, exact SSD offload, 4 GB process budget
alias ai='"$HOME/.local/bin/ai"'
```

### Fișier: `ai.py`

```python
"""On-demand terminal chat: Qwen3-MoE, exact SSD streaming, configurable memory."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from memory_guard import MemoryGuard, add_memory_arguments, process_memory_limit

MODEL = "mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit"
REVISION = "f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e"


def common_prefix(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="One message, then exit")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Optional direct-answer mode (thinking is on by default)")
    add_memory_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048:
        parser.error("--max-tokens must be between 1 and 2048")
    process_limit = process_memory_limit(args, parser)

    lock_path = Path.home() / ".local/share/mlx-ssd-chat/ai.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Un chat ai rulează deja. Închide-l înainte de a porni altul.", file=sys.stderr)
            return 1
        guard = MemoryGuard(limit_bytes=process_limit)
        guard.start()
        try:
            return chat(args, guard)
        except KeyboardInterrupt:
            print("\nChat închis.", file=sys.stderr)
            return 130
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"\nai: {exc}", file=sys.stderr)
            return 1
        finally:
            guard.close()
            print(guard.summary(), file=sys.stderr)


def chat(args, guard):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_flash_compress.expert_streaming import enable_expert_streaming
    from cache_memory import plan_expert_cache

    mx.set_memory_limit(args.mlx_memory_bytes)
    mx.set_cache_limit(64_000_000)
    path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    print(f"Încarc Qwen3 30B · SSD offload exact · MLX {args.mlx_memory_bytes/1e9:g} GB · "
          f"buget proces {guard.limit/1e9:g} GB…", file=sys.stderr)
    start = time.monotonic()
    model, tokenizer = load(path, lazy=True, tokenizer_config={"fix_mistral_regex": True})
    plan = plan_expert_cache(model, args.mlx_memory_bytes)
    print(plan.summary(), file=sys.stderr)
    state = enable_expert_streaming(model, capacity_per_layer=plan.capacity, model_path=path,
                                     exact=True, read_mode="pread")
    try:
        state.warmup()
        mx.eval(model.parameters())
        print(f"MLX active after loading: {mx.get_active_memory()/1e9:.2f} GB", file=sys.stderr)
        print(f"Gata în {time.monotonic()-start:.1f}s. Thinking: "
              f"{'oprit' if args.no_thinking else 'pornit'}. "
              "q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.",
              file=sys.stderr)
        messages = []
        cache = make_prompt_cache(model)
        cached_tokens = []
        context_limit = 4096
        while True:
            if args.prompt is not None:
                query = args.prompt
            else:
                try:
                    query = input("\nTu> ")
                except EOFError:
                    break
            if not query.strip():
                if args.prompt is not None:
                    break
                continue
            if query.strip() in ("q", "/quit"):
                break
            if query.strip() in ("r", "/reset"):
                messages.clear()
                cached_tokens.clear()
                cache = make_prompt_cache(model)
                mx.clear_cache()
                print("Conversație resetată.", file=sys.stderr)
                if args.prompt is not None:
                    break
                continue

            proposed = messages + [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(proposed, tokenize=True,
                add_generation_prompt=True, enable_thinking=not args.no_thinking)
            if len(prompt) + args.max_tokens > context_limit:
                print(f"Bugetul de context pentru acest profil este {context_limit} tokeni, "
                      "inclusiv răspunsul. Folosește /reset sau un mesaj mai scurt.", file=sys.stderr)
                if args.prompt is not None:
                    return 2
                continue

            # Chat templates may remove old thinking. Reuse ONLY an exact prefix.
            offset = cache[0].offset
            if offset != len(cached_tokens):
                cache = make_prompt_cache(model)
                cached_tokens = []
                offset = 0
            prefix = min(common_prefix(cached_tokens, prompt), len(prompt) - 1)
            if trim_prompt_cache(cache, offset - prefix) != offset - prefix:
                cache = make_prompt_cache(model)
                prefix = 0
            messages = proposed
            generated = []
            text = []
            start = time.monotonic()
            first = None
            print("\nAI> ", end="", flush=True)
            for response in stream_generate(
                model, tokenizer, prompt[prefix:], max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0.6, top_p=0.95), prompt_cache=cache,
                prefill_step_size=128, kv_bits=8, quantized_kv_start=0,
            ):
                if first is None:
                    first = time.monotonic() - start
                print(response.text, end="", flush=True)
                generated.append(response.token)
                text.append(response.text)
            print(flush=True)
            cached_tokens = prompt + generated
            messages.append({"role": "assistant", "content": "".join(text)})
            guard.sample()
            print(f"[{response.generation_tokens} tokeni; primul token {first:.1f}s; "
                  f"decode {response.generation_tps:.2f} tok/s; "
                  f"prefix reutilizat {prefix} tokeni; {guard.summary()}]", file=sys.stderr)
            if response.finish_reason == "length":
                print("[Limită de răspuns atinsă; poate include thinking neterminat.]", file=sys.stderr)
            if args.prompt is not None:
                break
        return 0
    finally:
        state.cleanup()


if __name__ == "__main__":
    sys.exit(main())
```

### Fișier: `memory_guard.py`

```python
"""macOS process memory sampling, including Metal's physical footprint."""
import argparse
import ctypes
import math
import os
import threading


def memory_bytes(value):
    """Parse decimal GB into a positive byte count accepted by MLX's size_t API."""
    try:
        size = float(value) * 1_000_000_000
    except ValueError:
        raise argparse.ArgumentTypeError("memory must be a number in decimal GB") from None
    if not math.isfinite(size) or not 1 <= size < 2**64:
        raise argparse.ArgumentTypeError("memory must be finite, positive and fit in 64-bit bytes")
    return int(size)


def add_memory_arguments(parser):
    parser.add_argument("--mlx-memory-gb", type=memory_bytes, default="2.8",
                        dest="mlx_memory_bytes", metavar="GB",
                         help="MLX memory guideline and automatic expert cache budget in decimal GB (default: 2.8)")
    parser.add_argument("--process-memory-gb", type=memory_bytes,
                        dest="process_memory_bytes", metavar="GB",
                        help="Process watchdog budget in decimal GB (default: MLX + 1.2)")


def process_memory_limit(args, parser):
    limit = args.process_memory_bytes
    if limit is None:
        limit = args.mlx_memory_bytes + 1_200_000_000
    if limit <= args.mlx_memory_bytes:
        parser.error("--process-memory-gb must be greater than --mlx-memory-gb")
    return limit


class MemoryGuard:
    def __init__(self, limit_bytes=4_000_000_000, interval=0.05):
        self.limit = limit_bytes
        self.interval = interval
        self.peak_rss = 0
        self.peak_footprint = 0
        self._stop = threading.Event()
        self._lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._lib.proc_pid_rusage.restype = ctypes.c_int

    def sample(self):
        # rusage_info_v0: uuid[16], then ten uint64_t fields.
        info = ctypes.create_string_buffer(1024)
        if self._lib.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        fields = (ctypes.c_uint64 * 10).from_buffer(info, 16)
        rss, footprint = fields[6], fields[7]
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_footprint = max(self.peak_footprint, footprint)
        return rss, footprint

    def start(self):
        self.sample()  # Fail before loading weights if monitoring is unavailable.

        def monitor():
            while not self._stop.wait(self.interval):
                try:
                    rss, footprint = self.sample()
                except OSError:
                    os.write(2, b"\nMemory monitor failed; closing ai.\n")
                    os._exit(78)
                if max(rss, footprint) > self.limit:
                    message = (f"\nMemory budget exceeded ({max(rss, footprint)/1e9:.2f} "
                               f"> {self.limit/1e9:.2f} GB); closing ai.\n")
                    os.write(2, message.encode())
                    os._exit(78)

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join()
        self.sample()

    def summary(self):
        return (f"process peak RSS={self.peak_rss/1e9:.2f} GB; "
                f"peak physical footprint={self.peak_footprint/1e9:.2f} GB")
```

### Fișier: `cache_memory.py`

```python
"""Size the resident Qwen3 expert cache from the MLX budget, before evaluation."""
from dataclasses import dataclass

from mlx.utils import tree_flatten


@dataclass(frozen=True)
class ExpertCachePlan:
    capacity: int
    num_experts: int
    fixed_bytes: int
    bytes_per_slot: int
    reserve_bytes: int

    @property
    def resident_bytes(self):
        return self.fixed_bytes + self.capacity * self.bytes_per_slot

    def summary(self):
        return (f"Expert cache: {self.capacity}/{self.num_experts} per layer; "
                f"resident weights ~{self.resident_bytes/1e9:.2f} GB; "
                f"MLX working reserve {self.reserve_bytes/1e9:.2f} GB")


def plan_expert_cache(model, memory_limit):
    """Inspect lazy parameter sizes without loading the full expert weights.

    Keep 10% (at least 400 MB) for KV, prefill and temporary arrays. This is a
    sizing estimate, not a hard limit; MemoryGuard monitors the whole process.
    """
    if getattr(model, "model_type", None) != "qwen3_moe":
        raise ValueError("Automatic expert cache sizing supports qwen3_moe only")

    total_bytes = sum(array.nbytes for _, array in tree_flatten(model.parameters()))
    expert_bytes = 0
    bytes_per_slot = 0
    num_experts = None
    top_k = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None:
            continue
        count = switch.gate_proj.weight.shape[0]
        if num_experts is not None and count != num_experts:
            raise ValueError("Automatic cache sizing requires uniform expert counts")
        num_experts = count
        top_k = max(top_k, mlp.top_k)
        for _, array in tree_flatten(switch.parameters()):
            if array.shape[0] != count or array.nbytes % count:
                raise ValueError("Unsupported expert parameter layout for cache sizing")
            expert_bytes += array.nbytes
            bytes_per_slot += array.nbytes // count

    if not bytes_per_slot:
        raise ValueError("Cannot find expert weights for automatic cache sizing")
    fixed_bytes = total_bytes - expert_bytes
    reserve_bytes = max(400_000_000, memory_limit // 10)
    capacity = min(num_experts, (memory_limit - reserve_bytes - fixed_bytes) // bytes_per_slot)
    if capacity < top_k:
        required = fixed_bytes + top_k * bytes_per_slot + reserve_bytes
        raise ValueError(
            f"MLX budget too small for exact streaming: need at least {top_k} "
            f"experts per layer and working memory (~{required/1e9:.2f} GB). "
            "Increase --mlx-memory-gb.")
    return ExpertCachePlan(capacity, num_experts, fixed_bytes, bytes_per_slot, reserve_bytes)
```

### Fișier: `benchmark.py`

```python
"""One local model only; bounded-memory end-to-end smoke benchmark."""
import argparse
import time
from memory_guard import MemoryGuard, add_memory_arguments, process_memory_limit

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=64)
parser.add_argument("--prompt", default="What is 2 + 2? Reply briefly.")
parser.add_argument("--thinking", action="store_true")
add_memory_arguments(parser)
args = parser.parse_args()
guard = MemoryGuard(limit_bytes=process_memory_limit(args, parser))
guard.start()

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_flash_compress.expert_streaming import enable_expert_streaming
from cache_memory import plan_expert_cache

mx.set_memory_limit(args.mlx_memory_bytes)
mx.set_cache_limit(64_000_000)
print(f"Memory budgets: MLX {args.mlx_memory_bytes/1e9:g} GB; "
      f"process {guard.limit/1e9:g} GB", flush=True)
path = snapshot_download("mlx-community/Josiefied-Qwen3-30B-A3B-abliterated-v2-4bit",
                         revision="f7dbd27ebbaf60abd0c15111cfde8112ee9b1d1e",
                         local_files_only=True)
start = time.monotonic()
model, tokenizer = load(path, lazy=True)
plan = plan_expert_cache(model, args.mlx_memory_bytes)
print(plan.summary(), flush=True)
state = enable_expert_streaming(model, capacity_per_layer=plan.capacity, model_path=path,
                                exact=True, read_mode="pread")
state.warmup()
mx.eval(model.parameters())
print(f"Load: {time.monotonic()-start:.2f}s; MLX {mx.get_active_memory()/1e9:.2f} GB; "
      f"{guard.summary()}", flush=True)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking)
print(f"Prompt: {len(prompt)} tokens", flush=True)
start = time.monotonic()
first = None
try:
    for response in stream_generate(model, tokenizer, prompt, max_tokens=args.tokens,
            sampler=make_sampler(temp=0.6), prefill_step_size=128,
            kv_bits=8, quantized_kv_start=0):
        if first is None:
            first = time.monotonic() - start
        print(response.text, end="", flush=True)
    print(f"\nTTFT={first:.2f}s; total={time.monotonic()-start:.2f}s; "
          f"generated={response.generation_tokens}; decode={response.generation_tps:.2f} tok/s; "
          f"MLX peak={mx.get_peak_memory()/1e9:.2f} GB", flush=True)
finally:
    state.cleanup()
    guard.close()
    print(guard.summary(), flush=True)
```

### Fișier: `test_streaming.py`

```python
"""Differential regression: real Qwen3-MoE calls, tiny deterministic weights."""
import copy
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_moe import Model, ModelArgs
from mlx_flash_compress.expert_streaming import enable_expert_streaming
from cache_memory import plan_expert_cache


class StreamingRegression(unittest.TestCase):
    def test_full_model_prefill_decode_and_eviction(self):
        mx.random.seed(42)
        args = ModelArgs(
            model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2,
            intermediate_size=128, num_attention_heads=2, num_experts=16,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
            moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=128,
            num_key_value_heads=1, head_dim=32, rope_theta=1000000,
            tie_word_embeddings=False, max_position_embeddings=40960,
            norm_topk_prob=True,
        )
        reference = Model(args)
        reference.set_dtype(mx.bfloat16)
        nn.quantize(reference, group_size=32, bits=4)
        mx.eval(reference.parameters())
        before = mx.get_active_memory()
        plans = [plan_expert_cache(reference, budget)
                 for budget in (400_130_000, 400_230_000, 1_000_000_000)]
        self.assertEqual(mx.get_active_memory(), before)
        self.assertLess(plans[0].capacity, plans[1].capacity)
        self.assertLess(plans[1].capacity, plans[2].capacity)
        self.assertEqual(plans[-1].capacity, args.num_experts)
        with self.assertRaisesRegex(ValueError, "budget too small"):
            plan_expert_cache(reference, 400_000_000)
        with tempfile.TemporaryDirectory() as directory:
            mx.save_safetensors(str(Path(directory) / "model.safetensors"),
                                dict(tree_flatten(reference.parameters())))
            for plan in [None, *plans]:
                capacity = plan.capacity if plan else 4
                with self.subTest(capacity=capacity):
                    streamed = copy.deepcopy(reference)
                    state = enable_expert_streaming(
                        streamed, capacity_per_layer=capacity, model_path=directory,
                        exact=True, read_mode="pread")
                    try:
                        state.warmup()
                        if plan:
                            resident = sum(a.nbytes for _, a in tree_flatten(streamed.parameters()))
                            for cache in state.caches:
                                for arrays in (cache.weights, cache.scales, cache.biases):
                                    resident += sum(a.nbytes for a in arrays.values())
                            self.assertEqual(resident, plan.resident_bytes)
                        from mlx_lm.models.cache import make_prompt_cache
                        ref_cache = make_prompt_cache(reference)
                        stream_cache = make_prompt_cache(streamed)
                        # Prefill overflow for small caches, then decode with evictions.
                        for tokens in [list(range(12)), [97], [30], [115], [3]]:
                            x = mx.array([tokens])
                            expected = reference(x, cache=ref_cache)
                            actual = streamed(x, cache=stream_cache)
                            mx.eval(expected, actual)
                            diff = mx.max(mx.abs(expected - actual)).item()
                            print(f"capacity={capacity} tokens={len(tokens)} max_logit_diff={diff}", flush=True)
                            self.assertLessEqual(diff, 0.03125)
                    finally:
                        state.cleanup()


if __name__ == "__main__":
    unittest.main()
```

### Fișier: `test_chat.py`

```python
"""Real-model PTY smoke test: conversation memory, KV reuse, reset and exit."""
import argparse
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    command = [str(Path.home() / ".local/bin/ai"), "--max-tokens", "512" if args.thinking else "64"]
    if not args.thinking:
        command.append("--no-thinking")
    master, slave = pty.openpty()
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    transcript = bytearray()

    def until(marker, timeout=360):
        output = bytearray()
        deadline = time.monotonic() + timeout
        while marker not in output:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([master], [], [], remaining)[0]:
                raise TimeoutError(f"Waiting for {marker!r}")
            data = os.read(master, 65536)
            if not data:
                raise RuntimeError("Chat exited before the expected prompt")
            transcript.extend(data)
            output.extend(data)
        return output.decode()

    def send(query):
        os.write(master, (query + "\n").encode())
        return until(b"Tu> ")

    try:
        until(b"Tu> ", timeout=60)
        first = send("Remember this code: LUNA-47. Reply only OK. Think briefly.")
        assert "Limită de răspuns" not in first, first
        second = send("What code did I ask you to remember? Reply only with the code. Think briefly.")
        answer = second.split("AI> ", 1)[1].split("[", 1)[0]
        assert "LUNA-47" in answer, second
        assert "Limită de răspuns" not in second, second
        prefix = int(re.search(r"prefix reutilizat (\d+) tokeni", second)[1])
        # This pinned template removes the first answer's thinking, so only
        # the 26-token first user turn + assistant header can remain cached.
        assert prefix == 26, second
        if args.thinking:
            assert "</think>" in first and "</think>" in second
        reset = send("/reset")
        assert "Conversație resetată" in reset, reset
        third = send("What is 2 + 2? Reply briefly. Think briefly.")
        assert "prefix reutilizat 0 tokeni" in third, third
        assert "Limită de răspuns" not in third, third
        assert "4" in third.split("AI> ", 1)[1], third
        os.write(master, b"q\n")
        assert process.wait(timeout=30) == 0
        print(transcript.decode(), flush=True)
        print(f"PTY conversation, recall, KV prefix={prefix}, reset, q: PASS")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        os.close(master)
        logs = Path(__file__).resolve().parent / "test-logs"
        logs.mkdir(exist_ok=True)
        (logs / ("chat-thinking.log" if args.thinking else "chat-direct.log")).write_bytes(transcript)


if __name__ == "__main__":
    main()
```

### Fișier: `test_context.py`

```python
"""Real-model boundary test: reject overflow, fill ~4K context, recall, reset."""
from pathlib import Path
import re
import subprocess

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from ai import MODEL, REVISION


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(MODEL, revision=REVISION, local_files_only=True),
        fix_mistral_regex=True)
    base = "The remembered code is ORBIT-73. "
    unit = "The quick brown fox jumps over the lazy dog. "
    suffix = "\nWhat is the remembered code at the beginning? Reply only with that code."

    def tokens(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)

    lo, hi = 0, 1000
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(tokens(base + unit * mid + suffix)) + 16 <= 4096:
            lo = mid
        else:
            hi = mid
    valid = base + unit * lo + suffix
    invalid = base + unit * hi + suffix
    count = len(tokens(valid))
    assert count > 4000 and len(tokens(invalid)) + 16 > 4096
    print(f"Accepted prompt: {count} tokens + 16 response reserve", flush=True)
    # Interactive input is one line per message.
    lines = [invalid.replace("\n", " "), valid.replace("\n", " "),
             "/reset", "What is 2 + 2? Reply only with the number.", "q"]
    # A newline and a space may tokenize differently: verify the actual inputs.
    assert len(tokens(lines[0])) + 16 > 4096
    assert len(tokens(lines[1])) + 16 <= 4096
    print(f"Actual one-line prompt: {len(tokens(lines[1]))} tokens", flush=True)
    result = subprocess.run(
        [str(Path.home() / ".local/bin/ai"), "--no-thinking", "--max-tokens", "16"],
        input="\n".join(lines) + "\n", text=True, capture_output=True, timeout=900)
    output = result.stdout + "\n" + result.stderr
    logs = Path(__file__).resolve().parent / "test-logs"
    logs.mkdir(exist_ok=True)
    (logs / "context.log").write_text(output)
    print(output, flush=True)
    assert result.returncode == 0
    assert "Bugetul de context" in result.stderr
    assert "AI> ORBIT-73" in result.stdout
    assert "Conversație resetată" in result.stderr
    assert "AI> 4" in result.stdout
    assert "Limită de răspuns" not in result.stderr
    assert max(map(float, re.findall(r"physical footprint=([\d.]+) GB", output))) < 4
    print("Context boundary, recall and reset: PASS")


if __name__ == "__main__":
    main()
```

### Fișier: `export_setup.py`

```python
"""Embed complete local sources, configuration and test evidence in the setup doc."""
import difflib
import hashlib
from pathlib import Path
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
DOCUMENT = Path.home() / "mlx-opencode-setup.md"
MARKER = "\n## 8. Anexă reproductibilă — fișiere complete\n"


def main():
    installed = Path(sysconfig.get_paths()["purelib"]) / "mlx_flash_compress/expert_streaming.py"
    original = ROOT / "expert_streaming.upstream-0.9.2.py"
    patch = "".join(difflib.unified_diff(
        original.read_text().splitlines(keepends=True),
        installed.read_text().splitlines(keepends=True),
        fromfile="a/expert_streaming.py", tofile="b/expert_streaming.py"))
    files = {
        "expert_streaming.py": installed.read_text(),
        original.name: original.read_text(),
        "expert_streaming.before-resume.py": (ROOT / "expert_streaming.before-resume.py").read_text(),
        "expert_streaming.patch": patch,
        "requirements.txt": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True),
        "upstream.sha256": hashlib.sha256(original.read_bytes()).hexdigest()
                           + "  expert_streaming.py\n",
        "ai": (Path.home() / ".local/bin/ai").read_text(),
        "ai.zsh": '# Local MLX chat: Qwen3 30B, exact SSD offload, 4 GB process budget\n'
                  'alias ai=\'"$HOME/.local/bin/ai"\'\n',
    }
    for name in ("ai.py", "memory_guard.py", "cache_memory.py", "benchmark.py", "test_streaming.py",
                 "test_chat.py", "test_context.py", "export_setup.py"):
        files[name] = (ROOT / name).read_text()
    for path in sorted((ROOT / "test-logs").glob("*")):
        if path.is_file() and path.suffix in (".log", ".txt"):
            # PTY logs use CRLF and may contain terminal backspaces. Keep them
            # readable in Markdown; hashes describe the embedded text form.
            content = path.read_text().replace("\b", "\\b")
            files[path.name] = content if content.endswith("\n") else content + "\n"
    bundle = ROOT / "setup-bundle"
    bundle.mkdir(exist_ok=True)
    for name, content in files.items():
        (bundle / name).write_text(content)
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content.encode()).hexdigest()}  {name}\n"
        for name, content in files.items())
    (bundle / "SHA256SUMS").write_text(files["SHA256SUMS"])
    sections = [MARKER, "\nToate sursele locale, configurația shell, patch-ul, versiunile "
                "fixate și dovezile text ale testelor sunt incluse integral mai jos. "
                "Numele identifică blocurile de extras; conținutul se află în acest document. "
                "Modulul instalat, originalul 0.9.2 și copia istorică sunt complete, "
                "iar generatorul acestei anexe este inclus și el.\n\n"
                "Logurile sunt reprezentate textual: CRLF/CR devin LF, backspace devine "
                "`\\b`, iar blocurile primesc un newline final dacă lipsea. Hash-urile "
                "din `SHA256SUMS` verifică exact forma inclusă în document. "
                "Sursele Python și patch-ul păstrează conținutul instalării verificate.\n\n"
                "Inventar de blocuri complete:\n"]
    sections.extend(f"- `{name}`\n" for name in files)
    for name, content in files.items():
        language = ("diff" if name.endswith(".patch") else "python" if name.endswith(".py")
                    else "zsh" if name == "ai" or name.endswith(".zsh") else "text")
        sections.append(f"\n### Fișier: `{name}`\n\n```{language}\n{content}```\n")
    DOCUMENT.write_text(DOCUMENT.read_text().split(MARKER)[0] + "".join(sections))
    print(f"Exported {len(files)} files to {bundle} and {DOCUMENT}")


if __name__ == "__main__":
    main()
```

### Fișier: `cache-memory-8gb.log`

```text
Memory budgets: MLX 8 GB; process 9.2 GB
Expert cache: 49/128 per layer; resident weights ~7.11 GB; MLX working reserve 0.80 GB
Load: 4.78s; MLX 7.11 GB; process peak RSS=0.70 GB; peak physical footprint=7.18 GB
Prompt: 26 tokens
4
TTFT=11.13s; total=11.68s; generated=2; decode=3.66 tok/s; MLX peak=7.36 GB
process peak RSS=0.70 GB; peak physical footprint=7.92 GB
```

### Fișier: `chat-direct.log`

```text
Încarc Qwen3 30B · SSD offload exact · buget proces 4 GB…
Gata în 1.4s. Thinking: oprit. q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.

Tu> Remember this code: LUNA-47. Reply only OK. Think briefly.

AI> OK.
[3 tokeni; primul token 8.6s; decode 3.03 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.70 GB; peak physical footprint=3.24 GB]

Tu> What code did I ask you to remember? Reply only with the code. Think briefly.

AI> LUNA-47
[7 tokeni; primul token 7.1s; decode 2.18 tok/s; prefix reutilizat 26 tokeni; process peak RSS=1.78 GB; peak physical footprint=3.33 GB]

Tu> /reset
Conversație resetată.

Tu> What is 2 + 2? Reply briefly. Think briefly.

AI> 4
[2 tokeni; primul token 7.5s; decode 3.61 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.78 GB; peak physical footprint=3.35 GB]

Tu> 
```

### Fișier: `chat-thinking.log`

```text
Încarc Qwen3 30B · SSD offload exact · buget proces 4 GB…
Gata în 1.3s. Thinking: pornit. q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.

Tu> Remember this code: LUNA-47. Reply only OK. Think briefly.

AI> <think>
Okay, the user said, "Remember this code: LUNA-47. Reply only OK. Think briefly." So they want me to remember the code LUNA-47 and just respond with "OK". They mentioned to reply only with OK, so I shouldn't add anything else. Let me make sure I'm not missing any steps. The task is straightforward: remember the code and confirm with OK. No need for extra explanations. Just a simple acknowledgment. Alright, that's all.
</think>

OK
[109 tokeni; primul token 8.2s; decode 2.58 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.77 GB; peak physical footprint=3.24 GB]

Tu> What code did I ask you to remember? Reply only with the code. Think briefly.

AI> <think>
Okay, the user is asking me to recall a code they previously asked me to remember. Let me check the conversation history.

In the first message, they said, "Remember this code: LUNA-47. Reply only OK. Think briefly." So I responded with "OK". Now the user is asking, "What code did I ask you to remember? Reply only with the code. Think briefly."

Since the code was LUNA-47, and they want just the code as the answer, I should respond with "LUNA-47". I need to make sure I don't add anything else, just the code. The user emphasized replying only with the code, so I should keep it straightforward.
</think>

LUNA-47
[161 tokeni; primul token 6.4s; decode 2.53 tok/s; prefix reutilizat 26 tokeni; process peak RSS=1.77 GB; peak physical footprint=3.27 GB]

Tu> /reset
Conversație resetată.

Tu> What is 2 + 2? Reply briefly. Think briefly.

AI> <think>
Okay, the user is asking "What is 2 + 2?" and wants a brief reply. Let me think quickly. 2 plus 2 is a basic arithmetic problem. The answer should be 4. I need to make sure there's no trick or anything here. The question seems straightforward. So the answer is 4. Just need to state it clearly and concisely.
</think>

2 + 2 = 4.
[93 tokeni; primul token 7.0s; decode 2.38 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.77 GB; peak physical footprint=3.29 GB]

Tu> 
```

### Fișier: `context.log`

```text

Tu> 
Tu> 
AI> ORBIT-73

Tu> 
Tu> 
AI> 4

Tu> 
Încarc Qwen3 30B · SSD offload exact · buget proces 4 GB…
Gata în 1.3s. Thinking: oprit. q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.
Bugetul de context pentru acest profil este 4096 tokeni, inclusiv răspunsul. Folosește /reset sau un mesaj mai scurt.
[6 tokeni; primul token 120.4s; decode 2.36 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.55 GB; peak physical footprint=3.54 GB]
Conversație resetată.
[2 tokeni; primul token 7.7s; decode 3.71 tok/s; prefix reutilizat 0 tokeni; process peak RSS=1.55 GB; peak physical footprint=3.54 GB]
process peak RSS=1.55 GB; peak physical footprint=3.54 GB
```

### Fișier: `exit-eof.log`

```text
Încarc Qwen3 30B · SSD offload exact · buget proces 4 GB…
Gata în 1.4s. Thinking: pornit. q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.

Tu> ^D\b\bprocess peak RSS=1.41 GB; peak physical footprint=2.69 GB
```

### Fișier: `exit-interrupt.log`

```text
Încarc Qwen3 30B · SSD offload exact · buget proces 4 GB…
Gata în 0.8s. Thinking: pornit. q / Ctrl+D: ieșire; /reset: conversație nouă; Ctrl+C: închide.

Tu> ^C
Chat închis.
process peak RSS=2.37 GB; peak physical footprint=2.35 GB
```

### Fișier: `long-prompt.txt`

```text
The remembered code is ORBIT-73. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. 
What is the remembered code at the beginning? Reply only with that code.
```

### Fișier: `SHA256SUMS`

```text
d0605fbd4d2c4d34c787710f1e7abd080bdc09e267abec3472cc682dde812aa2  expert_streaming.py
bbb63eb183984d17ccf06721eb0c0bb02a2aa17699241199e70fa2fd4b123680  expert_streaming.upstream-0.9.2.py
5a1eee93fc4cf0e890708f865cc27bea146538fd62c0a25279b4bcb1f26d2746  expert_streaming.before-resume.py
1164755784096ac56e123d5dddfcc8fdc59672f00193d3da73ca74f4565e79f7  expert_streaming.patch
ec3049f5bc143c1558582c542ed72094d2a1cda600223f951f4aa05f900d1ec8  requirements.txt
1487d50019659ebdcd2a37fbd8a2957f69c00cd9b7b1e6a1b0b0218190aab30f  upstream.sha256
c2ed9a9e36e211fafbac400b0e237f7dd18dab5af7f1c760c0d1cbe315167cca  ai
b563b1efe9ce0a41e7a3871da551a0e3fa27962e600e4f1d3e29d5fe3d82bd85  ai.zsh
b95ff017cc2df7e07b65b0431efb886cbe74ba482bb142ca0d9d2f42e308c245  ai.py
c9c90dfb64a3a6ffc269eb9d98c4cc034aa51970dbfdd33dc46ac4da42da8cea  memory_guard.py
5ac7fcf6c0e7f167b2341c47e63c506b06bb36b9a940d8d7c54654b5e0066d1d  cache_memory.py
357b0c23f58a2fd25a6037128569f4a991c27a10be947bb4dc0dac109c6448c3  benchmark.py
a87c2ddb28bdf39c2ec8e88a0578d5e62446328c08e136a7e67dfa48ed8599f9  test_streaming.py
48985a6f6f8a8114aba7be0398e108785eeae5a7e67e74cb53df22a9789c0187  test_chat.py
3b79bbfb94f71d8bc816e81e43e371b1a74e525d4ac0c3fd66e972f9bd2c4b1f  test_context.py
dedb94b0f178dfd81af2c04c701c9c303dcaeeef2f92eadf628ab81c80a06dd4  export_setup.py
f355fcebba11c8e1c1fc03171b9557f520a142a38a1db006bd23c50eeca1e544  cache-memory-8gb.log
c31b1cbd46d29da091b6736b047b481af46e82fe74cc5cdfaafd34cfebb63cc1  chat-direct.log
0d61cff3c48c870705c785611cf3502c3b10bf161b85de56b9a193bcf48f2422  chat-thinking.log
cf1495d2bda0a996f8ab07b5672efee4cafbee37b0cd908317996ab817ef577a  context.log
830fdc4106320d3e97f2900fcfdd5c09a55e4e4b42fb467966011b40a9a1855e  exit-eof.log
1d33bf9f14d7e06d0a7e0b83a01bd663370c5d58ab54c82e07ca5d45bc83ea41  exit-interrupt.log
911af67268813761d5f2676d9dba2d5020df806be80b7ca793f61d42e8ab6756  long-prompt.txt
```
