# Build și proveniență

## Baze fixate

- MLX: `https://github.com/ml-explore/mlx.git`, tag `v0.32.4`, commit
  `36969c094354086e15d2a327a9c50feb20b9cf11`.
- MLX-Flash: `https://github.com/szibis/MLX-Flash.git`, tag `v0.9.2`, commit
  `c3561916ae2e4fbfd2278483d8be570b4cce70f7`.
- SHA-256 streaming upstream:
  `bbb63eb183984d17ccf06721eb0c0bb02a2aa17699241199e70fa2fd4b123680`.

Commitul inițial din acest proiect importă sursa upstream MLX-Flash;
următorul aplică numai corecțiile verificate în document. Directorul `reference/`
este o extragere nemodificată, verificată cu toate cele 23 de hash-uri din anexă.
`scripts/extract_reference.py` poate repeta verificarea.

Pentru un checkout MLX-Flash separat, cu fixurile ca commit local:

```sh
git clone --branch v0.9.2 https://github.com/szibis/MLX-Flash.git .sources/MLX-Flash
git -C .sources/MLX-Flash switch -c local/exact-ssd c3561916ae2e4fbfd2278483d8be570b4cce70f7
cmp .sources/MLX-Flash/mlx_flash_compress/expert_streaming.py reference/expert_streaming.upstream-0.9.2.py
cp reference/expert_streaming.py .sources/MLX-Flash/mlx_flash_compress/expert_streaming.py
git -C .sources/MLX-Flash add mlx_flash_compress/expert_streaming.py
git -C .sources/MLX-Flash commit -m 'Apply verified exact SSD streaming patch'
```

Build-ul curent folosește copia completă versionată din `vendor/mlx-flash`,
deci reproducerea nu depinde de un commit local nepublicat al acestui checkout.

## Build nativ

Instrucțiunile upstream sunt în checkout-ul MLX: `docs/src/install.rst`,
`setup.py` și `pyproject.toml`. Scriptul proiectului folosește backend-ul PEP 517
upstream, fără modificări C++/Metal. Comportamentul de sincronizare de la limita
MLX nu este tratat ca defect; nu există un fix nativ justificat de probele inițiale.

`scripts/build.py native`:

1. Verifică platforma și disponibilitatea compilatorului Metal.
2. Creează `.build-venv` cu Python 3.12.13 și cele 8 pachete din lock-ul de build.
3. Construiește MLX-Flash din copia versionată și verifică hash-ul patch-ului.
4. Clonează/verifică checkout-ul MLX fixat și curat.
5. Construiește frontend-ul cu `MLX_BUILD_FRONTEND_PACKAGE=1`, apoi backend-ul
   cu `MLX_BUILD_BACKEND_PACKAGE=1`. În ambele cazuri: `PYPI_RELEASE=1`,
   `DEBUG=0`, `MLX_BUILD_METAL=ON`, `MLX_BUILD_CPU=ON`, `MLX_METAL_JIT=OFF`, Release.
6. Folosește versiunea macOS a gazdei drept deployment target și 2 joburi implicit.
7. Instalează toate cele trei wheel-uri locale în `.venv`, cu celelalte pachete fixate.
8. Compară conținutul instalat cu fiecare wheel, verifică GPU-ul și calea libmlx.

Toolchain-ul efectiv, opțiunile, commiturile, logul și hash-urile wheel-urilor sunt
în `artifacts/native*.json`, `artifacts/logs/` și directoarele wheel-urilor.
Sursa exactă este fixată; SDK-ul și compilatorul sunt înregistrate, nu instalate
automat. Folosește aceleași versiuni pentru a reproduce configurația măsurată.
Un build cu alt SDK poate produce alte hash-uri și necesită validare proprie.

La schimbarea SDK-ului/configurației, folosește un checkout MLX nou sau elimină
doar directorul său generat `build/` înainte de rebuild pentru a evita un cache
CMake din alt toolchain. Nu modifica sursele upstream pentru a ocoli lipsa Metal.

## Validare

`scripts/validate.py --real-model` rulează secvențial regresiile numerice,
validarea CLI, protecția memoriei, concurența, smoke test, conversație directă și
cu thinking, context aproape de 4096 tokeni, reset, Ctrl+D și Ctrl+C.
Scripturile PTY folosesc interpretorul care le lansează, astfel încât testarea
unui mediu izolat să nu folosească accidental launcherul altui profil.

Testul de cache sincronizează GPU-ul înaintea măsurării memoriei active;
altfel eliberarea operațiilor precedente putea modifica eșantionul în timpul
planificării, deși planificatorul nu aloca memorie.

`scripts/compare_profiles.py` înregistrează separat efectul limitei alocatorului
la cache fix și efectul profilului cu cache automat. Nu se compară direct
timpii de pe M3 cu capturile istorice de pe M4 Pro.
