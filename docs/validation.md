# Validare pe gazda curentă

Gazdă: Apple M3, 16 GiB RAM, macOS 15.5. Python 3.12.13.

## Confirmat

- Anexa originală: 24 fișiere extrase, 23 hash-uri verificate.
- Sursa MLX-Flash din tag-ul fixat este identică cu originalul din document.
- Wheel MLX-Flash construit local; importurile, conținutul wheel-ului și GPU-ul verificate.
- Regresie diferențială: capacități 4, 6, 13, 16; prefill și 4 pași decode;
  diferență maximă logits 0,0 în fiecare probă.
- BF16 stacked/per-expert, `pread`/`mmap`, indici invalizi și citire incompletă.
- Argumente CLI invalide, lock comun chat/benchmark, watchdog cu ieșire 78.
- Typecheck: 0 erori la ultima verificare înaintea integrării finale.

## În curs

- Descărcarea modelului și probele reale de chat/context/thinking.
- Build nativ MLX: compilatorul Metal lipsește; utilizatorul instalează Xcode.
- Comparațiile controlate și validarea finală a exportului.

Logurile noi sunt generate în `artifacts/`; capturile din `reference/` sunt istorice.
