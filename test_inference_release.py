#!/usr/bin/env python3
"""Test di inferenza per il rilascio di Sentinel R-08 v7.2a dai pesi safetensors locali."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_geodesic import R08Model, R08Tokenizer

def main():
    print("1. Caricamento tokenizer da vocab_stages...")
    tokenizer = R08Tokenizer.from_pretrained(ROOT)
    print("2. Caricamento modello Sentinel R-08 v7.2a da model.safetensors...")
    model = R08Model.from_pretrained(ROOT)
    print(f"Modello caricato con successo! Parametri: {sum(p.numel() for p in model.parameters()):,}")

    prompts = [
        "Utente: Ciao, chi sei?\nAssistente:",
        "La capitale d'Italia è",
        "Per riparare un cilindro pneumatico Festo occorre",
    ]

    for p in prompts:
        print(f"\n--- PROMPT: {p.strip()} ---")
        risposta = model.generate(
            prompt=p,
            tokenizer=tokenizer,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        print(f"RISPOSTA:\n{risposta.strip()}")

if __name__ == "__main__":
    main()
