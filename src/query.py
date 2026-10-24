import argparse
import sys

from rag_pipeline import ROBOS, gerar_resposta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # acentos no terminal do Windows

    parser = argparse.ArgumentParser(description="Pergunte algo sobre a documentação dos robôs.")
    parser.add_argument("pergunta", help='ex.: "como desativar o desvio de obstáculos?"')
    parser.add_argument("--robo", choices=list(ROBOS), help="busca só na documentação deste robô")
    args = parser.parse_args()

    resposta, fontes = gerar_resposta(args.pergunta, args.robo)

    print("\n=== RESPOSTA ===\n")
    print(resposta)

    print("\n=== FONTES (menor score = mais parecido) ===\n")
    for doc, score in fontes:
        m = doc.metadata
        print(f"[{score:.3f}] {ROBOS[m['robot']]} | {m['source']} | pág. {m['page_number']}")
        print("   ", doc.page_content[:150].replace("\n", " "), "...\n")
