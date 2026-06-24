import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. Inicializa o modelo de embedding
print("Carregando o modelo de embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Nosso dataset de intenções
dataset_intencoes = {
    "saudacao": [
        "Olá",
        "Oi",
        "Bom dia",
        "Boa tarde",
        "E aí"
    ],
    "cancelar_conta": [
        "Quero cancelar minha assinatura",
        "Como faço para excluir minha conta?",
        "Desejo encerrar meu contrato",
        "Não quero mais o serviço",
        "Pedir reembolso e cancelar"
    ],
    "suporte_tecnico": [
        "Meu aplicativo quebrou",
        "Está dando erro",
        "Não consigo acessar",
        "O sistema caiu",
        "A página está travada"
    ]
}

# 3. Mapeia e vetoriza de forma limpa e performática
print("Mapeando e vetorizando as intenções de referência...")
frases_referencia = []
intencoes_referencia = []

for intencao, exemplos in dataset_intencoes.items():
    for frase in exemplos:
        frases_referencia.append(frase)
        intencoes_referencia.append(intencao)

# Gera os embeddings de todas as frases de referência de uma vez só
embeddings_referencia = model.encode(frases_referencia, convert_to_tensor=True)

# 4. Função principal de classificação usando utilitários nativos
def classificar_intencao(texto_usuario):
    # Transforma a entrada do usuário em vetor
    vetor_usuario = model.encode(texto_usuario, convert_to_tensor=True)
    
    # Calcula a similaridade de cosseno contra TODOS os embeddings de uma só vez
    scores_similaridade = util.cos_sim(vetor_usuario, embeddings_referencia)[0]
    
    # Pega o índice do maior score encontrado
    indice_maior_score = int(np.argmax(scores_similaridade.cpu().numpy()))
    maior_score = float(scores_similaridade[indice_maior_score])
    
    # Retorna a intenção mapeada naquele índice
    intencao_detectada = intencoes_referencia[indice_maior_score]
    
    return intencao_detectada, maior_score

# --- Testando o Sistema ---
print("\n--- Sistema Pronto para Testes ---")

testes = [
    "Fala dev, beleza?",                     
    "Quero deletar meu perfil do sistema",   
    "O app quebrou e não abre",              
    "Quero comprar um carro"                 
]

for frase in testes:
    intencao, score = classificar_intencao(frase)
    print(f"\nEntrada: '{frase}'")
    print(f"👉 Intenção Detectada: {intencao} (Confiança: {score:.4f})")