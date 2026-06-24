import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Inicializa o modelo de embedding (leve, rápido e roda localmente)
print("Carregando o modelo de embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Define o nosso "banco de dados" de intenções com exemplos
dataset_intencoes = {
    "saudacao": [
        "Olá, como vai?",
        "Oi, bom dia",
        "E aí, tudo bem?",
        "Hello!",
        "Quero iniciar o atendimento"
    ],
    "cancelar_conta": [
        "Quero cancelar minha assinatura",
        "Como faço para excluir minha conta?",
        "Desejo encerrar meu contrato",
        "Não quero mais o serviço",
        "Pedir reembolso e cancelar"
    ],
    "suporte_tecnico": [
        "Meu aplicativo não está funcionando",
        "Está dando erro na tela de login",
        "Não consigo acessar minha conta",
        "O sistema caiu",
        "A página está travada"
    ]
}

# 3. Pré-calcula os embeddings de referência (Treinamento do classificador)
print("Mapeando e vetorizando as intenções de referência...")
vencedores_por_intencao = []

for intencao, exemplos in dataset_intencoes.items():
    # Gera os vetores para cada frase de exemplo daquela intenção
    embeddings_exemplos = model.encode(exemplos)
    for emb in embeddings_exemplos:
        vencedores_por_intencao.append({
            "intencao": intencao,
            "vetor": emb
        })

# 4. Função matemática para calcular a Similaridade de Cosseno
def similaridade_cosseno(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 5. Função principal de classificação
def classificar_intencao(texto_usuario):
    # Transforma a entrada do usuário em vetor
    vetor_usuario = model.encode(texto_usuario)
    
    maior_similaridade = -1
    intencao_vencedora = "desconhecida"
    
    # Compara o vetor do usuário com TODOS os vetores do dataset
    for item in vencedores_por_intencao:
        sim = similaridade_cosseno(vetor_usuario, item["vetor"])
        
        # Se a similaridade for maior que a anterior, atualiza o vencedor
        if sim > maior_similaridade:
            maior_similaridade = sim
            intencao_vencedora = item["intencao"]
            
    return intencao_vencedora, maior_similaridade

# --- Testando o Sistema ---
print("\n--- Sistema Pronto para Testes ---")

testes = [
    "Fala dev, beleza?",                     # Deve aproximar de 'saudacao'
    "Quero deletar meu perfil do sistema",   # Deve aproximar de 'cancelar_conta'
    "O app quebrou e não abre",              # Deve aproximar de 'suporte_tecnico'
    "Quero comprar um carro"                 # Fora do escopo, mas vai achar a "menos pior" (Veremos isso no Repo 2)
]
    
for frase in testes:
    intencao, score = classificar_intencao(frase)
    print(f"\nEntrada: '{frase}'")
    print(f"👉 Intenção Detectada: {intencao} (Confiança: {score:.4f})")