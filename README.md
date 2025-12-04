# 🚆 Sistema de Monitoramento de Trens Urbanos (IoT + Microservices)

> Trabalho de Conclusão de Curso (TCC)

Este projeto consiste em uma solução completa de IoT (Internet das Coisas) para monitoramento em tempo real do transporte ferroviário urbano. O sistema captura dados telemétricos (localização e lotação) diretamente dos trens, processa-os em uma arquitetura de microsserviços e disponibiliza previsões de chegada e status para os usuários via aplicativo Android.

---

## 🛠️ Arquitetura e Tecnologias

O projeto foi desenvolvido seguindo o padrão de Arquitetura de Microsserviços, dividido em três camadas principais:

### 1. Hardware (Edge Computing)
* **Dispositivo:** Raspberry Pi 3B.
* **Sensores:** Módulo GPS (Serial) e Câmera USB.
* **Software:** Python + OpenCV.
* **Funcionalidade:** Processamento de imagem local (Haar Cascade) para contagem de passageiros e leitura de coordenadas GPS. Envio de dados via HTTP.

### 2. Backend (Cloud/Server)
* **Linguagem:** Python 3.9.
* **Framework:** FastAPI.
* **Banco de Dados:** PostgreSQL.
* **Infraestrutura:** Docker e Docker Compose (Orquestração).
* **Serviços:**
    * `auth_service`: Autenticação e Gestão de Usuários (JWT).
    * `iot_service`: Ingestão de dados, Cálculo de ETA (Fórmula de Haversine) e Lógica de Contingência (Estimativa sem GPS).
    * `report_service`: Gestão de ocorrências e estatísticas.

### 3. Frontend (Mobile)
* **Plataforma:** Android Nativo (Java).
* **Comunicação:** Retrofit (REST API).
* **Recursos:**
    * Dashboard de Linhas e Estações.
    * Visualização de Previsão de Chegada (Timer em Tempo Real).
    * Indicador Visual de Lotação (Semáforo).
    * Módulo de Report de Problemas pelo Usuário.
    * Gráficos Estatísticos (MPAndroidChart).

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos
* **Docker Desktop** instalado e rodando.
* **Android Studio** (versão recente com SDK Java atualizado).
* **Python 3.9+** (Opcional, apenas para rodar o script de hardware simulado).

### Passo 1: Inicializar o Backend
Todos os serviços e o banco de dados são conteinerizados.

1.  Abra o terminal na pasta `backend_tcc`.
2.  Execute o comando para construir e subir os containers:
    ```bash
    docker-compose up --build
    ```
3.  Aguarde até que o log indique que o `tcc_postgres` está saudável (Healthy) e os serviços (Uvicorn) iniciaram.
    * *Nota:* O banco de dados será criado automaticamente na primeira execução.

### Passo 2: Simular o Hardware (Trem)
Se você não tiver o Raspberry Pi físico, pode usar o script cliente para simular um trem enviando dados.

1.  Abra um novo terminal na pasta `hardware` (ou onde está o script).
2.  Execute:
    ```bash
    python trem_client.py
    ```
    *Isso começará a enviar dados de telemetria fictícios para o `iot_service`.*

### Passo 3: Rodar o Aplicativo Android

1.  Abra a pasta `mobile` (ou a raiz do projeto Android) no **Android Studio**.
2.  **Aguarde o Gradle Sync:** Na primeira vez, o Android Studio baixará bibliotecas externas (Mapas, Gráficos, Retrofit). Isso pode levar alguns minutos.
3.  **Configuração de IP (CRUCIAL):**
    * Como o Backend roda localmente no seu PC, o celular precisa saber o IP da sua máquina.
    * Abra o arquivo `app/java/com.seutcc.app/network/RetrofitClient.java`.
    * Descubra o IPv4 do seu computador (comando `ipconfig` ou `ifconfig`).
    * Atualize a variável `BASE_URL`:
        ```java
        private static final String BASE_URL = "http://SEU_IPV4_AQUI"; 
        // Ex: "[http://192.168.15.10](http://192.168.15.10)" (Mantenha sem a porta e sem barra no final)
        ```
4.  Conecte seu celular via USB (ou use o Emulador) e clique em **Run (Play)**.

---

## 📱 Funcionalidades do App

1.  **Cadastro/Login:** Crie um usuário para acessar o sistema.
2.  **Seleção de Linha:** Visualize as linhas disponíveis na cidade.
3.  **Detalhes da Estação:** Ao selecionar uma estação, o sistema calcula qual trem está mais próximo e exibe:
    * Tempo estimado de chegada (Minutos e Segundos).
    * Nível de lotação do vagão (Verde/Amarelo/Vermelho).
    * *Fallback:* Se o GPS falhar, o sistema avisa que a localização é estimada.
4.  **Reportar Problema:** Botão para enviar ocorrências (atraso, sujeira, segurança).
5.  **Dashboard:** Gráfico de barras mostrando o volume de ocorrências nas últimas 24h.

---

## 📂 Estrutura do Repositório

```text
/
├── backend_tcc/          # Código Fonte dos Microsserviços e Docker
│   ├── auth_service/     # Autenticação
│   ├── iot_service/      # Lógica de Trens e Estações
│   ├── report_service/   # Lógica de Reports
│   └── docker-compose.yml
│
├── mobile/               # Código Fonte Android
│   ├── app/src/main/java # Classes Java (Activities, Models, Adapters)
│   └── app/src/main/res  # Layouts XML
│
└── hardware/             # Scripts Python para o Raspberry Pi
    └── trem_client.py
