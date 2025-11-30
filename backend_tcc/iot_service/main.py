from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import Session
from datetime import datetime
import math

# Importa do seu arquivo database.py
from database import Base, engine, get_db

# ==========================================
# 1. DADOS ESTÁTICOS (TOPOLOGIA DA REDE)
# ==========================================
# No TCC, isso substitui um banco de dados complexo de rotas.
# Adicionei coordenadas reais de algumas estações de SP.

TOPOLOGIA = {
    "LINHA-9": {
        "nome": "Linha 9 - Esmeralda",
        "cor": "#00A88E",
        "estacoes": [
            {"id": "osasco", "nome": "Osasco", "lat": -23.5323, "lon": -46.7725},
            {"id": "pinheiros", "nome": "Pinheiros", "lat": -23.5673, "lon": -46.7023},
            {"id": "morumbi", "nome": "Morumbi", "lat": -23.6231, "lon": -46.7028},
            {"id": "sto_amaro", "nome": "Santo Amaro", "lat": -23.6563, "lon": -46.7095},
            {"id": "grajau", "nome": "Grajaú", "lat": -23.7584, "lon": -46.6932}
        ]
    },
    "LINHA-4": {
        "nome": "Linha 4 - Amarela",
        "cor": "#FFD500",
        "estacoes": [
            {"id": "luz", "nome": "Luz", "lat": -23.5365, "lon": -46.6358},
            {"id": "paulista", "nome": "Paulista", "lat": -23.5551, "lon": -46.6622},
            {"id": "butanta", "nome": "Butantã", "lat": -23.5718, "lon": -46.7081}
        ]
    }
}

# Velocidade média de contingência (caso GPS falhe ou trem esteja parado)
VELOCIDADE_MEDIA_TREM = 60.0 

# ==========================================
# 2. MODELO DE BANCO (Posição dos Trens)
# ==========================================
class TremModel(Base):
    __tablename__ = "trens"
    id = Column(Integer, primary_key=True, index=True)
    trem_id = Column(String, unique=True, index=True) 
    linha_id = Column(String) # Ex: "LINHA-9" (Para saber de qual linha ele é)
    latitude = Column(Float)
    longitude = Column(Float)
    lotacao = Column(Integer)
    velocidade = Column(Float)
    ultima_atualizacao = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="IoT Service - TCC")

# ==========================================
# 3. SCHEMAS (Comunicação)
# ==========================================

# Entrada (Raspberry Pi)
class TelemetriaInput(BaseModel):
    trem_id: str
    linha_id: str  # <-- Importante: O Raspberry tem que dizer de qual linha ele é
    latitude: float
    longitude: float
    lotacao: int
    velocidade: float

# Saída 1: Lista de Linhas
class LinhaResponse(BaseModel):
    id: str
    nome: str
    cor: str

# Saída 2: Lista de Estações
class EstacaoResponse(BaseModel):
    id: str
    nome: str
    lat: float
    lon: float

# Saída 3: Previsão Detalhada
class PrevisaoResponse(BaseModel):
    estacao_destino: str
    trem_id: str
    distancia_km: float
    eta_minutos: int
    eta_segundos: int
    lotacao: int
    velocidade_ref: float
    is_estimado: bool
    msg: str = "Operação Normal"

# ==========================================
# 4. LÓGICA MATEMÁTICA
# ==========================================
def calcular_haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Raio da terra
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ==========================================
# 5. ENDPOINTS
# ==========================================

# --- ROTA 1: Receber Dados (Raspberry) ---
@app.post("/telemetria")
def receber_dados(dados: TelemetriaInput, db: Session = Depends(get_db)):
    trem = db.query(TremModel).filter(TremModel.trem_id == dados.trem_id).first()
    timestamp = datetime.now()
    
    if trem:
        trem.linha_id = dados.linha_id
        trem.latitude = dados.latitude
        trem.longitude = dados.longitude
        trem.lotacao = dados.lotacao
        trem.velocidade = dados.velocidade
        trem.ultima_atualizacao = timestamp
    else:
        trem = TremModel(
            trem_id=dados.trem_id,
            linha_id=dados.linha_id,
            latitude=dados.latitude,
            longitude=dados.longitude,
            lotacao=dados.lotacao,
            velocidade=dados.velocidade,
            ultima_atualizacao=timestamp
        )
        db.add(trem)
    
    db.commit()
    return {"status": "ok"}

# --- ROTA 2: Listar Linhas (Tela Home) ---
@app.get("/linhas", response_model=list[LinhaResponse])
def get_linhas():
    lista = []
    for k, v in TOPOLOGIA.items():
        lista.append({"id": k, "nome": v["nome"], "cor": v["cor"]})
    return lista

# --- ROTA 3: Listar Estações da Linha (Tela Estacoes) ---
@app.get("/estacoes/{linha_id}", response_model=list[EstacaoResponse])
def get_estacoes(linha_id: str):
    if linha_id in TOPOLOGIA:
        return TOPOLOGIA[linha_id]["estacoes"]
    return []

# --- ROTA 4: O CÁLCULO (Tela Detalhes) ---
@app.get("/previsao/{linha_id}/{estacao_id}", response_model=PrevisaoResponse)
def calcular_previsao(linha_id: str, estacao_id: str, db: Session = Depends(get_db)):
    
    # 1. Encontrar coordenadas da estação alvo
    estacao_alvo = None
    if linha_id in TOPOLOGIA:
        for est in TOPOLOGIA[linha_id]["estacoes"]:
            if est["id"] == estacao_id:
                estacao_alvo = est
                break
    
    if not estacao_alvo:
        raise HTTPException(status_code=404, detail="Estação não encontrada")

    # 2. Buscar trens DESSA linha no banco
    trens = db.query(TremModel).filter(TremModel.linha_id == linha_id).all()
    
    if not trens:
        # Retorna um objeto vazio/zerado se não tiver trem
        return {
            "estacao_destino": estacao_alvo["nome"],
            "trem_id": "Nenhum",
            "distancia_km": 0.0,
            "eta_minutos": 0, "eta_segundos": 0,
            "lotacao": 0, "velocidade_ref": 0.0,
            "is_estimado": False,
            "msg": "Nenhum trem circulando nesta linha"
        }

    # 3. Descobrir qual trem chega mais rápido (Menor ETA)
    melhor_eta_segundos = float('inf')
    melhor_trem = None
    distancia_final = 0.0
    
    agora = datetime.now()

    for t in trens:
        # Lógica 2.2: GPS Falho/Estimativa
        tempo_sem_sinal = (agora - t.ultima_atualizacao).total_seconds()
        is_gps_velho = tempo_sem_sinal > 30
        
        # Define velocidade de cálculo (usa média se parado ou sem sinal)
        vel_calculo = t.velocidade
        if is_gps_velho or vel_calculo < 1.0:
            vel_calculo = VELOCIDADE_MEDIA_TREM

        # Calcula Distância
        dist = calcular_haversine(t.latitude, t.longitude, estacao_alvo["lat"], estacao_alvo["lon"])
        
        # Calcula Tempo (Horas = km / km/h) -> Segundos
        tempo_s = (dist / vel_calculo) * 3600
        
        # É o trem mais próximo?
        if tempo_s < melhor_eta_segundos:
            melhor_eta_segundos = tempo_s
            melhor_trem = t
            distancia_final = dist
            # Se for o escolhido, guarda o status dele
            trem_escolhido_gps_velho = is_gps_velho 
            trem_escolhido_vel = vel_calculo

    # 4. Montar Resposta Final
    minutos = int(melhor_eta_segundos // 60)
    segundos = int(melhor_eta_segundos % 60)

    return {
        "estacao_destino": estacao_alvo["nome"],
        "trem_id": melhor_trem.trem_id,
        "distancia_km": round(distancia_final, 2),
        "eta_minutos": minutes,
        "eta_segundos": segundos,
        "lotacao": melhor_trem.lotacao,
        "velocidade_ref": trem_escolhido_vel,
        "is_estimado": trem_escolhido_gps_velho,
        "msg": "Previsão calculada com sucesso"
    }