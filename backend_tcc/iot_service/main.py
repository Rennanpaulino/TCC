from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import Session
from datetime import datetime
import math
from database import Base, engine, get_db

# --- CONFIGURAÇÕES ---
# Coordenadas fixas da Estação Santo Amaro
DESTINO_LAT = -23.65637
DESTINO_LON = -46.70956
VELOCIDADE_MEDIA_TREM = 60.0 # km/h (fallback se GPS falhar)

# --- MODELO DB ---
class TremModel(Base):
    __tablename__ = "trens"
    id = Column(Integer, primary_key=True)
    trem_id = Column(String, unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    lotacao = Column(Integer)
    velocidade = Column(Float) # Em km/h
    ultima_atualizacao = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="IoT Service")

# --- SCHEMA DE RESPOSTA (O que o Android recebe) ---
class TremResponse(BaseModel):
    trem_id: str
    latitude: float
    longitude: float
    lotacao: int
    velocidade: float
    is_estimado: bool  # <-- NOVO: Avisa se o GPS falhou
    eta_minutos: int   # <-- NOVO: Tempo para chegar
    eta_segundos: int  # <-- NOVO

class TelemetriaInput(BaseModel):
    trem_id: str
    latitude: float
    longitude: float
    lotacao: int
    velocidade: float

# --- FUNÇÃO MATEMÁTICA (Haversine) ---
def calcular_distancia_km(lat1, lon1, lat2, lon2):
    R = 6371 # Raio da Terra em km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- ROTAS ---
@app.post("/telemetria")
def receber_dados(dados: TelemetriaInput, db: Session = Depends(get_db)):
    trem = db.query(TremModel).filter(TremModel.trem_id == dados.trem_id).first()
    
    if not trem:
        trem = TremModel(trem_id=dados.trem_id)
        db.add(trem)
    
    trem.latitude = dados.latitude
    trem.longitude = dados.longitude
    trem.lotacao = dados.lotacao
    trem.velocidade = dados.velocidade
    trem.ultima_atualizacao = datetime.now()
    
    db.commit()
    return {"status": "atualizado"}

@app.get("/trens", response_model=list[TremResponse])
def listar_trens(db: Session = Depends(get_db)):
    trens_db = db.query(TremModel).all()
    lista_resposta = []
    agora = datetime.now()

    for t in trens_db:
        # LÓGICA 2.2: Verificar se dados são velhos (Estimativa)
        tempo_sem_sinal = (agora - t.ultima_atualizacao).total_seconds()
        is_estimado = False
        
        # Se não recebe dados há mais de 30 segundos, considera "GPS Perdido"
        if tempo_sem_sinal > 30:
            is_estimado = True
            # Estimativa simples: Mantém a última posição conhecida (para não jogar o trem no mar)
            # Mas avisa o Front que é estimado
            # Se quiser avançar: projetar a posição na direção de Sto Amaro seria geometria complexa
            # Para TCC, marcar como "Estimado" e usar vel. média já é suficiente.
            velocidade_calculo = VELOCIDADE_MEDIA_TREM
        else:
            velocidade_calculo = t.velocidade if t.velocidade > 0 else 1.0 # Evita div por zero

        # LÓGICA 2.1: Calcular ETA (Tempo de Chegada)
        distancia_km = calcular_distancia_km(t.latitude, t.longitude, DESTINO_LAT, DESTINO_LON)
        
        # Tempo (horas) = Distancia / Velocidade
        tempo_horas = distancia_km / velocidade_calculo
        segundos_totais = int(tempo_horas * 3600)
        
        eta_min = segundos_totais // 60
        eta_seg = segundos_totais % 60

        lista_resposta.append({
            "trem_id": t.trem_id,
            "latitude": t.latitude,
            "longitude": t.longitude,
            "lotacao": t.lotacao,
            "velocidade": velocidade_calculo,
            "is_estimado": is_estimado,
            "eta_minutos": eta_min,
            "eta_segundos": eta_seg
        })

    return lista_resposta