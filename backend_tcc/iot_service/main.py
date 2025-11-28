from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import Session
from datetime import datetime
from database import Base, engine, get_db

# --- 1. MODELO DE BANCO DE DADOS (Tabela) ---
class TremModel(Base):
    __tablename__ = "trens"
    
    id = Column(Integer, primary_key=True, index=True)
    trem_id = Column(String, unique=True, index=True) # Ex: "TREM-01" (Único)
    latitude = Column(Float)
    longitude = Column(Float)
    lotacao = Column(Integer)
    velocidade = Column(Float)
    ultima_atualizacao = Column(DateTime, default=datetime.now)

# Cria a tabela no banco automaticamente
Base.metadata.create_all(bind=engine)

# --- 2. SCHEMAS (Pydantic - Validação de Dados) ---
class TelemetriaSchema(BaseModel):
    trem_id: str
    latitude: float
    longitude: float
    lotacao: int
    velocidade: float

app = FastAPI(title="IoT Service")

# --- 3. ROTAS ---

@app.post("/telemetria")
def receber_telemetria(dados: TelemetriaSchema, db: Session = Depends(get_db)):
    # Lógica de UPSERT (Update or Insert)
    
    # Busca se o trem já existe no banco
    trem_existente = db.query(TremModel).filter(TremModel.trem_id == dados.trem_id).first()
    
    if trem_existente:
        # SE EXISTE: Atualiza os dados
        trem_existente.latitude = dados.latitude
        trem_existente.longitude = dados.longitude
        trem_existente.lotacao = dados.lotacao
        trem_existente.velocidade = dados.velocidade
        trem_existente.ultima_atualizacao = datetime.now()
        msg = f"Dados do {dados.trem_id} atualizados."
    else:
        # SE NÃO EXISTE: Cria um novo registro
        novo_trem = TremModel(
            trem_id=dados.trem_id,
            latitude=dados.latitude,
            longitude=dados.longitude,
            lotacao=dados.lotacao,
            velocidade=dados.velocidade,
            ultima_atualizacao=datetime.now()
        )
        db.add(novo_trem)
        msg = f"Trem {dados.trem_id} cadastrado."
    
    db.commit() # Salva efetivamente no PostgreSQL
    return {"status": "sucesso", "mensagem": msg}

@app.get("/trens")
def listar_trens(db: Session = Depends(get_db)):
    # Retorna todos os trens cadastrados
    return db.query(TremModel).all()