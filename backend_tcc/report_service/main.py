from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Text, func
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from database import Base, engine, get_db

class ReportModel(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    usuario = Column(String)
    linha_id = Column(String)   # <-- MUDOU: Agora guardamos a linha
    estacao_id = Column(String) # <-- NOVO: Guardamos a estação
    tipo_problema = Column(String)
    data_criacao = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Report Service")

class ReportInput(BaseModel):
    usuario: str
    linha_id: str
    estacao_id: str
    tipo_problema: str

# --- ROTA DE CRIAÇÃO ---
@app.post("/report")
def criar_report(report: ReportInput, db: Session = Depends(get_db)):
    novo = ReportModel(
        usuario=report.usuario,
        linha_id=report.linha_id,
        estacao_id=report.estacao_id,
        tipo_problema=report.tipo_problema,
        data_criacao=datetime.now()
    )
    db.add(novo)
    db.commit()
    return {"msg": "Reportado com sucesso"}

# --- ROTA PARA O GRÁFICO (Estatísticas) ---
@app.get("/reports/stats")
def estatisticas(db: Session = Depends(get_db)):
    # Queremos saber: Quantos reports por HORA nas últimas 24h?
    
    limite = datetime.now() - timedelta(hours=24)
    
    # Query SQL traduzida para SQLAlchemy:
    # SELECT date_part('hour', data_criacao), count(*) 
    # FROM reports WHERE data > limite GROUP BY hour
    
    stats = db.query(
        func.extract('hour', ReportModel.data_criacao).label('hora'),
        func.count(ReportModel.id).label('total')
    ).filter(ReportModel.data_criacao >= limite)\
     .group_by('hora').all()
    
    # Formata para JSON simples: { "10": 5, "11": 2 } (Hora 10 teve 5 reports)
    resultado = {int(r.hora): r.total for r in stats}
    return resultado

# --- ROTA LISTA SIMPLES ---
@app.get("/reports")
def listar_todos(db: Session = Depends(get_db)):
    return db.query(ReportModel).order_by(ReportModel.data_criacao.desc()).limit(50).all()