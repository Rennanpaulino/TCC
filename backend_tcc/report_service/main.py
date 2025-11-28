from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import Session
from datetime import datetime
from database import Base, engine, get_db

# --- 1. MODELO DE BANCO DE DADOS ---
class ReportModel(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario = Column(String) # Nome ou ID do usuário que denunciou
    trem_id = Column(String) # Em qual trem ele estava
    tipo_problema = Column(String) # Ex: "Limpeza", "Segurança"
    descricao = Column(Text)       # Detalhes opcionais
    data_criacao = Column(DateTime, default=datetime.now)

# Cria a tabela
Base.metadata.create_all(bind=engine)

# --- 2. SCHEMAS ---
class ReportCreateSchema(BaseModel):
    usuario: str
    trem_id: str
    tipo_problema: str
    descricao: str = "" # Opcional

class ReportResponseSchema(ReportCreateSchema):
    id: int
    data_criacao: datetime
    class Config:
        orm_mode = True

app = FastAPI(title="Report Service")

# --- 3. ROTAS ---

@app.post("/report", response_model=ReportResponseSchema)
def criar_report(report: ReportCreateSchema, db: Session = Depends(get_db)):
    # Cria o objeto do modelo
    novo_report = ReportModel(
        usuario=report.usuario,
        trem_id=report.trem_id,
        tipo_problema=report.tipo_problema,
        descricao=report.descricao,
        data_criacao=datetime.now()
    )
    
    # Salva no banco
    db.add(novo_report)
    db.commit()
    db.refresh(novo_report) # Atualiza o objeto com o ID gerado pelo banco
    
    return novo_report

@app.get("/reports")
def listar_reports(db: Session = Depends(get_db)):
    # Retorna todos os reports ordenados por data (mais recentes primeiro)
    return db.query(ReportModel).order_by(ReportModel.data_criacao.desc()).all()