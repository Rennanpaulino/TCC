from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session
from database import Base, engine, get_db

# 1. Cria as tabelas no Banco (Modelo)
class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

# Cria a tabela se não existir
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Auth Service")

# Modelo para validar dados que vêm da requisição (Pydantic)
class UserSchema(BaseModel):
    username: str
    password: str

# --- ROTAS ---

@app.post("/register")
def register(user: UserSchema, db: Session = Depends(get_db)):
    # Verifica se já existe no banco
    db_user = db.query(UserModel).filter(UserModel.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Usuário já existe")
    
    # Cria novo usuário
    new_user = UserModel(username=user.username, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"msg": "Usuário criado com sucesso", "id": new_user.id}

@app.post("/login")
def login(user: UserSchema, db: Session = Depends(get_db)):
    # Busca no banco
    db_user = db.query(UserModel).filter(UserModel.username == user.username).first()
    
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    
    return {
        "access_token": f"token-real-do-{db_user.username}",
        "token_type": "bearer"
    }