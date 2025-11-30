from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from database import Base, engine, get_db
from sqlalchemy import Column, Integer, String

# --- CONFIGURAÇÕES ---
SECRET_KEY = "segredo_do_tcc" # Em produção use env var
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- MODELO DB ---
class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role = Column(String, default="user") # user ou admin

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Auth Service")

# --- SCHEMAS ---
class UserLogin(BaseModel):
    username: str
    password: str

# --- UTILITÁRIOS ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=60) # Token dura 1h
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- ROTAS ---
@app.post("/register")
def register(user: UserLogin, db: Session = Depends(get_db)):
    if db.query(UserModel).filter(UserModel.username == user.username).first():
        raise HTTPException(status_code=400, detail="Usuário já existe")
    
    new_user = UserModel(
        username=user.username,
        password_hash=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    return {"msg": "Criado com sucesso"}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(UserModel).filter(UserModel.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    
    token = create_token({"sub": db_user.username, "role": db_user.role})
    return {"access_token": token, "token_type": "bearer"}