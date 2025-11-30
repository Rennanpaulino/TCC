package com.seutcc.app.models;
import com.google.gson.annotations.SerializedName;

public class Estacao {
    @SerializedName("id")
    private String id;

    @SerializedName("nome")
    private String nome;

    public String getId() { return id; }
    public String getNome() { return nome; }
}