package com.seutcc.app.models;
import com.google.gson.annotations.SerializedName;

public class Linha {
    @SerializedName("id")
    private String id;

    @SerializedName("nome")
    private String nome;

    @SerializedName("cor")
    private String corHex; // Ex: "#00A88E"

    public String getId() { return id; }
    public String getNome() { return nome; }
    public String getCorHex() { return corHex; }
}