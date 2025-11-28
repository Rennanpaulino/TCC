package com.seutcc.app.models;
import com.google.gson.annotations.SerializedName;

public class Trem {
    @SerializedName("trem_id") // Nome exato que vem do Python
    private String id;

    @SerializedName("lotacao")
    private int lotacao;

    @SerializedName("velocidade")
    private double velocidade;

    // Getters
    public String getId() { return id; }
    public int getLotacao() { return lotacao; }
    public double getVelocidade() { return velocidade; }
}