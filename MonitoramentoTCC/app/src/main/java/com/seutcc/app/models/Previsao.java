package com.seutcc.app.models;
import com.google.gson.annotations.SerializedName;

public class Previsao {
    @SerializedName("estacao_destino")
    private String nomeEstacao;

    @SerializedName("trem_id")
    private String tremId;

    @SerializedName("eta_minutos")
    private int minutos;

    @SerializedName("eta_segundos")
    private int segundos;

    @SerializedName("distancia_km")
    private double distancia;

    @SerializedName("is_estimado")
    private boolean isEstimado;

    @SerializedName("msg")
    private String mensagem;

    // Getters
    public String getNomeEstacao() { return nomeEstacao; }
    public int getMinutos() { return minutos; }
    public int getSegundos() { return segundos; }
    public boolean isEstimado() { return isEstimado; }
    public String getMensagem() { return mensagem; }
}