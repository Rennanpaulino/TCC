package com.seutcc.app.models;

import com.google.gson.annotations.SerializedName;

public class Trem {

    // Campos já existentes
    @SerializedName("trem_id")
    private String id;

    @SerializedName("latitude")
    private double latitude;

    @SerializedName("longitude")
    private double longitude;

    @SerializedName("lotacao")
    private int lotacao;

    @SerializedName("velocidade")
    private double velocidade;

    // --- NOVOS CAMPOS (Que o Backend atualizado envia) ---

    // Indica se o GPS falhou e estamos usando cálculo aproximado
    @SerializedName("is_estimado")
    private boolean isEstimado;

    // Tempo restante para chegar em Santo Amaro (Minutos)
    @SerializedName("eta_minutos")
    private int etaMinutos;

    // Tempo restante (Segundos)
    @SerializedName("eta_segundos")
    private int etaSegundos;

    // --- GETTERS (Necessários para o Adapter e Activity lerem) ---

    public String getId() {
        return id;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    public int getLotacao() {
        return lotacao;
    }

    public double getVelocidade() {
        return velocidade;
    }

    // Getters dos Novos Campos
    public boolean isEstimado() {
        return isEstimado;
    }

    public int getEtaMinutos() {
        return etaMinutos;
    }

    public int getEtaSegundos() {
        return etaSegundos;
    }
}