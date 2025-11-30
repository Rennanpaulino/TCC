package com.seutcc.app.models;
import com.google.gson.annotations.SerializedName;

public class Report {
    private String usuario;
    @SerializedName("linha_id") private String linhaId;
    @SerializedName("estacao_id") private String estacaoId;
    @SerializedName("tipo_problema") private String tipoProblema;

    public Report(String u, String l, String e, String t) {
        this.usuario = u; this.linhaId = l; this.estacaoId = e; this.tipoProblema = t;
    }
}